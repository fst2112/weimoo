import numpy as np
import torch
from scipy.stats import qmc
from tqdm import tqdm

from weimoo.interfaces.function import Function
from weimoo.interfaces.minimizer import Minimizer, get_torch_prediction, minimizer_objective_fun
from weimoo.moos.helper_functions.ehvi_2d import ehvi_2d
from weimoo.moos.helper_functions.return_pareto_front_2d import (
    return_pareto_front_2d,
)
from weimoo.surrogates.gpr import GPR


class EHVI2dAdaptedReferencePointMOO:
    def __call__(
            self,
            function: Function,
            minimizer: Minimizer,
            minimizer_workers: int,
            upper_bounds: np.ndarray,
            lower_bounds: np.ndarray,
            number_designs_LH: int,
            max_evaluations: int,
            max_iter_minimizer: 1000,
            training_iter: int = 1000,
            learning_rate=0.01,
            tolerance_reference_point: float = 1e-3,
    ) -> np.ndarray:
        function.clear_evaluations()
        # Setting up a LH for the seed
        train_x = qmc.scale(
            qmc.LatinHypercube(d=len(lower_bounds)).random(n=number_designs_LH),
            lower_bounds,
            upper_bounds,
        )
        evaluations = np.array([function(x) for x in train_x])
        gpr = GPR(training_iter=training_iter, learning_rate=learning_rate)
        # define criteria for normalization is final
        all_maxima = []
        all_minima = []
        for i in range(max_evaluations - number_designs_LH):
            print(
                f"{i + 1}/{max_evaluations - number_designs_LH} Training of the GPR...\n"
            )
            gpr.train(train_x=train_x, train_y=evaluations)
            print(f"\n finished!\n")

            print(f"\n Calculating maxima of components...\n")
            maxima = []
            for model in tqdm(gpr._models):
                x = minimizer(
                    function=get_torch_prediction,
                    minimizer_workers=minimizer_workers,
                    fcn_args=(model,),
                    max_iter=max_iter_minimizer,
                    upper_bounds=upper_bounds,
                    lower_bounds=lower_bounds,
                )

                maxima.append(
                    model.predict_torch(torch.Tensor([x.tolist()])).mean.numpy()[0]
                )
            maxima = np.array(maxima)
            # norming and 1+1/1-n

            print(f"\n Maxima found: {maxima}\n")

            max = np.max(maxima)

            print(f"\n Reference point is {max * np.ones(len(maxima))}...")

            print(f"\nStarting minimization...")
            pf = return_pareto_front_2d(evaluations)

            sub_models = [m._model for m in gpr._models]
            sub_hoods = [m._likelihood for m in gpr._models]

            for m, h in zip(sub_models, sub_hoods):
                m.eval()
                h.eval()
            res = minimizer(
                function=minimizer_objective_fun,
                minimizer_workers=minimizer_workers,
                fcn_args=(ehvi_2d, pf, max, maxima, sub_hoods),
                max_iter=max_iter_minimizer,
                upper_bounds=upper_bounds,
                lower_bounds=lower_bounds
            )

            train_x = np.append(train_x, res.reshape(1, train_x.shape[1]), axis=0)
            evaluations = np.append(
                evaluations, function(res).reshape(1, evaluations.shape[1]), axis=0
            )

            print(f"\n finished!")

        return evaluations
