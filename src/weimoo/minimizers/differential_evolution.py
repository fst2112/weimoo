from typing import Callable

import numpy as np
import torch
from scipy.optimize import differential_evolution

from weimoo.interfaces.minimizer import Minimizer


class DifferentialEvolution(Minimizer):
    def __init__(self, display=False):
        self.display = display
        self._number_evaluations_last_call = None

    def __call__(
        self,
        function: Callable,
        minimizer_workers: int,
        fcn_args: tuple,
        upper_bounds: np.ndarray,
        lower_bounds: np.ndarray,
        max_iter: int = 1000,
    ) -> np.ndarray:
        t_initial = (upper_bounds + lower_bounds) / 2
        # print(f'{fcn_args=}')
        #
        # fun = lambda x: -fcn_args.predict_torch(torch.Tensor([x.tolist()])).mean.numpy()[0]
        #
        # print(f'{fun(t_initial)=}')
        #
        # print(f'{t_initial=}')
        # print(f'{function=}')
        # print(f'{function(t_initial, fcn_args)=}')

        res = differential_evolution(
            func=function,
            args=fcn_args,
            workers=minimizer_workers,
            x0=t_initial,
            disp=self.display,
            tol=1e-5,
            bounds=[
                (lower_bounds[i], upper_bounds[i]) for i in range(len(lower_bounds))
            ],
            maxiter=max_iter,
        )

        self.result = res
        self._number_evaluations_last_call = res.nfev
        return res.x

    @property
    def number_evaluations_last_call(self):
        return self._number_evaluations_last_call
