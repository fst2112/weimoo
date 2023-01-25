from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution

from weimoo.interfaces.minimizer import Minimizer


class DifferentialEvolution(Minimizer):
    def __init__(self, display=False):
        self.display = display
        self._number_evaluations_last_call = None

    def __call__(
            self,
            function: Callable,
            upper_bounds: np.ndarray,
            lower_bounds: np.ndarray,
            max_iter: int = 1000,
            fcn_args: tuple = None,
            minimizer_workers: int = 1,
    ) -> np.ndarray:
        t_initial = (upper_bounds + lower_bounds) / 2

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
