from abc import ABC, abstractmethod
from typing import Callable
import gpytorch

import numpy as np
import torch


def get_torch_prediction(x: np.ndarray, *args) -> float:
    (model,) = args
    return -model.predict_torch(torch.Tensor([x.tolist()])).mean.numpy()[0]


def minimizer_objective_fun(x: np.ndarray, *args) -> float:
    ehvi_2d, pf, max, maxima, sub_hoods = args
    x = torch.Tensor([x.tolist()])
    mu = []
    sigma = []

    for h in sub_hoods:
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean_module = gpytorch.means.ConstantMean()
            mean_x = mean_module(torch.Tensor([x.tolist()]))
            covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
            covar_x = covar_module(x)
            observed_pred = h(gpytorch.distributions.MultivariateNormal(mean_x, covar_x))
            mu.append(observed_pred.mean.detach().numpy()[0])
            sigma.append(observed_pred.stddev.detach().numpy()[0])
    mu = np.array(mu).T
    sigma = np.array(sigma).T
    return -ehvi_2d(pf, max * np.ones(len(maxima)), mu, sigma)


class Minimizer(ABC):
    @abstractmethod
    def __call__(
            self,
            function: Callable,
            upper_bounds: np.ndarray,
            lower_bounds: np.ndarray,
            max_iter: int,
            fcn_args: tuple = None,
            minimizer_workers: int = 1
    ) -> np.ndarray:
        # x value; not f(x) (stored in function)
        pass
