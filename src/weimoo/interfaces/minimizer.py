from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch





class Minimizer(ABC):
    @abstractmethod
    def __call__(
        self,
        function: Callable,
        minimizer_workers: int,
        fcn_args: tuple,
        upper_bounds: np.ndarray,
        lower_bounds: np.ndarray,
        max_iter: int,
    ) -> np.ndarray:
        # x value; not f(x) (stored in function)
        pass

    def get_torch_prediction(self, x: np.ndarray, model) -> float:
        return -model.predict_torch(torch.Tensor([x.tolist()])).mean.numpy()[0]