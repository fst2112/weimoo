import numpy as np
import plotly.graph_objects as go
from pymoo.factory import get_problem

from weimoo.moos.gpr_multiple_weight_based_moo import GPRMultipleWeightsBasedMOO
from weimoo.moos.helper_functions.return_pareto_front_2d import (
    return_pareto_front_2d,
)
from weimoo.interfaces.function import Function
from weimoo.minimizers.differential_evolution import DifferentialEvolution
from weimoo.weight_functions.scalar_potency import ScalarPotency

input_dimensions = 20
output_dimensions = 2

lower_bounds_x = np.zeros(input_dimensions)
upper_bounds_x = np.ones(input_dimensions)

minimizer = DifferentialEvolution()

max_iter_minimizer = 100
max_evaluations_per_weight = 5

problem = get_problem("dtlz2", n_var=input_dimensions, n_obj=output_dimensions)


class ExampleFunction(Function):
    def __call__(self, x):
        self._evaluations.append([x, problem.evaluate(x)])
        return problem.evaluate(x)


# Initialiaze the function
function = ExampleFunction()

# Initialize weight function
weight_function_1 = ScalarPotency(
    potency=2 * np.ones(output_dimensions), scalar=np.array([1, 0.2])
)

weight_function_2 = ScalarPotency(
    potency=2 * np.ones(output_dimensions), scalar=np.array([0.2, 1])
)

weight_function_3 = ScalarPotency(
    potency=2 * np.ones(output_dimensions), scalar=np.array([1, 1])
)


MOO = GPRMultipleWeightsBasedMOO(
    weight_functions=[weight_function_1, weight_function_2, weight_function_3]
)

result = MOO(
    function=function,
    minimizer=minimizer,
    upper_bounds=upper_bounds_x,
    lower_bounds=lower_bounds_x,
    number_designs_LH=7 * max_evaluations_per_weight,
    max_evaluations_per_weight=max_evaluations_per_weight,
    max_iter_minimizer=max_iter_minimizer,
    training_iter=5000,
)

real_PF = problem.pareto_front()

PF = return_pareto_front_2d([point[1] for point in function.evaluations])

data = [
    go.Scatter(x=PF.T[0], y=PF.T[1], mode="markers"),
    go.Scatter(x=real_PF.T[0], y=real_PF.T[1], mode="markers"),
    go.Scatter(
        x=np.array([function(result[0])[0]]),
        y=np.array([function(result[0])[1]]),
        mode="markers",
    ),
    go.Scatter(
        x=np.array([function(result[1])[0]]),
        y=np.array([function(result[1])[1]]),
        mode="markers",
    ),
]

fig = go.Figure(data=data)
# fig.show()

from pymoo.indicators.hv import Hypervolume

reference_point = np.array([2, 2])
real_PF = problem.pareto_front()

metric = Hypervolume(ref_point=reference_point, normalize=False)

hypervolume_max = metric.do(problem.pareto_front())
hypervolume_weight = metric.do(PF)

print(hypervolume_weight / hypervolume_max)

y = np.array([evaluation[1] for evaluation in function.evaluations])

data = [
    go.Scatter(x=real_PF.T[0], y=real_PF.T[1], mode="markers"),
    go.Scatter(x=y.T[0], y=y.T[1], mode="markers"),
    go.Scatter(x=PF.T[0], y=PF.T[1], mode="markers"),
]

fig1 = go.Figure(data=data)

fig1.update_layout(
    width=800,
    height=600,
    plot_bgcolor="rgba(0,0,0,0)",
    title=f"GPR multiple weight based MOO: relative Hypervolume: {hypervolume_weight / hypervolume_max * 100}%",
)

fig1.show()
