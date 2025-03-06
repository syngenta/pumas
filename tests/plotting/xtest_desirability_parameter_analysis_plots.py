# type: ignore
import pytest

from pumas.plotting.desirability_parameter_analysis import (
    plot_bell_analysis,
    plot_double_sigmoid_analysis,
    plot_leftstep_analysis,
    plot_multistep_analysis,
    plot_rightstep_analysis,
    plot_sigmoid_analysis,
    plot_sigmoid_bell_analysis,
    plot_step_analysis,
)
from pumas.plotting.plotter import MATPLOTLIB_AVAILABLE


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib is not available")
@pytest.mark.parametrize(
    "func",
    [
        plot_sigmoid_analysis,
        plot_double_sigmoid_analysis,
        plot_sigmoid_bell_analysis,
        plot_bell_analysis,
        plot_multistep_analysis,
        plot_leftstep_analysis,
        plot_rightstep_analysis,
        plot_step_analysis,
    ],
)
def test_create_parameter_plot(func):
    try:
        func()
        assert True, "create_parameter_plot() executed successfully"
    except Exception as e:
        assert False, f"create_parameter_plot() raised an unexpected exception: {e}"
