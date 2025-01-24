import pytest

from pumas.plotting.plotter import MATPLOTLIB_AVAILABLE, plot_parameter_analysis


# Mock function for testing
def mock_func(x, a, b):
    return a * x + b


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib is not available")
def test_parameter_analysis_save(tmp_path):
    parameters = [
        {"parameter_name": "a", "title": "Parameter A", "values": [1, 2, 3]},
        {"parameter_name": "b", "title": "Parameter B", "values": [0, 1, 2]},
    ]
    reference_coefficient_parameters = {"a": 1, "b": 0}
    save_path = tmp_path / "test_plot_high_res.png"
    test_dpi = 600

    plot_parameter_analysis(
        func=mock_func,
        parameters=parameters,
        reference_coefficient_parameters=reference_coefficient_parameters,
        vertical_lines_x_values=[2, 5],
        x_range=(0, 10),
        figsize=(10, 5),
        save_path=str(save_path),
        size_dpi=test_dpi,
    )

    assert save_path.exists(), f"Plot was not saved to {save_path}"
    assert save_path.stat().st_size > 0, f"Saved plot file {save_path} is empty"

    # Optional: Check file format (requires Pillow)
    from PIL import Image

    with Image.open(save_path) as img:
        assert img.format == "PNG", "Saved file is not in PNG format"


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib is not available")
def test_parameter_analysis_display(monkeypatch):
    # Mock plt.show to avoid actually displaying the plot
    show_called = False

    def mock_show():
        nonlocal show_called
        show_called = True

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", mock_show)

    parameters = [
        {"parameter_name": "a", "title": "Parameter A", "values": [1, 2, 3]},
        {"parameter_name": "b", "title": "Parameter B", "values": [0, 1, 2]},
    ]
    reference_coefficient_parameters = {"a": 1, "b": 0}

    plot_parameter_analysis(
        func=mock_func,
        parameters=parameters,
        reference_coefficient_parameters=reference_coefficient_parameters,
        vertical_lines_x_values=[2, 5],
        x_range=(0, 10),
        figsize=(10, 5),
    )

    assert show_called, "plt.show() was not called when save_path is None"


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib is not available")
def test_parameter_analysis_single_subplot():
    parameters = [
        {"parameter_name": "a", "title": "Parameter A", "values": [1, 2, 3]},
    ]
    reference_coefficient_parameters = {"a": 1, "b": 0}

    # This should not raise an exception
    plot_parameter_analysis(
        func=mock_func,
        parameters=parameters,
        reference_coefficient_parameters=reference_coefficient_parameters,
        vertical_lines_x_values=[2, 5],
        x_range=(0, 10),
        figsize=(10, 5),
    )


@pytest.mark.skipif(not MATPLOTLIB_AVAILABLE, reason="Matplotlib is not available")
def test_parameter_analysis_save_failure(tmp_path, monkeypatch):
    def mock_savefig(*args, **kwargs):
        raise IOError("Mocked save failure")

    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt.Figure, "savefig", mock_savefig)

    parameters = [
        {"parameter_name": "a", "title": "Parameter A", "values": [1, 2, 3]},
    ]
    reference_coefficient_parameters = {"a": 1, "b": 0}
    save_path = tmp_path / "test_plot_failure.png"

    # This should not raise an exception, but print an error message
    plot_parameter_analysis(
        func=mock_func,
        parameters=parameters,
        reference_coefficient_parameters=reference_coefficient_parameters,
        vertical_lines_x_values=[2, 5],
        x_range=(0, 10),
        figsize=(10, 5),
        save_path=str(save_path),
    )

    assert not save_path.exists(), "File should not be created when save fails"


def test_parameter_analysis_no_matplotlib(monkeypatch):
    import pumas.plotting.plotter as pa

    monkeypatch.setattr(pa, "MATPLOTLIB_AVAILABLE", False)

    with pytest.raises(ImportError, match="Matplotlib is required for plotting"):
        plot_parameter_analysis(
            func=mock_func,
            parameters=[],
            reference_coefficient_parameters={},
            vertical_lines_x_values=[],
        )
