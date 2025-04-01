# type: ignore
from typing import cast

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    plt = None
    Axes = None


def plot_subplot(
    desirability_class,
    ax: Axes,
    x_values,
    param_cases,
    param_name,
    title,
    reference_coefficient_parameters: dict,
    vertical_lines_x_values: list[float] = None,
    points: list[tuple[float, float]] = None,
    plot_reference: bool = True,
):
    ax.set_xlim([min(x_values), max(x_values)])
    ax.set_ylim([-0.1, 1.1])
    ax.set_xlabel("Property Value", fontsize=14)
    ax.set_ylabel("Desirability Value", fontsize=14)
    ax.set_title(title, fontsize=16)

    # Plot reference
    if plot_reference:
        y_values_ref = np.array(
            [
                desirability_class(
                    params=reference_coefficient_parameters
                ).compute_numeric(x=xi)
                for xi in x_values
            ]
        )
        ref_def = f"Ref.: ({param_name}={reference_coefficient_parameters[param_name]})"

        ax.plot(
            x_values,
            y_values_ref,
            label=ref_def,
            linewidth=3,
            color="black",
            linestyle="dashed",
        )

    # plot the different cases
    for param_value in param_cases:
        coefficient_parameters = reference_coefficient_parameters.copy()
        coefficient_parameters[param_name] = param_value

        y_values = np.array(
            [
                desirability_class(params=coefficient_parameters).compute_numeric(x=xi)
                for xi in x_values
            ]
        )
        ax.plot(x_values, y_values, label=f"{param_name}={param_value}", linewidth=4)

    # Plot points
    if points:
        for coord in points:
            ax.plot(coord[0], coord[1], "ro", markersize=10)
            ax.annotate(str(coord), coord, xytext=(10, 10), textcoords="offset points")
    ax.legend(fontsize=12)
    if vertical_lines_x_values:
        for k in vertical_lines_x_values:
            ax.axvline(x=k, color="black", linestyle="dotted")

    for k in [0, 1]:
        ax.axhline(
            y=k,
            color="black",
        )


def plot_parameter_analysis(
    desirability_class,
    parameters: list,
    reference_coefficient_parameters: dict,
    vertical_lines_x_values: list[float] = None,
    points: list[tuple[float, float]] = None,
    x_range: tuple = (1, 100),
    figsize: tuple = (18, 12),
    save_path: str = None,
    size_dpi: int = 300,
    plot_reference: bool = True,
) -> None:
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "Matplotlib is required for plotting. "
            "Install it with 'pip install matplotlib'."
        )

    x_values = np.linspace(start=x_range[0], stop=x_range[1], num=200)
    alphabet = "abcdefghijklmnopqrstuvwxyz"

    nrows = (len(parameters) + 1) // 2  # Calculate number of rows
    ncols = min(2, len(parameters))  # Use 2 columns unless there's only 1 parameter

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    if nrows == 1 and ncols == 1:
        axs = np.array([axs])  # Make it 2D for consistent indexing

    for i, param in enumerate(parameters):
        row = i // 2
        col = i % 2
        title = f"({alphabet[i]}) {param['title']}"
        current_ax = cast(Axes, axs[row, col] if nrows > 1 else axs[col])
        plot_subplot(
            desirability_class,
            ax=current_ax,
            x_values=x_values,
            param_cases=param["values"],
            param_name=param["parameter_name"],
            title=title,
            reference_coefficient_parameters=reference_coefficient_parameters,
            vertical_lines_x_values=vertical_lines_x_values,
            points=points,
            plot_reference=plot_reference,
        )

    plt.tight_layout()

    if save_path:
        try:
            plt.savefig(save_path, dpi=size_dpi, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
        finally:
            plt.close(fig)
    else:
        plt.show()
