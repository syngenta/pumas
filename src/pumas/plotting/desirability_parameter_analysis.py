import numpy as np

from pumas.desirability import desirability_catalogue
from pumas.plotting.plotter import plot_parameter_analysis


def plot_sigmoid_analysis():
    func = desirability_catalogue.get("sigmoid")().utility_function

    reference_coefficient_parameters = {
        "low": 20.0,
        "high": 80.0,
        "k": 0.5,
        "base": 10.0,
        "shift": 0.0,
    }
    parameters = [
        {
            "parameter_name": "low",
            "title": "Low Threshold",
            "values": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        },
        {
            "parameter_name": "high",
            "title": "High Threshold",
            "values": [20, 30, 40, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        },
        {
            "parameter_name": "k",
            "title": "K (Positive)",
            "values": [0.0, 0.1, 0.3, 0.5, 1.0, 10.0],
        },
        {
            "parameter_name": "k",
            "title": "K (Negative)",
            "values": [-10.0, -1.0, -0.5, -0.3, -0.1, 0.0],
        },
        {
            "parameter_name": "base",
            "title": "Base",
            "values": [1.5, 2.0, np.e, 3.0, 10.0],
        },
        {
            "parameter_name": "shift",
            "title": "Shift",
            "values": [0.0, 0.1, 0.2, 0.3, 0.4],
        },
    ]

    plot_parameter_analysis(
        func=func,
        parameters=parameters,
        reference_coefficient_parameters=reference_coefficient_parameters,
        vertical_lines_x_values=[20.0, 50.0, 80.0],
        x_range=(1, 100),
        figsize=(18, 12),
    )


def plot_double_sigmoid_analysis():
    func = desirability_catalogue.get("double_sigmoid")().utility_function

    reference_coefficient_parameters = {
        "low": 20.0,
        "high": 80.0,
        "coef_div": 5.0,
        "coef_si": 1.0,
        "coef_se": 1.0,
        "base": 10.0,
        "invert": False,
        "shift": 0.0,
    }

    parameters = [
        {
            "parameter_name": "low",
            "title": "low",
            "values": [0.0, 10.0, 20.0, 30.0, 40.0, 50.0],
        },
        {
            "parameter_name": "high",
            "title": "high",
            "values": [50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        },
        {
            "parameter_name": "coef_si",
            "title": "coef_si",
            "values": [0.1, 0.5, 1.0, 5.0, 10.0],
        },
        {
            "parameter_name": "coef_se",
            "title": "coef_se",
            "values": [0.1, 0.5, 1.0, 5.0, 10.0],
        },
        {
            "parameter_name": "coef_div",
            "title": "coef_div",
            "values": [0.1, 0.5, 1.0, 5.0, 10.0],
        },
        {"parameter_name": "base", "title": "Base", "values": [2.0, np.e, 3.0, 10.0]},
        {"parameter_name": "invert", "title": "Invert", "values": [True, False]},
        {
            "parameter_name": "shift",
            "title": "Shift",
            "values": [0.0, 0.2, 0.4, 0.6, 0.8],
        },
    ]
    plot_parameter_analysis(
        func=func,
        parameters=parameters,
        reference_coefficient_parameters=reference_coefficient_parameters,
        vertical_lines_x_values=[20.0, 50.0, 80.0],
        x_range=(1, 100),
        figsize=(18, 12),
    )


def plot_sigmoid_bell_analysis():
    func = desirability_catalogue.get("sigmoid_bell")().utility_function
    reference_coefficient_parameters = {
        "x1": 20.0,
        "x4": 80.0,
        "x2": 45.0,
        "x3": 60.0,
        "k": 1.0,
        "base": 10.0,
        "invert": False,
        "shift": 0.0,
    }

    parameters = [
        {
            "parameter_name": "x1",
            "title": "x1",
            "values": [0.0, 10.0, 20.0, 30.0, 40.0],
        },
        {
            "parameter_name": "x4",
            "title": "x4",
            "values": [60.0, 70.0, 80.0, 90.0, 100.0],
        },
        {
            "parameter_name": "x2",
            "title": "x2",
            "values": [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        },
        {
            "parameter_name": "x3",
            "title": "x3",
            "values": [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        },
        {
            "parameter_name": "k",
            "title": "K",
            "values": [1.0, 2.0, 10.0],
        },
        {
            "parameter_name": "invert",
            "title": "Invert",
            "values": [True, False],
        },
        {"parameter_name": "base", "title": "Base", "values": [2.0, np.e, 3.0, 10.0]},
        {
            "parameter_name": "shift",
            "title": "Shift",
            "values": [0.0, 0.2, 0.4, 0.6, 0.8],
        },
    ]

    plot_parameter_analysis(
        func=func,  # Replace with actual function name
        parameters=parameters,
        reference_coefficient_parameters=reference_coefficient_parameters,
        vertical_lines_x_values=[20.0, 80.0],  # x_left and x_right
        x_range=(0, 100),
        figsize=(21, 24),
    )  # Increased figure size due to more parameters


def plot_bell_analysis():
    func = desirability_catalogue.get("bell")().utility_function
    reference_coefficient_parameters = {
        "width": 20.0,
        "slope": 2.0,
        "center": 50.0,
        "invert": False,
        "shift": 0.0,
    }
    parameters = [
        {
            "parameter_name": "width",
            "title": "Width",
            "values": [10.0, 20.0, 30.0, 40.0, 50.0],
        },
        {
            "parameter_name": "slope",
            "title": "Slope",
            "values": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0],
        },
        {
            "parameter_name": "center",
            "title": "Center",
            "values": [20.0, 30.0, 40.0, 50.0],
        },
        {
            "parameter_name": "center",
            "title": "Center",
            "values": [50.0, 60.0, 70.0, 80.0],
        },
        {
            "parameter_name": "invert",
            "title": "Invert",
            "values": [True, False],
        },
        {
            "parameter_name": "shift",
            "title": "Shift",
            "values": [0.0, 0.2, 0.4, 0.6, 0.8],
        },
    ]

    plot_parameter_analysis(
        func=func,
        parameters=parameters,
        reference_coefficient_parameters=reference_coefficient_parameters,
        vertical_lines_x_values=[20.0, 50.0, 80.0],
        x_range=(1, 100),
        figsize=(18, 12),
    )


def plot_multistep_analysis():
    func = desirability_catalogue.get("multistep")().utility_function
    reference_coefficient_parameters = {
        "coordinates": [(50.0, 0.0), (51.0, 0.8)],
    }
    parameters = [
        {
            "parameter_name": "coordinates",
            "title": "Sharp filter-like function",
            "values": [[(49.5, 0.0), (50.5, 1.0)]],
        },
        {
            "parameter_name": "coordinates",
            "title": "Sharp filter-like function",
            "values": [[(49.5, 1.0), (50.5, 0.0)]],
        },
        {
            "parameter_name": "coordinates",
            "title": "Growing with plateau",
            "values": [[(30.0, 0.0), (40.0, 0.8), (50.0, 1.0)]],
        },
        {
            "parameter_name": "coordinates",
            "title": "Decreasing from plateau",
            "values": [[(30.0, 1.0), (40.0, 0.8), (50.0, 0.0)]],
        },
        {
            "parameter_name": "coordinates",
            "title": "Bell-like function",
            "values": [[(25.0, 0.0), (40.0, 1.0), (60.0, 1.0), (80.0, 0.0)]],
        },
        {
            "parameter_name": "coordinates",
            "title": "Custom maximum",
            "values": [[(25.0, 0.0), (40.0, 0.8), (60.0, 0.8), (80.0, 0.0)]],
        },
        {
            "parameter_name": "coordinates",
            "title": "Inverse bell-like function",
            "values": [[(25.0, 1.0), (40.0, 0.0), (60.0, 0.0), (80.0, 1.0)]],
        },
        {
            "parameter_name": "shift",
            "title": "Shift",
            "values": [0.0, 0.2, 0.5, 0.8],
        },
    ]
    plot_parameter_analysis(
        func=func,
        reference_coefficient_parameters=reference_coefficient_parameters,
        parameters=parameters,
        vertical_lines_x_values=[20.0, 50.0, 80.0],
        x_range=(1, 100),
        figsize=(18, 12),
        plot_reference=False,
    )


def plot_leftstep_analysis():
    func = desirability_catalogue.get("leftstep")().utility_function
    reference_coefficient_parameters = {
        "low": 50.0,
        "high": 50.0,
        "shift": 0.0,
    }
    parameters = [
        {
            "parameter_name": "low",
            "title": "low",
            "values": [20.0, 50.0, 80.0],
        },
        {
            "parameter_name": "shift",
            "title": "shift",
            "values": [0.0, 0.2, 0.5, 0.8],
        },
    ]
    plot_parameter_analysis(
        func=func,
        reference_coefficient_parameters=reference_coefficient_parameters,
        parameters=parameters,
        vertical_lines_x_values=[20.0, 50.0, 80.0],
        x_range=(1, 100),
        figsize=(18, 12),
        plot_reference=False,
    )


def plot_rightstep_analysis():
    func = desirability_catalogue.get("rightstep")().utility_function
    reference_coefficient_parameters = {
        "low": 50.0,
        "high": 50.0,
        "shift": 0.0,
    }
    parameters = [
        {
            "parameter_name": "high",
            "title": "high",
            "values": [20.0, 50.0, 80.0],
        },
        {
            "parameter_name": "shift",
            "title": "shift",
            "values": [0.0, 0.2, 0.5, 0.8],
        },
    ]
    plot_parameter_analysis(
        func=func,
        reference_coefficient_parameters=reference_coefficient_parameters,
        parameters=parameters,
        vertical_lines_x_values=[20.0, 50.0, 80.0],
        x_range=(1, 100),
        figsize=(18, 12),
        plot_reference=False,
    )


def plot_step_analysis():
    func = desirability_catalogue.get("step")().utility_function
    reference_coefficient_parameters = {
        "low": 20.0,
        "high": 80.0,
        "shift": 0.0,
    }
    parameters = [
        {
            "parameter_name": "low",
            "title": "low",
            "values": [20.0, 30.0, 40.0],
        },
        {
            "parameter_name": "high",
            "title": "high",
            "values": [60.0, 70.0, 80.0],
        },
        {
            "parameter_name": "shift",
            "title": "shift",
            "values": [0.0, 0.2, 0.5, 0.8],
        },
    ]
    plot_parameter_analysis(
        func=func,
        reference_coefficient_parameters=reference_coefficient_parameters,
        parameters=parameters,
        vertical_lines_x_values=[20.0, 50.0, 80.0],
        x_range=(1, 100),
        figsize=(18, 12),
        plot_reference=True,
    )


if __name__ == "__main__":
    print(
        """
    Run the functions to plot the parameter analysis for desirability functions
    """
    )
    # plot_sigmoid_analysis()
    # plot_double_sigmoid_analysis()
    # plot_sigmoid_bell_analysis()
    # plot_bell_analysis()
    # plot_multistep_analysis()
    # plot_leftstep_analysis()
    # plot_rightstep_analysis()
    # plot_step_analysis()
