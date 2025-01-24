"""
The Parameterized Strategy Architecture for Aggregation Functions
------------------------------------------------------------------
The `aggregation` module defines an architecture for implementing aggregation methods,
particularly applied to the concept of weighted means in the context of multi-objective scoring and optimization.


The core of the architecture is an abstract class The concrete classes are responsible for defining the specific
algorithms that constitute various aggregation methods and how they are applied.
The interface imposes that each concrete implements a `compute` method that accepts two input values,
a list of values and a list of weights. The method ensures that the returned aggregation score is a UFloat,

Class Diagram
~~~~~~~~~~~~~


.. mermaid::

    classDiagram

            class AbstractParametrizedStrategy {
            <<abstract>>
            utility_function Callable
            parameter_manager ParameterManager
            coefficient_parameters_names List[str]
            input_parameters_names List[str]
            get_coefficient_parameters_values() Dict[str, Any]
            set_coefficient_parameters_values(Dict[str, Any])

        }

        class  BaseAggregation {
         +compute_score(values[float], weights[float]): float
         +compute_uscore(values[ufloat], weights[float]): UFloat
        }

        class WeightedArithmeticMeanAggregation
        class WeightedGeometricMeanAggregation
        class WeightedHarmonicMeanAggregation
        class WeightedDeviationIndexAggregation {
        parameters ['ideal_value']
        }

        class Catalogue {
            +item_type Type
            +register
        }

        class ParameterManager {
            +parameters_map Dict[str, Parameter]
            +set_parameter_value()
            +set_parameter_attributes()
        }

        class Parameter {
            <<interface>>
            +value Any
            +set_value()
        }


        AbstractParametrizedStrategy <|--BaseAggregation
        BaseAggregation <|-- WeightedArithmeticMeanAggregation
        BaseAggregation <|-- WeightedGeometricMeanAggregation
        BaseAggregation <|-- WeightedHarmonicMeanAggregation
        BaseAggregation <|-- WeightedDeviationIndexAggregation

        AbstractParametrizedStrategy --> ParameterManager : uses
        AbstractParametrizedStrategy --> Parameter : has


        WeightedArithmeticMeanAggregation ..> aggregation_catalogue : "registers in aggregation_catalogue"
        WeightedGeometricMeanAggregation ..> aggregation_catalogue : "registers in aggregation_catalogue"
        WeightedHarmonicMeanAggregation ..> aggregation_catalogue : "registers in aggregation_catalogue"
        WeightedDeviationIndexAggregation ..> aggregation_catalogue : "registers in aggregation_catalogue"

        Catalogue <|-- aggregation_catalogue

Example Usage
~~~~~~~~~~~~~~~~~~~~~~~~~~
Retrieve the available aggregation functions from the catalogue.

>>> from pumas.aggregation import aggregation_catalogue
>>> aggregation_catalogue.list_items()
['arithmetic_mean', 'geometric_mean', 'harmonic_mean', 'deviation_index']


Data Validation Flowchart
~~~~~~~~~~~~~~~~~~~~~~~~~~

Prior of computing the aggregation, the input data_frame is validated and cleaned.

The following conditions are checked and reported as warnings/log entries:
    - Null (`None`) values.

Warning: value-weight pairs containing null values are removed from the input data_frame.

The following conditions must be met otherwise an exception is raised:
    - The values and weights arrays should have the same length.
    - No value can be negative.
    - No weight can be negative.
    - Weights cannot be null (`None`).
    - Values cannot be null (`None`) (after the cleaning step and the warning raising, this is a catch-all failsafe).



The following flowchart describes the process:



.. mermaid::

    flowchart TB

        start(("Start"))
        start --> checkLengthMatch["check_length_match(values, weights)"]

        checkLengthMatch --> |"length mismatch found)"| raiseLengthMismatchException("Raise LengthMismatchException")
        checkLengthMatch --> reportNullValues["report_null_values(values)"]

        reportNullValues --> reportNullWeights["report_null_weights(weights)"]
        reportNullValues --> raiseNullValueswarning

        reportNullWeights --> filterValues["filter_out_null_values_weights_pairs(values, weights)"]
        reportNullWeights--> raiseNullWeightswarning

        filterValues --> checkNoneWeights["check_null_weights(weights)"]
        checkNoneWeights --> |"Raise Exception"| raiseNoneWeightsException("Raise NoneWeightsException")

        checkNoneWeights --> checkNoneValues["check_null_values(values)"]
        checkNoneValues --> |"Raise Exception"| raiseNoneValuesException("Raise NoneValuesException")

        checkNoneValues --> checkNegativeWeights["check_negative_weights(weights)"]
        checkNegativeWeights --> |"Raise Exception"| raiseNegativeWeightsException("Raise NegativeWeightsException")

        checkNegativeWeights --> checkNegativeValues["check_negative_values(values)"]
        checkNegativeValues --> |"Raise Exception"| raiseNegativeValuesException("Raise NegativeValuesException")
        checkNegativeValues --> proceedWithCalculation[("Proceed with Calculation")]

        raiseNullWeightswarning --> Log
        raiseNullValueswarning --> Log

        raiseLengthMismatchException --> endState["End State: cannot proceed with calculation"]
        raiseNegativeWeightsException --> endState
        raiseNegativeValuesException --> endState
        raiseNoneWeightsException --> endState
        raiseNoneValuesException --> endState



"""  # noqa: E501,  W293
