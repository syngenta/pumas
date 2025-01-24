"""
The Parameterized Strategy Architecture for Desirability Functions
------------------------------------------------------------------

The `desirability` module defines an architecture for implementing parameterized strategies,
particularly applied to the concept of desirability functions in the context
of multi-objective scoring and optimization.

The core of the architecture is an abstract class that encapsulates
all necessary checks and validations, providing a foundational framework
for concrete strategy classes. The concrete classes are responsible
for defining the specific algorithms that constitute various desirability
functions and how they are applied.

In each algorithm, parameters are considered to fall into two categories:

1. Coefficient Parameters: These parameters are essential for defining the functional form
    of the desirability equation. They remain consistent across multiple executions of the
    algorithm and are set when a concrete strategy class is instantiated.

2. Input Parameters: These parameters are assigned values each time the desirability
    algorithm is executed. They represent the inputs to the utility functions that are
    subject to variation during the algorithm's application.

The design enables the utility functions to be treated as parametric equations.
Practitioners have the flexibility to compute outcomes by varying the input
parameters while keeping coefficients fixed. Alternatively, different coefficient
values can be explored to understand their influence on the
functional form of the equation.

The method ensures that the returned desirability
score strictly ranges between 0 (inclusive) and 1 (inclusive), adhering to the principles of
desirability function algorithms. Some algorithms may require additional coefficient parameters
to define this range and shape the desirability curve.

By adhering to the structure provided by the abstract class, the concrete implementations
encapsulate utility functions with distinct desirability function algorithms.
This approach offers a standardized way to manage and execute these specialized
functions, facilitating the evaluation and optimization
of multiple objectives in a consistent and extensible manner.

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
        class BaseDesirability {
        +compute_score(x: float): float
        +compute_uscore(x: UFloat): UFloat
        }
        class SigmoidDesirability {
        +parameters ['low', 'high', 'k', 'base', 'shift']
        }
        class GeneralizedBellMembershipDesirability {
        +parameters ['width', 'slope', 'center', 'shift']
        }
        class StepDesirability
        class CategoricalDesirability
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
        AbstractParametrizedStrategy <|--BaseDesirability
        BaseDesirability <|-- SigmoidDesirability
        BaseDesirability <|-- GeneralizedBellMembershipDesirability
        BaseDesirability <|-- StepDesirability
        BaseDesirability <|-- CategoricalDesirability
        AbstractParametrizedStrategy --> ParameterManager : uses
        AbstractParametrizedStrategy --> Parameter : has
        SigmoidDesirability ..> desirability_catalogue : "registers in desirability_catalogue"
        GeneralizedBellMembershipDesirability ..> desirability_catalogue : "registers in desirability_catalogue"
        StepDesirability ..> desirability_catalogue : "registers in desirability_catalogue"
        CategoricalDesirability ..> desirability_catalogue : "registers in desirability_catalogue"
        Catalogue <|-- desirability_catalogue

"""  # noqa: E501
