Strategy Pattern
================

Usage and Purpose
-----------------
The Strategy Pattern is a fundamental design pattern in software engineering,
primarily used to enable an object to change its behavior dynamically.
It's particularly useful in scenarios where an object must be able to switch between different algorithms or strategies at runtime.
The essence of this pattern is to define a family of algorithms, encapsulate each one of them, and make them interchangeable.
This allows for algorithm variations independently from clients that use them.

Limitations
-----------
The greatest shortcoming of the traditional strategy pattern is that it does not allow for passing parameters to strategies.
This is necessary in many real-world applications, where algorithms require different parameters to run.


Common Suboptimal Solutions
---------------------------
In practice, several approaches have been used to handle the issue of passing parameters to strategies, each with its shortcomings:

1. **Passing set of parameters as Keyword Arguments (kwargs):** This dynamic approach reduces code readability and can lead to silent contract breaches, causing potential issues that are not immediately apparent.

2. **Generic Parameters Interface:** Creating a generic Parameters container class to encompass all parameters required by different algorithms leads to significant coupling, which can hinder maintenance and scalability.

3. **Dedicated Parameters Classes for Each Concrete Strategy:** While this reduces coupling compared to a generic interface, it can lead to bloated code, reduced readability, and synchronization challenges between strategies and their parameters.

4. **Defining Parameters as Properties:** A commonly effective solution involves defining different parameters as properties of concrete strategy classes. These are then initialized as needed, maintaining separation of concerns and adhering to modularity principles.

Standard Strategy Pattern UML Diagram
-------------------------------------
.. mermaid::

    classDiagram
        class Context {
            -Strategy strategy
            +Context(Strategy strategy)
            +executeStrategy()
        }
        class Strategy {
            <<interface>>
            +executeAlgorithm()
        }
        class ConcreteStrategyA {
            +executeAlgorithm()
        }
        class ConcreteStrategyB {
            +executeAlgorithm()
        }

        Context "1" --> "1" Strategy : has-a
        Strategy <|-- ConcreteStrategyA : implements
        Strategy <|-- ConcreteStrategyB : implements



Parameterized Strategy Pattern
------------------------------
The Parameterized Strategy Pattern[1] presents an innovative solution to the limitations of the traditional strategy pattern.
This approach involves creating an abstract `Parameter` class, which is then extended by concrete parameter
implementations for each specific parameter type (ex `int`, `float`, `boolean`).
Each of the concrete strategy classes then contains a list of these concrete `Parameter` classes, which are initialized in the constructor.
This approach allows for the passing of parameters to strategies, while maintaining separation of concerns and adhering to modularity principles.
Besides, it allows for the automatic detection of parameters by the client, which is particularly useful for creating dynamic graphical user interfaces (GUIs)
where the available controls and inputs adjust according to the chosen strategy's parameters.
This method allows each concrete strategy to define its own set of parameters, leading to more flexible, dynamic, and user-friendly applications.

[1] Sobajic, O., et al. (2010). Parameterized strategy pattern. Proceedings of the 17th Conference on Pattern Languages of Programs. Reno, Nevada, USA, Association for Computing Machinery: Article 9.
- `Link to Paper <https://dl.acm.org/doi/10.1145/2493288.2493297>`_
- `Link to PDF <https://hillside.net/plop/2010/papers/ACMVersions/papers/sobajic.pdf>`_


Relation to Other Design Patterns
----------------------------------
- **Strategy Pattern:** This approach extends the traditional strategy pattern by incorporating parameter flexibility.
- **Adapter Pattern:** Each concrete algorithm can be seen as an adapter, transforming the required interface to a specific algorithm interface using parameter classes.
- **Factory Pattern:** The client can be seen as a factory, creating concrete algorithms and their parameters.

UML Diagram for Parameterized Strategy Pattern
----------------------------------------------

The key agents in this pattern and their responsibilities are as follows:

1. **Client:** Responsible for selecting algorithms and handling parameter instances, including presenting them in the GUI for user interaction.
2. **Concrete Algorithm:** Contains the specific algorithm routine with a unique set of parameters.
3. **Parameter Classes:** Abstract and concrete parameter classes handle specific types and constraints of parameters.

.. mermaid::

    classDiagram
        class Client {
            +selectAlgorithm()
            +executeAlgorithm()
        }
        class AbstractAlgorithm {
            <<interface>>
            +execute()
            +getParameters()
        }
        class ConcreteAlgorithmA {
            +execute()
            +getParameters()
        }
        class ConcreteAlgorithmB {
            +execute()
            +getParameters()
        }
        class AbstractParameter {
            <<interface>>
            +getValue()
            +setValue()
        }
        class ConcreteParameter1
        class ConcreteParameter2

        Client --> AbstractAlgorithm : uses
        AbstractAlgorithm <|-- ConcreteAlgorithmA
        AbstractAlgorithm <|-- ConcreteAlgorithmB
        AbstractAlgorithm "1" --> "*" AbstractParameter : has
        AbstractParameter <|-- ConcreteParameter1
        AbstractParameter <|-- ConcreteParameter2

Sequence Diagram for Parameterized Strategy Pattern
----------------------------------------------------

.. mermaid::

    sequenceDiagram;
        participant User
        participant Client
        participant Algorithm
        participant Parameter1
        participant Parameter2

        User->>+Client: Choose Algorithm
        Client->>+Algorithm: Instantiate
        Algorithm->>-Client: Algorithm Instance
        Client->>+Algorithm: getParameters()
        Algorithm->>-Client: Parameter1, Parameter2
        Client->>User: Display Parameters (Parameter1, Parameter2)
        User->>Client: Modify Parameter Values
        Client->>+Parameter1: SetValue(newValue1)
        Parameter1->>-Client: Value Set
        Client->>+Parameter2: SetValue(newValue2)
        Parameter2->>-Client: Value Set
        User->>+Client: Execute Algorithm
        Client->>+Algorithm: execute()
        Algorithm->>+Parameter1: GetValue()
        Parameter1->>-Algorithm: Value1
        Algorithm->>+Parameter2: GetValue()
        Parameter2->>-Algorithm: Value2
        Algorithm->>Algorithm: Run Algorithm (Value1, Value2)
        Algorithm->>-Client: Execution Complete
        Client->>-User: Results/Output


Implemented Parameter types
---------------------------

.. automodule:: pumas.architecture.parameters
    :no-index:
