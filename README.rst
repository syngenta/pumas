PUMAS (Probabilities & Uncertainties in Multi-objective Assessment & Scoring)
===============================================================================


.. image:: https://img.shields.io/pypi/v/pumas
   :target: https://pypi.python.org/pypi/pumas
   :alt: PyPI - Version
.. image:: https://static.pepy.tech/badge/pumas/month
   :target: https://pepy.tech/project/pumas
   :alt: Downloads monthly
.. image:: https://static.pepy.tech/badge/pumas
   :target: https://pepy.tech/project/pumas
   :alt: Downloads total
.. image:: https://img.shields.io/github/actions/workflow/status/syngenta/pumas/test_suite.yml?branch=main
   :alt: GitHub Workflow Status
.. image:: https://readthedocs.org/projects/pumas-toolkit/badge/?version=latest
   :target: https://pumas-toolkit.readthedocs.io/en/latest/?badge=latest
   :alt: Documentation Status
.. image:: https://img.shields.io/badge/contributions-welcome-blue
   :target: https://github.com/syngenta/pumas/blob/main/CONTRIBUTING.md
   :alt: Contributions
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT

---------------------


This Python package implements a flexible multi-objective scoring system based on desirability functions and aggregation.

Key Features
------------

* Define custom scoring profiles with:

  - Desirability functions for each objective
  - Aggregation algorithm selection
  - Optional weighting and importance factors

* Calculate individual desirability scores for each property
* Aggregate scores using the specified method
* Process data from various input formats (e.g., dictionaries, dataframes)

Use Cases
---------

* Decision support systems
* Multi-criteria optimization
* Performance evaluation
* Product or candidate ranking


Installation
------------

Create a dedicated Python environment for this package with your favorite environment manager.

.. code-block:: shell

   conda create -n pumas python=3.9
   conda activate pumas


* Option 1: Install the package from the github repository:

.. code-block:: shell

   pip install git+ssh://git@github.com/syngenta/pumas.git@main

* Option 2: Install the package from the Python Package Index (PyPI):

.. code-block:: shell

   pip install pumas

Installing optional dependencies
---------------------------------
Extensions to Pumas, have conditional dependencies on a variety of third-party Python packages.
All dependencies are installed

A full list of conditional dependencies can be found in Pumas's pyproject.toml (stored related requirements text files).

Uncertainty Management and Probabilistic Scoring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The core installation of PUMAS support a basic scoring framework based on numerical values.
To enable probabilistic scoring frameworks, to use and propagate value uncertainty, please install optional libraries with:

.. code-block:: shell

   pip install pumas[uncertainty]

Graphical bindings and plotting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The core installation of PUMAS does not include any plotting capability, and, hence, the graphical bindings are unavailable.
To enable both the plotting module and the graphical binding, please install the optional libraries with:

.. code-block:: shell

   pip install pumas[graphics]


Development Installation
---------------------------

When working on the development of this package, the developer wants to work
directly on the source code while still using the packaged installation.

Please install the package in development mode, including all dependencies.

.. code-block:: shell

   git clone git@github.com:syngenta/pumas.git
   pip install -e pumas/[dev]
