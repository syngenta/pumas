PUMAS
=======================================


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

PUMAS

Installation
------------

Create a dedicated python environment for this package with your favorite environment manager.

.. code-block:: shell

   conda create -n pumas python=3.9
   conda activate pumas


* Option 1: Install the package from the git repository:

.. code-block:: shell

   pip install git+ssh://git@github.com/syngenta/pumas.git

* Option 2: Install the package from the python package repository if its url is configured in the pip configuration file:

.. code-block:: shell

   pip install pumas

* Option 3: Install the package from the python package repository if its url is not configured in the pip configuration file:

.. code-block:: shell

    pip install pumas


For Development
---------------

When working on the development of this package, the developer wants to work
directly on the source code while still using the packaged installation. For
that, run:

.. code-block:: shell

   git clone git@github.com:syngenta/pumas.git
   pip install -e pumas/[dev]
