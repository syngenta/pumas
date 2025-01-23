pumas
=======================================

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
