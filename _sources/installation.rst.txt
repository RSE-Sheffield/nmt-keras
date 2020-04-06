############
Installation
############

deepQuest is written in Python and we highly recommend that you use a Conda_ environment in order to keep under control your working environment, without interfering with your system-wide configuration, neither former installation of dependencies.

There are a few ways to install deepQuest:

via pip
----------------
deepQuest can be installed using the python package manager, pip_, via one of several methods.
We recommend installing within a Conda_ environment and ensuring you are using the latest versions of setuptools and pip.
deepQuest supports python versions 3.6 and 3.7

1. pip install from remote repo
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Within a conda environment (for example named ``conda-env``)

.. code-block:: shell

  (conda-env)$ pip install https://github.com/RSE-Sheffield/deepQuest

2. pip install from remote wheel, tar or zip file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using the URL of a built wheel, tar or zip file from the releases tab of the `deepQuest gitHub repo <https://github.com/RSE-Sheffield/deepQuest>`_
Again, within a conda environment

.. code-block:: shell

  (conda-env)$ pip install https://github.com/RSE-Sheffield/deepQuest/master/archive/something.whl


3. pip install from local wheel, tar or zip file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Download the wheel, tar or zip from the releases tab of the `deepQuest gitHub repo <https://github.com/RSE-Sheffield/deepQuest>`_  to a local location and

.. code-block:: shell

  (conda-env)$ pip install deepQuest.whl



via Git clone
----------------------
If you need to have access to the source code for development, then we suggest git cloning the deepQuest gitHub repository and installing the package using ``pip`` to run ``setup.py``.

git clone the gitHub repository and build via ``setup.py``

.. code-block:: shell

  (conda-env)$ git clone https://github.com/RSE-Sheffield/deepQuest
  (conda-env)$ cd deepquest
  (conda-env)$ pip install -e .

*Note*: The ``-e`` option for pip_ puts the package source code in a user editable location, in this example, it is left in the location from which it is installed, allowing developer changes to the code.
Please consider contributing useful additions to the code via a gitHub pull request (see 'How to contribute').

Computational Requirements
--------------------------

deepQuest is GPU compatible and we highly recommend to train your models on GPU with a minimum of 8GB memory available.
Of course, if you don't have access to such resources, you can also use deepQuest on a CPU server.
This will considerably extend the training time, especially for complex architecture, such as the POSTECH model (`Kim et al., 2017`_), on large datasets, but architectures such as BiRNN should work fine and take about 12 hours to be trained (while ~20min on GPU).


.. ==============================================================================
.. _Conda: https://conda.io/docs/user-guide/tasks/manage-environments.html
.. _Keras: https://github.com/MarcBS/keras
.. _Multimodal Keras Wrapper: https://github.com/lvapeab/multimodal_keras_wrapper
.. _pip: https://en.wikipedia.org/wiki/pip_(package_manager)
.. _`NMT-Keras`: https://nmt-keras.readthedocs.io/en/latest/requirements.html
.. _`Kim et al., 2017`: http://www.statmt.org/wmt17/pdf/WMT63.pdf
