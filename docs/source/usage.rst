========
Usage
========

From the command line
*********************

deepQuest can be run from the command line by using the commands ``deepquest`` or ``dq`` followed by either ``train``, ``predict`` or ``score``.
Alternatively use ``-h`` or ``--help`` for information on running deepQuest.
Each mode for deepQuest has some required and optional user input.

As a python module
******************

deepQuest can be imported as a python module via

.. code:: python

  import deepquest as dq

Training, Prediction and Scoring
********************************

1. Training
"""""""""""
To train a new model, you must pass in a configuration file as a YAML or PKL file using the following syntax for command line or python execution respectively:

.. code:: shell

  dq train -c configs/config-file.yml

or

.. code:: python

  dq.train('configs/config-file.yml')
  dq.train('config.pkl')

For information on configuration files see Tutorial_.
The pickle option is included for back-compatability of previously trained models and is not applicable to models trained with deepQuest version 2 and above.


Configuration changes can be provided as key=value pairs, over-riding those in the config file, for instance the following commands will resume training of an existing model.

.. code:: shell

  dq train -c configs/config-file.yml --changes RELOAD=True RELOAD_EPOCH=8

.. code:: python

  dq.train('configs/config-file.yml', {RELOAD:True, RELOAD_EPOCH:8})

*Note*: The python function takes configuration changes pairs as dictionary elements.

2. Prediction
"""""""""""""
To use a trained model in performing quality estimation inference, use deepQuest's `predict` function. Operation is very similar to using `train`:

.. code:: shell

  dq predict -c configs/config-predict.yml

or

.. code:: python

  dq.predict('configs/config-predict.yml')
  dq.predict('configs/config-file.yml', {RELOAD:True, RELOAD_EPOCH:8})


3. Scoring
""""""""""
deepQuest's scoring function takes two files of the same length, i.e. lists of known and predicted scores, in any order and computes three score metrics: Pearson's correlation coefficient; Mean Absolute Error; and Root Mean Squared Error.

.. code:: shell

  dq score known.scores predicted.scores

.. code:: python

  dq.score(['known.scores', 'predicted.scores'])

Where `known.scores` and `predicted.scores` are paths to the two files, as a two-element list.


GPU Training
************
To train using a GPU, the operation is different depending on whether you are using the command-line interface or the python module.
For the command line: include `GPU_ID` as a parameter in the config file (or using the `--changes` option), set as a series of integers corresponding to the number of GPUs you intend to use. For example: `GPU_ID: 0,1` will train on GPU devices 0 and 1 (order is determined by PCI BUS ID).
Or for the python module, two environment variables must be set before importing the deepQuest package using the following:

.. code:: python

  import os
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

where the `CUDA_VISIBLE_DEVICES` environment variable must be set to a comma-separated string of device IDs. Then, set the config parameter `N_GPUS` to the number of devices specified (in this example, `2`.) Note: this is done automatically in the command line example.

To run on the CPU only, omit `GPU_ID` from the configuration parameters for the command-line mode, or for the python module set `CUDA_VISIBLE_DEVICES` as an empty string.

Setting random seed for reproducible results
********************************************
To obtain the same results with every run, for instance in development or testing, the random seed must be set. This is done slightly differently for the command line mode and the python module mode.
For the command line, set the config parameter `SEED` as an integer, use the same value each time for reproducible results.
Or for the python module, use the numpy and random module seed functions before importing the deepQuest module:

.. code:: python

  seed = 1
  import numpy.random
  numpy.random.seed(seed)
  import random
  random.seed(seed)


.. _Tutorial: tutorials.html
