========
Usage
========

From the command line
*********************

deepQuest can be run from the command line by using the commands ``deepquest`` or ``dq`` followed by either ``train``, ``predict`` or ``score``.
Alternatively use ``-h`` or ``--help`` for information on how to run deepQuest.
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
To train a new model, you must pass in a configuration file as a YAML or pickle file using the following syntax for command line or python execution respectively:

.. code:: shell

  dq train --config configs/config-file.yml

or

.. code:: python

  dq.train('configs/config-file.yml')

For information on configuration files see Tutorial_.
The pickle option is included for back-compatability of previously trained models and is not applicable to models trained with deepQuest version 2 and above.

.. _Tutorial: tutorials.html

Configuration changes can be provided as key=value pairs, over-riding those in the config file, for instance the following commands will resume training of an existing model.

.. code:: shell

  dq train --config configs/config-file.yml RELOAD=True RELOAD_EPOCH=8

.. code:: python

  dq.train('configs/config-file.yml', {RELOAD:True, RELOAD_EPOCH:8})

*Note*: The python function takes configuration changes pairs as dictionary elements.

2. Prediction
"""""""""""""
To use a trained model in performing quality estimation, use deepQuest's `predict` function.

.. todo:: add documentation for predict when this is finalised.

3. Scoring
""""""""""
deepQuest's scoring function takes two files of the same length, i.e. lists of known and predicted scores, in any order and computes three score metrics: Pearson's correlation coefficient; Mean Absolute Error; and Root Mean Squared Error.

.. code:: shell

  dq score known.scores predicted.scores

.. code:: python

  dq.score(['known.scores', 'predicted.scores'])

Where `known.scores` and `predicted.scores` are paths to the two files.
