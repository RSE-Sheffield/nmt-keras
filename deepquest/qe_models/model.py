#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# model.py
#
# Copyright (C) 2019 Frederic Blain (feedoo) <f.blain@sheffield.ac.uk>
#
# Licensed under the "THE BEER-WARE LICENSE" (Revision 42):
# Fred (feedoo) Blain wrote this file. As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a tomato juice or coffee in return
#

import os
import sys
import logging

from abc import ABCMeta, abstractmethod
from keras.utils import multi_gpu_model
from .utils import *


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


"""
QE model class, based on the TranslationModel class from lvapeab/NMT-Keras.
"""

class QEModel(Model_Wrapper, metaclass=ABCMeta):
    """
    QEModel class. Instance of the Model_Wrapper class.

    :param dict params: all hyperparameters of the model.
    :param str model_type: network name type (corresponds to any method defined in the section 'MODELS' of this class).
                 Only valid if 'structure_path' == None.
    :param int verbose: set to 0 if you don't want the model to output informative messages
    :param str structure_path: path to a Keras' model json file.
                          If we speficy this parameter then 'type' will be only an informative parameter.
    :param str weights_path: path to the pre-trained weights file (if None, then it will be initialized according to params)
    :param str model_name: optional name given to the network (if None, then it will be assigned to current time as its name)
    :param dict vocabularies: vocabularies used for word embedding
    :param str store_path: path to the folder where the temporal model packups will be stored
    :param bool set_optimizer: Compile optimizer or not.
    :param bool clear_dirs: Clean model directories or not.
    """

    def __init__(self, params, model_type='QEModel',
            structure_path=None, weights_path=None, model_name=None,
            vocabularies=None, store_path=None, set_optimizer=True,
            clear_dirs=True, verbose=1, trainable=True
            ):

        """
        QEModel object constructor.

        :param params: all hyperparams of the model.
        :param model_type: network name type (corresponds to any method defined in the section 'MODELS' of this class).
                     Only valid if 'structure_path' == None.
        :param verbose: set to 0 if you don't want the model to output informative messages
        :param structure_path: path to a Keras' model json file.
                              If we speficy this parameter then 'type' will be only an informative parameter.
        :param weights_path: path to the pre-trained weights file (if None, then it will be randomly initialized)
        :param model_name: optional name given to the network (if None, then it will be assigned to current time as its name)
        :param vocabularies: vocabularies used for word embedding
        :param store_path: path to the folder where the temporal model packups will be stored
        :param set_optimizer: Compile optimizer or not.

        :param trainable_est: Is estimator trainable?
        :param trainable_pred: Is predictor trainable?
        """

        # inheritance kept that way for compatilibity with Py2.7
        super(QEModel, self).__init__(model_type=params.get('MODEL_TYPE', 'QEModel'),
            model_name=params.get('MODEL_NAME', None),
            silence=params.get('VERBOSE', 0) == 0,
            models_path=params.get('STORE_PATH', None),
            inheritance=True
            )

        # mandatory config values before building a ModelWrapper object
        self.params = params
        # self._model_type = params['MODEL_TYPE']
        self.vocabularies = params.get('VOCABULARY', vocabularies)
        self.name = params.get('MODEL_NAME', None)
        self.models_path = params.get('STORE_PATH', store_path)

        self.verbose = params.get('VERBOSE', verbose)

        self.ids_inputs = params['INPUTS_IDS_MODEL']
        self.ids_outputs = params['OUTPUTS_IDS_MODEL']
        #TODO: is 'return_alphas' useful for our QEModel, or simply used for MT?
        self.return_alphas = params['COVERAGE_PENALTY'] or params['POS_UNK']

        # Sets the model name and prepares the folders for storing the models
        # clear_dirs: Clean model directories or not (default: False).
        self.setName(self.name, models_path=self.models_path, clear_dirs=False)
        self.trainable = trainable
        self.use_CuDNN = 'CuDNN' if K.backend() == 'tensorflow' and params.get('USE_CUDNN', True) else ''
        # self.__toprint = ['_model_type', 'name', 'model_path', 'verbose']
        self.__toprint = ['name', 'net_type', 'model_path', 'vocabularies', 'verbose']

        # Prepare source word embedding
        if params['SRC_PRETRAINED_VECTORS'] is not None:
            if self.verbose > 0:
                logger.info("<<< Loading pretrained word vectors from:  " + params['SRC_PRETRAINED_VECTORS'] + " >>>")
            src_word_vectors = np.load(os.path.join(params['SRC_PRETRAINED_VECTORS']), allow_pickle=True).item()
            self.src_embedding_weights = np.random.rand(params['INPUT_VOCABULARY_SIZE'],
                                                        params['SOURCE_TEXT_EMBEDDING_SIZE'])
            for word, index in iteritems(self.vocabularies[self.ids_inputs[0]]['words2idx']):
                if src_word_vectors.get(word) is not None:
                    self.src_embedding_weights[index, :] = src_word_vectors[word]
            self.src_embedding_weights = [self.src_embedding_weights]
            self.src_embedding_weights_trainable = params['SRC_PRETRAINED_VECTORS_TRAINABLE'] and params.get('TRAINABLE_ENCODER', True)
            del src_word_vectors

        else:
            self.src_embedding_weights = None
            self.src_embedding_weights_trainable = params.get('TRAINABLE_ENCODER', True)

        # Prepare target word embedding
        if params['TRG_PRETRAINED_VECTORS'] is not None:
            if self.verbose > 0:
                logger.info("<<< Loading pretrained word vectors from: " + params['TRG_PRETRAINED_VECTORS'] + " >>>")
            trg_word_vectors = np.load(os.path.join(params['TRG_PRETRAINED_VECTORS']), allow_pickle=True).item()
            self.trg_embedding_weights = np.random.rand(params['OUTPUT_VOCABULARY_SIZE'],
                                                        params['TARGET_TEXT_EMBEDDING_SIZE'])
            for word, index in iteritems(self.vocabularies[self.ids_outputs[0]]['words2idx']):
                if trg_word_vectors.get(word) is not None:
                    self.trg_embedding_weights[index, :] = trg_word_vectors[word]
            self.trg_embedding_weights = [self.trg_embedding_weights]
            self.trg_embedding_weights_trainable = params['TRG_PRETRAINED_VECTORS_TRAINABLE'] and params.get('TRAINABLE_DECODER', True)
            del trg_word_vectors
        else:
            self.trg_embedding_weights = None
            self.trg_embedding_weights_trainable = params.get('TRAINABLE_DECODER', True)

        # # # Prepare model
        # if structure_path:
        #     # Load a .json model
        #     if self.verbose > 0:
        #         logger.info("<<< Loading model structure from file " + structure_path + " >>>")
        #     self.model = model_from_json(open(structure_path).read())
        # else:
        #     # Build model from scratch
        #     if hasattr(self, model_type):
        #         if self.verbose > 0:
        #             logger.info("<<< Building " + model_type + " Translation_Model >>>")
        #         eval('self.' + model_type + '(params)')
        #     else:
        #         raise Exception('Translation_Model model_type "' + model_type + '" is not implemented.')

        self.build()

        if set_optimizer:
            self.setOptimizer()

        # Print information of self
        if verbose > 0:
            print(str(self))
            self.model.summary()
            sys.stdout.flush()


    @abstractmethod
    def build(self):
        raise NotImplementedError


    def from_file(self, weights_path):
        """
        Load weights from file
        """
        if self.verbose > 0:
            logger.info("<<< Loading weights from file " + weights_path + " >>>")
        self.model.load_weights(weights_path, by_name=True)


    def __str__(self):
        """
        Plots basic model information.

        :return: String containing model information.
        """
        obj_str = '-----------------------------------------------------------------------------------\n'
        class_name = self.__class__.__name__
        obj_str += '\t\t' + class_name + ' instance\n'
        obj_str += '-----------------------------------------------------------------------------------\n'

        # Print pickled attributes
        for att in self.__toprint:
            obj_str += att + ': ' + str(self.__dict__[att])
            obj_str += '\n'

        obj_str += '\n'
        obj_str += 'Params:\n\t'
        obj_str += "\n\t".join([str(key) + ": " + str(self.params[key]) for key in sorted(self.params.keys())])
        obj_str += '\n'
        obj_str += '-----------------------------------------------------------------------------------'

        return obj_str


    def setParams(self, params):
        self.params = params


    def setTask(self):
        # use to normalise the task label
        normalised_task_labels = {
                "word": "word",
                "phrase": "phrase",
                "sent": "sent",
                "sentence": "sent",
                "doc": "doc",
                "document": "doc"
                }

        if 'TASK' in self.params:
            # we retrieve the requested task
            task_label = self.params['TASK'].lower()
            if not task_label in normalised_task_labels:
                logger.warning("(QEModel, setTask()) The required task hasn't been recognised: {}".format(req_task))
                raise ValueError("(QEModel, setTask()) The required task hasn't been recognised: {}".format(req_task))
            else:
                norm_task_label = normalised_task_labels[task_label]
                if not norm_task_label in self._task_levels:
                    logger.warning('The model {} does not support {}-level QE. Supported tasks are: {}'.format(self.__class__.__name__, norm_task_label, *self._tasks))
                    raise ValueError("The model {} does not support {}-level QE. Supported tasks are: {}".format(self.__class__.__name__, norm_task_label, *self._tasks))
                else:
                    self.task_level = norm_task_label
        else:
            # default task: sentence-level QE
            logger.warning('No specific task label given.')
            raise ValueError("No specific task label given.")


    def setOptimizer(self, **kwargs):
        """
        Sets and compiles a new optimizer for the Translation_Model.
        The configuration is read from Translation_Model.params.
        :return: None
        """
        if int(self.params.get('ACCUMULATE_GRADIENTS', 1)) > 1 and self.params['OPTIMIZER'].lower() != 'adam':
            logger.warning('Gradient accumulate is only implemented for the Adam optimizer. Setting "ACCUMULATE_GRADIENTS" to 1.')
            self.params['ACCUMULATE_GRADIENTS'] = 1

        optimizer_str = '\t LR: ' + str(self.params.get('LR', 0.01)) + \
                        '\n\t LOSS: ' + str(self.params.get('LOSS', 'categorical_crossentropy'))

        if self.params.get('USE_TF_OPTIMIZER', False) and K.backend() == 'tensorflow':
            if self.params['OPTIMIZER'].lower() not in ['sgd', 'adagrad', 'adadelta', 'rmsprop', 'adam']:
                logger.warning('The optimizer %s is not natively implemented in Tensorflow. Using the Keras version.' % (str(self.params['OPTIMIZER'])))
            if self.params.get('LR_DECAY') is not None:
                logger.warning('The learning rate decay is not natively implemented in native Tensorflow optimizers. Using the Keras version.')
                self.params['USE_TF_OPTIMIZER'] = False
            if self.params.get('ACCUMULATE_GRADIENTS', 1) > 1:
                logger.warning('The gradient accumulation is not natively implemented in native Tensorflow optimizers. Using the Keras version.')
                self.params['USE_TF_OPTIMIZER'] = False

        if self.params.get('USE_TF_OPTIMIZER', False) and K.backend() == 'tensorflow' and self.params['OPTIMIZER'].lower() in ['sgd', 'adagrad', 'adadelta', 'rmsprop', 'adam']:
            import tensorflow as tf
            if self.params['OPTIMIZER'].lower() == 'sgd':
                if self.params.get('MOMENTUM') is None:
                    optimizer = TFOptimizer(tf.train.GradientDescentOptimizer(self.params.get('LR', 0.01)))
                else:
                    optimizer = TFOptimizer(tf.train.MomentumOptimizer(self.params.get('LR', 0.01),
                                                                       self.params.get('MOMENTUM', 0.0),
                                                                       use_nesterov=self.params.get('NESTEROV_MOMENTUM', False)))
                    optimizer_str += '\n\t MOMENTUM: ' + str(self.params.get('MOMENTUM', 0.0)) + \
                                     '\n\t NESTEROV: ' + str(self.params.get('NESTEROV_MOMENTUM', False))

            elif self.params['OPTIMIZER'].lower() == 'adam':
                optimizer = TFOptimizer(tf.train.AdamOptimizer(learning_rate=self.params.get('LR', 0.01),
                                                               beta1=self.params.get('BETA_1', 0.9),
                                                               beta2=self.params.get('BETA_2', 0.999),
                                                               epsilon=self.params.get('EPSILON', 1e-7)))
                optimizer_str += '\n\t BETA_1: ' + str(self.params.get('BETA_1', 0.9)) + \
                                 '\n\t BETA_2: ' + str(self.params.get('BETA_2', 0.999)) + \
                                 '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))

            elif self.params['OPTIMIZER'].lower() == 'adagrad':
                optimizer = TFOptimizer(tf.train.AdagradOptimizer(self.params.get('LR', 0.01)))

            elif self.params['OPTIMIZER'].lower() == 'rmsprop':
                optimizer = TFOptimizer(tf.train.RMSPropOptimizer(self.params.get('LR', 0.01),
                                                                  decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                                                  momentum=self.params.get('MOMENTUM', 0.0),
                                                                  epsilon=self.params.get('EPSILON', 1e-7)))
                optimizer_str += '\n\t MOMENTUM: ' + str(self.params.get('MOMENTUM', 0.9)) + \
                                 '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))

            elif self.params['OPTIMIZER'].lower() == 'adadelta':
                optimizer = TFOptimizer(tf.train.AdadeltaOptimizer(learning_rate=self.params.get('LR', 0.01),
                                                                   rho=self.params.get('RHO', 0.95),
                                                                   epsilon=self.params.get('EPSILON', 1e-7)))
                optimizer_str += '\n\t RHO: ' + str(self.params.get('RHO', 0.9)) + \
                                 '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))

            else:
                raise Exception('\tThe chosen optimizer is not implemented.')
        else:
            if self.params['OPTIMIZER'].lower() == 'sgd':
                optimizer = SGD(lr=self.params.get('LR', 0.01),
                                momentum=self.params.get('MOMENTUM', 0.0),
                                decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                nesterov=self.params.get('NESTEROV_MOMENTUM', False),
                                clipnorm=self.params.get('CLIP_C', 0.),
                                clipvalue=self.params.get('CLIP_V', 0.))
                optimizer_str += '\n\t MOMENTUM: ' + str(self.params.get('MOMENTUM', 0.0)) + \
                                 '\n\t NESTEROV: ' + str(self.params.get('NESTEROV_MOMENTUM', False))

            elif self.params['OPTIMIZER'].lower() == 'rsmprop':
                optimizer = RMSprop(lr=self.params.get('LR', 0.001),
                                    rho=self.params.get('RHO', 0.9),
                                    decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                    clipnorm=self.params.get('CLIP_C', 0.),
                                    clipvalue=self.params.get('CLIP_V', 0.),
                                    epsilon=self.params.get('EPSILON', 1e-7))
                optimizer_str += '\n\t RHO: ' + str(self.params.get('RHO', 0.9)) + \
                                 '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))

            elif self.params['OPTIMIZER'].lower() == 'adagrad':
                optimizer = Adagrad(lr=self.params.get('LR', 0.01),
                                    decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                    clipnorm=self.params.get('CLIP_C', 0.),
                                    clipvalue=self.params.get('CLIP_V', 0.),
                                    epsilon=self.params.get('EPSILON', 1e-7))
                optimizer_str += '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))

            elif self.params['OPTIMIZER'].lower() == 'adadelta':
                optimizer = Adadelta(lr=self.params.get('LR', 1.0),
                                     rho=self.params.get('RHO', 0.9),
                                     decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                     clipnorm=self.params.get('CLIP_C', 0.),
                                     clipvalue=self.params.get('CLIP_V', 0.),
                                     epsilon=self.params.get('EPSILON', 1e-7))
                optimizer_str += '\n\t RHO: ' + str(self.params.get('RHO', 0.9)) + \
                                 '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))

            elif self.params['OPTIMIZER'].lower() == 'adam':
                if self.params.get('ACCUMULATE_GRADIENTS', 1) > 1:
                    optimizer = AdamAccumulate(lr=self.params.get('LR', 0.001),
                                               beta_1=self.params.get('BETA_1', 0.9),
                                               beta_2=self.params.get('BETA_2', 0.999),
                                               amsgrad=self.params.get('AMSGRAD', False),
                                               decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                               clipnorm=self.params.get('CLIP_C', 0.),
                                               clipvalue=self.params.get('CLIP_V', 0.),
                                               epsilon=self.params.get('EPSILON', 1e-7),
                                               accum_iters=self.params.get('ACCUMULATE_GRADIENTS'))
                    optimizer_str += '\n\t BETA_1: ' + str(self.params.get('BETA_1', 0.9)) + \
                                     '\n\t BETA_2: ' + str(self.params.get('BETA_2', 0.999)) + \
                                     '\n\t AMSGRAD: ' + str(self.params.get('AMSGRAD', False)) + \
                                     '\n\t ACCUMULATE_GRADIENTS: ' + str(self.params.get('ACCUMULATE_GRADIENTS')) + \
                                     '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))
                else:
                    optimizer = Adam(lr=self.params.get('LR', 0.001),
                                     beta_1=self.params.get('BETA_1', 0.9),
                                     beta_2=self.params.get('BETA_2', 0.999),
                                     amsgrad=self.params.get('AMSGRAD', False),
                                     decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                     clipnorm=self.params.get('CLIP_C', 0.),
                                     clipvalue=self.params.get('CLIP_V', 0.),
                                     epsilon=self.params.get('EPSILON', 1e-7))
                    optimizer_str += '\n\t BETA_1: ' + str(self.params.get('BETA_1', 0.9)) + \
                                     '\n\t BETA_2: ' + str(self.params.get('BETA_2', 0.999)) + \
                                     '\n\t AMSGRAD: ' + str(self.params.get('AMSGRAD', False)) + \
                                     '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))

            elif self.params['OPTIMIZER'].lower() == 'adamax':
                optimizer = Adamax(lr=self.params.get('LR', 0.002),
                                   beta_1=self.params.get('BETA_1', 0.9),
                                   beta_2=self.params.get('BETA_2', 0.999),
                                   decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                   clipnorm=self.params.get('CLIP_C', 0.),
                                   clipvalue=self.params.get('CLIP_V', 0.),
                                   epsilon=self.params.get('EPSILON', 1e-7))
                optimizer_str += '\n\t BETA_1: ' + str(self.params.get('BETA_1', 0.9)) + \
                                 '\n\t BETA_2: ' + str(self.params.get('BETA_2', 0.999)) + \
                                 '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))
            elif self.params['OPTIMIZER'].lower() == 'nadam':
                optimizer = Nadam(lr=self.params.get('LR', 0.002),
                                  beta_1=self.params.get('BETA_1', 0.9),
                                  beta_2=self.params.get('BETA_2', 0.999),
                                  schedule_decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                  clipnorm=self.params.get('CLIP_C', 0.),
                                  clipvalue=self.params.get('CLIP_V', 0.),
                                  epsilon=self.params.get('EPSILON', 1e-7))
                optimizer_str += '\n\t BETA_1: ' + str(self.params.get('BETA_1', 0.9)) + \
                                 '\n\t BETA_2: ' + str(self.params.get('BETA_2', 0.999)) + \
                                 '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))

            elif self.params['OPTIMIZER'].lower() == 'sgdhd':
                optimizer = SGDHD(lr=self.params.get('LR', 0.002),
                                  clipnorm=self.params.get('CLIP_C', 10.),
                                  clipvalue=self.params.get('CLIP_V', 0.),
                                  hypergrad_lr=self.params.get('HYPERGRAD_LR', 0.001))
                optimizer_str += '\n\t HYPERGRAD_LR: ' + str(self.params.get('HYPERGRAD_LR', 0.001))

            elif self.params['OPTIMIZER'].lower() == 'qhsgd':
                optimizer = QHSGD(lr=self.params.get('LR', 0.002),
                                  momentum=self.params.get('MOMENTUM', 0.0),
                                  quasi_hyperbolic_momentum=self.params.get('QUASI_HYPERBOLIC_MOMENTUM', 0.0),
                                  decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                  nesterov=self.params.get('NESTEROV_MOMENTUM', False),
                                  dampening=self.params.get('DAMPENING', 0.),
                                  clipnorm=self.params.get('CLIP_C', 10.),
                                  clipvalue=self.params.get('CLIP_V', 0.))
                optimizer_str += '\n\t MOMENTUM: ' + str(self.params.get('MOMENTUM', 0.0)) + \
                                 '\n\t QUASI_HYPERBOLIC_MOMENTUM: ' + str(self.params.get('QUASI_HYPERBOLIC_MOMENTUM', 0.0)) + \
                                 '\n\t DAMPENING: ' + str(self.params.get('DAMPENING', 0.0)) + \
                                 '\n\t NESTEROV: ' + str(self.params.get('NESTEROV_MOMENTUM', False))

            elif self.params['OPTIMIZER'].lower() == 'qhsgdhd':
                optimizer = QHSGDHD(lr=self.params.get('LR', 0.002),
                                    momentum=self.params.get('MOMENTUM', 0.0),
                                    quasi_hyperbolic_momentum=self.params.get('QUASI_HYPERBOLIC_MOMENTUM', 0.0),
                                    dampening=self.params.get('DAMPENING', 0.),
                                    hypergrad_lr=self.params.get('HYPERGRAD_LR', 0.001),
                                    decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                    nesterov=self.params.get('NESTEROV_MOMENTUM', False),
                                    clipnorm=self.params.get('CLIP_C', 10.),
                                    clipvalue=self.params.get('CLIP_V', 0.))
                optimizer_str += '\n\t MOMENTUM: ' + str(self.params.get('MOMENTUM', 0.0)) + \
                                 '\n\t QUASI_HYPERBOLIC_MOMENTUM: ' + str(self.params.get('QUASI_HYPERBOLIC_MOMENTUM', 0.0)) + \
                                 '\n\t HYPERGRAD_LR: ' + str(self.params.get('HYPERGRAD_LR', 0.001)) + \
                                 '\n\t DAMPENING: ' + str(self.params.get('DAMPENING', 0.0)) + \
                                 '\n\t NESTEROV: ' + str(self.params.get('NESTEROV_MOMENTUM', False))

            elif self.params['OPTIMIZER'].lower() == 'adadeltahd':
                optimizer = AdadeltaHD(lr=self.params.get('LR', 0.002),
                                       hypergrad_lr=self.params.get('HYPERGRAD_LR', 0.001),
                                       rho=self.params.get('RHO', 0.9),
                                       decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                       epsilon=self.params.get('EPSILON', 1e-7),
                                       clipnorm=self.params.get('CLIP_C', 10.),
                                       clipvalue=self.params.get('CLIP_V', 0.))
                optimizer_str += '\n\t HYPERGRAD_LR: ' + str(self.params.get('HYPERGRAD_LR', 0.001)) + \
                                 '\n\t RHO: ' + str(self.params.get('RHO', 0.9)) + \
                                 '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))

            elif self.params['OPTIMIZER'].lower() == 'adamhd':
                optimizer = AdamHD(lr=self.params.get('LR', 0.002),
                                   hypergrad_lr=self.params.get('HYPERGRAD_LR', 0.001),
                                   beta_1=self.params.get('BETA_1', 0.9),
                                   beta_2=self.params.get('BETA_2', 0.999),
                                   decay=self.params.get('LR_OPTIMIZER_DECAY', 0.0),
                                   clipnorm=self.params.get('CLIP_C', 10.),
                                   clipvalue=self.params.get('CLIP_V', 0.),
                                   epsilon=self.params.get('EPSILON', 1e-7))
                optimizer_str += '\n\t HYPERGRAD_LR: ' + str(self.params.get('HYPERGRAD_LR', 0.001)) + \
                                 '\n\t BETA_1: ' + str(self.params.get('BETA_1', 0.9)) + \
                                 '\n\t BETA_2: ' + str(self.params.get('BETA_2', 0.999)) + \
                                 '\n\t EPSILON: ' + str(self.params.get('EPSILON', 1e-7))
            else:
                logger.info('\tWARNING: The modification of the LR is not implemented for the chosen optimizer.')
                optimizer = eval(self.params['OPTIMIZER'])

            optimizer_str += '\n\t CLIP_C ' + str(self.params.get('CLIP_C', 0.)) + \
                             '\n\t CLIP_V ' + str(self.params.get('CLIP_V', 0.)) + \
                             '\n\t LR_OPTIMIZER_DECAY ' + str(self.params.get('LR_OPTIMIZER_DECAY', 0.0)) + \
                             '\n\t ACCUMULATE_GRADIENTS ' + str(self.params.get('ACCUMULATE_GRADIENTS', 1)) + '\n'
        if self.verbose > 0:
            logger.info("Preparing optimizer and compiling. Optimizer configuration: \n" + optimizer_str)


        sample_weight_mode = []
        sample_weight_dict = self.params['SAMPLE_WEIGHTS'] 

        for out_id in self.ids_outputs:

            if out_id in sample_weight_dict:
                sample_weight_mode.append('temporal')
            else:
                sample_weight_mode.append(None)  
        

        if hasattr(self, 'multi_gpu_model') and self.multi_gpu_model is not None:
            model_to_compile = self.multi_gpu_model
        else:
            model_to_compile = self.model

        model_to_compile.compile(optimizer=optimizer,
                                 loss=self.params['LOSS'],
                                 metrics=self.params.get('KERAS_METRICS', []),
                                 loss_weights=self.params.get('LOSS_WEIGHTS', None),
                                 sample_weight_mode=sample_weight_mode,
                                 weighted_metrics=self.params.get('KERAS_METRICS_WEIGHTS', None),
                                 target_tensors=self.params.get('TARGET_TENSORS'))
