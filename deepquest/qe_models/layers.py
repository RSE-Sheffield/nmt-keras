#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# layers.py
#
# Copyright (C) 2019 Frederic Blain (feedoo) <f.blain@sheffield.ac.uk>
#
# Licensed under the "THE BEER-WARE LICENSE" (Revision 42):
# Fred (feedoo) Blain wrote this file. As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a tomato juice or coffee in return
#

"""
Defines the layers used in our QE models.
"""

from keras.engine.base_layer import Layer
from keras.layers import *
# from keras.layers.noise import GaussianNoise
# from keras.layers.advanced_activations import ChannelWisePReLU as PReLU
# from keras.layers.normalization import BatchNormalization, L2_norm
# from keras.layers.core import Dropout, Lambda
from keras.models import model_from_json, Model
from keras.utils import multi_gpu_model
from keras.optimizers import *
from keras.regularizers import l2, AlphaRegularizer
from keras.regularizers import l2

from keras_wrapper.cnn_model import Model_Wrapper

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import backend as K



class BertLayer(Layer):
    def __init__(self,
            bert_path="https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1",
            max_seq_len=70,
            pooling=None,
            trainable=True,
            fine_tune_n_layers=3,
            tune_embeddings=False,
            return_seq_output=False,
            verbose=False,
            name='bert_layer', **kwargs):

            # output_representation='sequence_output', tune_embeddings=False, trainable=True, **kwargs):
        self.bert = None
        self.bert_path = bert_path

        '''
        POOLING strategies:
        - pooling==None, no pooling is applied and the output tensor has shape
          [batch_size, seq_len, encoder_dim]. This mode is useful for solving
          token level tasks.
        - pooling==’mean’, the embeddings for all tokens are mean-pooled and
          the output tensor has shape [batch_size, encoder_dim]. This mode is
          particularly useful for sentence representation tasks.
        - pooling==’cls’, only the vector corresponding to first ‘CLS’ token is
          retrieved and the output tensor has shape [batch_size, encoder_dim].
          This pooling type is useful for solving sentence-pair classification
          tasks.
        '''
        self.pooling = pooling
        if self.pooling not in ["cls", "mean", None]:
            raise NameError(
                    f"Undefined pooling type (must be either 'cls', 'mean', or \
                    None, but is {self.pooling}"
                    )

        self.fine_tune_n_layers = fine_tune_n_layers
        self.trainable = trainable
        self.tune_embeddings = tune_embeddings

        self.max_seq_len = max_seq_len
        self.return_seq_output = return_seq_output

        self.var_per_encoder = 16

        self.verbose = verbose

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(self.bert_path,
                               trainable=self.trainable, name="{}_module".format(self.name))

        trainable_layers = []
        if self.tune_embeddings:
            trainable_layers.append("embeddings")

        if self.pooling == "cls":
            trainable_layers.append("pooler")

        if self.fine_tune_n_layers > 0:
            encoder_var_names = [var.name for var in self.bert.variables if 'encoder' in var.name]
            n_encoder_layers = int(len(encoder_var_names) / self.var_per_encoder)
            for i in range(self.fine_tune_n_layers):
                trainable_layers.append(f"encoder/layer_{str(n_encoder_layers - 1 - i)}/")

        # Add module variables to layer's trainable weights
        for var in self.bert.variables:
            if any([l in var.name for l in trainable_layers]):
                self._trainable_weights.append(var)
            else:
                self._non_trainable_weights.append(var)

        if self.verbose:
            print("*** BERT LAYER - TRAINABLE VARS ***")
            for var in self._trainable_weights:
                print(var)
            print("***********************************")

        super(BertLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs

        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )

        output = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)

        if self.pooling == "cls":
            pooled = output["pooled_output"]
        else:
            result = output["sequence_output"]

            input_mask = tf.cast(input_mask, tf.float32)
            mul_mask = lambda x, m: x * tf.expand_dims(m, axis=-1)
            masked_reduce_mean = lambda x, m: tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)

            if self.pooling == "mean":
                pooled = masked_reduce_mean(result, input_mask)
            else:
                pooled = mul_mask(result, input_mask)

        if self.return_seq_output:
            return_elements = [pooled, result]
        else:
            return_elements = pooled

        return return_elements

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        if self.pooling == None:
            return (None, self.max_seq_len, 768)
        else: #'mean' and 'cls'
            return (None, 768)


class NonMasking(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class DenseTranspose(Layer):

  def __init__(self, output_dim, other_layer, other_layer_name, **kwargs):
      self.output_dim = output_dim
      self.other_layer=other_layer
      self.other_layer_name = other_layer_name
      super(DenseTranspose, self).__init__(**kwargs)

  def call(self, x):
      # w = self.other_layer.get_layer(self.other_layer_name).layer.kernel
      w = self.other_layer.layer.kernel
      w_trans = K.transpose(w)
      return K.dot(x, w_trans)

  def compute_output_shape(self, input_shape):
      return (input_shape[0], input_shape[1], self.output_dim)


class Reverse(Layer):

    def __init__(self, output_dim, axes, **kwargs):
        self.output_dim = output_dim
        self.axes = axes
        self.supports_masking = True
        super(Reverse, self).__init__(**kwargs)

    def call(self, x):
        return K.reverse(x, axes=self.axes)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)

    def compute_mask(self, inputs, mask):
        if isinstance(mask, list):
            mask = mask[0]
        else:
            mask= None
        return mask


class GeneralReshape(Layer):

    def __init__(self, output_dim, params, **kwargs):
        self.output_dim = output_dim
        self.params = params
        super(GeneralReshape, self).__init__(**kwargs)

    def call(self, x):
        if len(self.output_dim)==2:
            return K.reshape(x, (-1, self.params['MAX_INPUT_TEXT_LEN']))
        if len(self.output_dim)==3:
            return K.reshape(x, (-1, self.output_dim[1], self.output_dim[2]))
        if len(self.output_dim)==4:
            return K.reshape(x, (-1, self.output_dim[1], self.output_dim[2], self.output_dim[3]))

    def compute_output_shape(self, input_shape):
        return self.output_dim
