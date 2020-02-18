#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# Bertlayer.py
#
# Copyright (C) 2019 Frederic Blain (feedoo) <f.blain@sheffield.ac.uk>
#
# Licensed under the "THE BEER-WARE LICENSE" (Revision 42):
# Fred (feedoo) Blain wrote this file. As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a tomato juice or coffee in return
#

"""

"""

# from keras.layers import *
from keras.engine.base_layer import Layer

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import backend as K
# import keras_wrapper.extra.bert_tokenization as tokenization

class BertLayer(Layer):
    def __init__(self, max_seq_len=70, output_representation='sequence_output', trainable=True, **kwargs):
        self.bert = None
        super(BertLayer, self).__init__(**kwargs)

        self.bert_path="https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1"
        self.output_representation = output_representation
        self.max_seq_len = max_seq_len
        self.trainable = trainable

    def build(self, input_shape):
        self.bert = hub.Module(self.bert_path,
                               trainable=True, name="{}_module".format(self.name))

        # Remove unused layers and set trainable parameters
        if self.output_representation in ['sequence_output', 'cls_output']:
            # self.trainable_weights += [var for var in self.bert.variables
                                       # if not "/cls/" in var.name and not "/pooler/" in var.name]
            self.trainable_weights += [var for var in self.bert.variables
                                       if "layer_9" in var.name or "layer_10" in var.name or "layer_11" in var.name or "layer_12" in var.name]
            # self.trainable_weights += [var for var in self.bert.variables
                                       # if "layer_5" in var.name or "layer_6" in var.name or "layer_7" in var.name or "layer_8" in var.name or "layer_9" in var.name or "layer_10" in var.name or "layer_11" in var.name or "layer_12" in var.name]
            self.trainable_weights += [var for var in self.bert.variables
                                       if "embeddings/word_embeddings" in var.name]
            # self.trainable_weights += [var for var in self.bert.variables
                                       # if "/layer_" in var.name]
            # self.trainable_weights += [var for var in self.bert.variables
                                       # if "embeddings" in var.name]
        else:
            self.trainable_weights += [var for var in self.bert.variables
                                       if not "/cls/" in var.name]
        super(BertLayer, self).build(input_shape)

    def call(self, x, mask=None):
        inputs = dict(input_ids=x[0], input_mask=x[1], segment_ids=x[2])

        if self.output_representation in ['sequence_output', 'cls_output']:
            outputs = self.bert(inputs, as_dict=True, signature='tokens')['sequence_output']
        else:
            outputs = self.bert(inputs, as_dict=True, signature='tokens')['pooled_output']

        if self.output_representation == 'cls_output':
            return K.tf.squeeze(outputs[:, 0:1, :], axis=1)
        else:
            return outputs

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        if self.output_representation in ['pooled_output', 'cls_output']:
            return (None, 768)
        else:
            return (None, self.max_seq_len, 768)

