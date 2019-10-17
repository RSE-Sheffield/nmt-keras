#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# utils.py
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

from keras.layers import *


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


def attention_3d_block(inputs, params, ext):
    '''
    simple attention: weights over time steps; as in https://github.com/philipperemy/keras-attention-mechanism
    '''
    # inputs.shape = (batch_size, time_steps, input_dim)
    TIME_STEPS = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]

    a = Permute((2, 1))(inputs)
    a = Dense(TIME_STEPS, activation='softmax', name='soft_att' + ext)(a)
    a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction' + ext, output_shape=(TIME_STEPS,))(a)
    a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec' + ext)(a)

    output_attention_mul = multiply([inputs, a_probs], name='attention_mul' + ext)
    sum = Lambda(reduce_sum, mask_aware_mean_output_shape)
    output = sum(output_attention_mul)

    return output


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


def mask_aware_mean(x):
    '''
    see: https://github.com/keras-team/keras/issues/1579
    '''
    # recreate the masks - all zero rows have been masked
    mask = K.not_equal(K.sum(K.abs(x), axis=2, keepdims=True), 0)
    # number of that rows are not all zeros
    n = K.sum(K.cast(mask, 'float32'), axis=1, keepdims=False)

    x_mean = K.sum(x, axis=1, keepdims=False)
    x_mean = x_mean / n

    return x_mean


def mask_aware_mean4d(x):
    '''
    see: https://github.com/keras-team/keras/issues/1579
    '''
    # recreate the masks - all zero rows have been masked
    mask = K.not_equal(K.sum(K.abs(x), axis=3, keepdims=True), 0)
    # number of that rows are not all zeros
    n = K.sum(K.cast(mask, 'float32'), axis=2, keepdims=False)

    x_mean = K.sum(x, axis=2, keepdims=False)
    x_mean = x_mean / n

    return x_mean


def mask_aware_merge_output_shape4d(input_shape):
    shape = list(input_shape)
    assert len(shape) == 4
    return (shape[0], shape[1], shape[3])


def merge(x, params, dim):
    return Lambda(lambda x: K.stack(x,axis=1), output_shape=(params['MAX_OUTPUT_TEXT_LEN'], dim * 2))(x)


def one_hot(x, params):
    return Lambda(lambda x: K.one_hot(x, params['OUTPUT_VOCABULARY_SIZE']), output_shape=(None, params['OUTPUT_VOCABULARY_SIZE']))(x)


def reduce_max(x):
    return K.max(x, axis=1, keepdims=False)


def reduce_sum(x):
    return K.sum(x, axis=1, keepdims=False)


def sum4d(x):
    return K.sum(x, axis=2, keepdims=False)

