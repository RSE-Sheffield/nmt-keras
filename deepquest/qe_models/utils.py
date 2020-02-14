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
Imports modules and defines functions necessary to create the models
"""

from keras.layers import *
# from keras.layers.noise import GaussianNoise
# from keras.layers.advanced_activations import ChannelWisePReLU as PReLU
# from keras.layers.normalization import BatchNormalization, L2_norm
# from keras.layers.core import Dropout, Lambda
from keras.models import model_from_json, Model
from keras.optimizers import *
from keras.regularizers import l2, AlphaRegularizer
from keras.regularizers import l2

from keras_wrapper.cnn_model import Model_Wrapper


def Regularize(layer, params,
               shared_layers=False,
               name='',
               apply_noise=True,
               apply_batch_normalization=True,
               apply_prelu=True,
               apply_dropout=True,
               apply_l2=True,
               trainable=True):
    """
    Apply the regularization specified in parameters to the layer
    :param layer: Layer to regularize
    :param params: Params specifying the regularizations to apply
    :param shared_layers: Boolean indicating if we want to get the used layers for applying to a shared-layers model.
    :param name: Name prepended to regularizer layer
    :param apply_noise: If False, noise won't be applied, independently of params
    :param apply_dropout: If False, dropout won't be applied, independently of params
    :param apply_prelu: If False, prelu won't be applied, independently of params
    :param apply_batch_normalization: If False, batch normalization won't be applied, independently of params
    :param apply_l2: If False, l2 normalization won't be applied, independently of params
    :return: Regularized layer
    """
    shared_layers_list = []

    if apply_noise and params.get('USE_NOISE', False):
        shared_layers_list.append(GaussianNoise(params.get('NOISE_AMOUNT', 0.01), name=name + '_gaussian_noise', trainable=trainable))

    if apply_batch_normalization and params.get('USE_BATCH_NORMALIZATION', False):
        if params.get('WEIGHT_DECAY'):
            l2_gamma_reg = l2(params['WEIGHT_DECAY'])
            l2_beta_reg = l2(params['WEIGHT_DECAY'])
        else:
            l2_gamma_reg = None
            l2_beta_reg = None

        bn_mode = params.get('BATCH_NORMALIZATION_MODE', 0)

        shared_layers_list.append(BatchNormalization(mode=bn_mode,
                                                     gamma_regularizer=l2_gamma_reg,
                                                     beta_regularizer=l2_beta_reg,
                                                     name=name + '_batch_normalization',
                                                     trainable=trainable))

    if apply_prelu and params.get('USE_PRELU', False):
        shared_layers_list.append(PReLU(name=name + '_PReLU', trainable=trainable))

    if apply_dropout and params.get('DROPOUT_P', 0) > 0:
        shared_layers_list.append(Dropout(params.get('DROPOUT_P', 0.5), name=name + '_dropout', trainable=trainable))

    if apply_l2 and params.get('USE_L2', False):
        shared_layers_list.append(Lambda(L2_norm, name=name + '_L2_norm', trainable=trainable))

    # Apply all the previously built shared layers
    for l in shared_layers_list:
        layer = l(layer)
    result = layer

    # Return result or shared layers too
    if shared_layers:
        return result, shared_layers_list
    return result

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


def mask_aware_mean_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    return (shape[0], shape[2])


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

