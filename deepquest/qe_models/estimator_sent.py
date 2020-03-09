#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# estimator_sent.py
#
# Copyright (C) 2019 Frederic Blain (feedoo) <f.blain@sheffield.ac.uk>
#
# Licensed under the "THE BEER-WARE LICENSE" (Revision 42):
# Fred (feedoo) Blain wrote this file. As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a tomato juice or coffee in return
#

"""
#TODO: add a description of the model here.
#TODO: add reference

#======================================================
# Sentence-level QE -- POSTECH-inspired Estimator model
#======================================================
#
## Inputs:
# 1. Sentences in src language (shape: (mini_batch_size, words))
# 2. One-position left-shifted machine-translated sentences to represent the right context (shape: (mini_batch_size, words))
# 3. One-position rigth-shifted machine-translated sentences to represent the left context (shape: (mini_batch_size, words))
# 4. Unshifted machine-translated sentences for evaluation (shape: (mini_batch_size, words))
#
## Output:
# 1. Sentence quality scores (shape: (mini_batch_size,))
#
## References:
# - Hyun Kim, Hun-Young Jung, Hongseok Kwon, Jong-Hyeok Lee, and Seung-Hoon Na. 2017a. Predictor- estimator: Neural quality estimation based on target word prediction for machine translation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 17(1):3:1-3:22, September.

"""

from .estimator_word import estimatorword
from .utils import *


class estimatorsent(estimatorword):

    def __init__(self, params, model_type='EstimatorSent',
            verbose=1, structure_path=None, weights_path=None,
            model_name=None, vocabularies=None, store_path=None,
            set_optimizer=True, clear_dirs=True):

        # define here attributes that are model specific
        self.trainable_pred = params.get('TRAIN_PRED', True)
        self.trainable_est = params.get('TRAIN_EST', True)

        # and init from the QEModel class
        super().__init__(params)


    def build(self):

        super().build()
        params = self.params

        enc_qe_frw, last_state_frw = self.model.get_layer('qe_frw').output
        enc_qe_bkw, last_state_bkw = self.model.get_layer('qe_bkw').output

        last_state_concat = concatenate([last_state_frw, last_state_bkw],
                trainable=self.trainable_est,
                name='last_state_concat'
                )

        # uncomment for Post QE
        # fin_seq = concatenate([seq_concat, merged_states])
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')

        output_qe_layer = Dense(1,
                activation=out_activation,
                trainable=self.trainable_est,
                name=self.ids_outputs[0]
                )(last_state_concat)

        # self.model = Model(inputs=[src_text, next_words, next_words_bkw], outputs=[merged_states,softout, softoutQE])
        # if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0.:
        #     self.model.add_loss(alpha_regularizer)
        self.model = Model(
                inputs=self.model.input,
                outputs=[output_qe_layer]
                )
