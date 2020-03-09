# -*- coding: utf-8 -*-
#
# birnn_doc.py
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
"""

from .birnn_doc import encdoc
from .utils import *


class encdocatt(encdoc):

    def __init__(self, params):
        # define here attributes that are model specific

        # and init from the QEModel class
        super().__init__(params)


    def build(self):
        """
        Defines the model architecture using parameters from self.params,
        and instantiate a Model object accordingly.
        """

        super().build()
        params = self.params

        #######################################################################
        ####      DECODERS                                                 ####
        #######################################################################
        dec_doc_frw, dec_doc_last_state_frw = self.model.get_layer('dec_doc_frw').output
        dec_doc_bkw, dec_doc_last_state_bkw = self.model.get_layer('dec_doc_bkw').output

        dec_doc_bkw = Reverse(
                dec_doc_bkw._keras_shape[2],
                axes=1,
                name='dec_reverse_doc_bkw'
                )(dec_doc_bkw)

        dec_doc_seq_concat = concatenate(
                [dec_doc_frw, dec_doc_bkw],
                trainable=self.trainable_est,
                name='dec_doc_seq_concat'
                )
        # dec_doc_last_state_concat = self.model.get_layer('dec_doc_last_state_concat').output

        # dec_doc_seq_concat = self.model.get_layer('dec_doc_seq_concat').output

        dec_doc_seq_concat = NonMasking()(dec_doc_seq_concat)

        # apply attention over doc sentences
        attention_mul = attention_3d_block(dec_doc_seq_concat, params, 'doc')


        #######################################################################
        ####      OUTPUTS OF THE MODEL                                     ####
        #######################################################################
        out_activation = params.get('OUT_ACTIVATION', 'sigmoid')

        output_qe_layer = Dense(1,
                activation=out_activation,
                name=self.ids_outputs[0]
                )(attention_mul)

        # instantiating a Model object
        self.model = Model(
                inputs=self.model.input,
                outputs=[output_qe_layer]
                )
