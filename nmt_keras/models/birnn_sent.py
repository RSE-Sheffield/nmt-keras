# -*- coding: utf-8 -*-
#
# birnn_sent.py
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

from .birnn_word import EncWord
from .utils import *


class EncSent(EncWord):

    def __init__(self, params):
        # define here attributes that are model specific

        # and init from the QEModel class
        super().__init__(params)


    def build(self):
        """
        Build the BiRNN QE model and return a "Model" object.
        #TODO: add reference
        """
        super().build()
        params = self.params

        annotations = self.model.get_layer('annot_seq_concat').output
        annotations = NonMasking()(annotations)
        # apply attention over words at the sentence-level
        annotations = attention_3d_block(annotations, params, 'sent')

        out_activation = params.get('OUT_ACTIVATION', 'sigmoid')

        output_qe_layer = Dense(1,
                activation=out_activation,
                name=self.ids_outputs[0]
                )(annotations)

        self.model = Model(
                inputs=self.model.input,
                output=output_qe_layer
                )
