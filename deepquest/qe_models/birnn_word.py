# -*- coding: utf-8 -*-
#
# birnn_word.py
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

from .model import QEModel
from .utils import *
from .layers import *

class EncWord(QEModel):

    def __init__(self, params):
        # define here attributes that are model specific
        if 'bert' in params['TOKENIZATION_METHOD'].lower():
            self.use_bert = True
            self.bert_pooling = None
        else:
            self.use_bert = False

        # and init from the QEModel class
        super().__init__(params)


    def build(self):
        """
        Defines the model architecture using parameters from self.params,
        and instantiate a Model object accordingly.
        """

        params = self.params

        if self.use_bert:
            # if we use BERT, we extend the list on ids_inputs to 
            # match with the dataset ids_inputs
            self.ids_inputs.extend([
                params['INPUTS_IDS_MODEL'][0] + '_mask',
                params['INPUTS_IDS_MODEL'][0] + '_segids',
                params['INPUTS_IDS_MODEL'][1] + '_mask',
                params['INPUTS_IDS_MODEL'][1] + '_segids'
                ])

            bert_layer = BertLayer(
                    max_seq_len=params.get('MAX_INPUT_TEXT_LEN', 70),
                    pooling=self.bert_pooling,
                    trainable=True,
                    fine_tune_n_layers=params.get('BERT_FINE_TUNE_N_LAYER', 3),
                    tune_embeddings = False,
                    verbose=True,
                    name="bert_embedding_layer"
                    )

        #######################################################################
        ####      INPUTS OF THE MODEL                                      ####
        #######################################################################
        src_words = Input(name=self.ids_inputs[0],
                batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]),
                dtype='int32')

        trg_words = Input(name=self.ids_inputs[1],
                batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]),
                dtype='int32')


        #######################################################################
        ####      ENCODERS                                                 ####
        #######################################################################
        ## SOURCE encoder
        if self.use_bert:
            src_words_mask = Input(name=self.ids_inputs[0] + '_mask',
            # src_words_mask = Input(name=self.ids_inputs[2],
                    batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]),
                    dtype='int32')

            src_words_segids = Input(name=self.ids_inputs[0] + '_segids',
            # src_words_segids = Input(name=self.ids_inputs[3],
                    batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]),
                    dtype='int32')

            src_bert_input = [src_words, src_words_mask, src_words_segids]

            src_embedding = bert_layer(src_bert_input)
        else:
            src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'],
                    params['SOURCE_TEXT_EMBEDDING_SIZE'],
                    embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                    embeddings_initializer=params['INIT_FUNCTION'],
                    trainable=self.trainable,
                    mask_zero=True,
                    name='src_word_embedding'
                    )(src_words)

        src_embedding = Regularize(
                src_embedding,
                params,
                trainable=self.trainable,
                name='src_state'
                )

        src_annotations = Bidirectional(
                eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                    kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                    recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                    bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                    dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                    recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                    kernel_initializer=params['INIT_FUNCTION'],
                    recurrent_initializer=params['INNER_INIT'],
                    return_sequences=True,
                    trainable=self.trainable),
                merge_mode='concat',
                name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE']
                )(src_embedding)


        ## TARGET encoder
        if self.use_bert:
            trg_words_mask = Input(name=self.ids_inputs[1] + '_mask',
            # trg_words_mask = Input(name=self.ids_inputs[4],
                    batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]),
                    dtype='int32')

            trg_words_segids = Input(name=self.ids_inputs[1] + '_segids',
            # trg_words_segids = Input(name=self.ids_inputs[5],
                    batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]),
                    dtype='int32')
            trg_bert_input = [trg_words, trg_words_mask, trg_words_segids]

            trg_embedding = bert_layer(trg_bert_input)

        else:
            trg_embedding = Embedding(params['OUTPUT_VOCABULARY_SIZE'],
                    params['TARGET_TEXT_EMBEDDING_SIZE'],
                    embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                    embeddings_initializer=params['INIT_FUNCTION'],
                    trainable=self.trainable,
                    mask_zero=True,
                    name='target_word_embedding'
                    )(trg_words)

        trg_embedding = Regularize(
                trg_embedding,
                params,
                trainable=self.trainable,
                name='trg_state'
                )

        trg_annotations = Bidirectional(
                eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                    kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                    recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                    bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                    dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                    recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                    kernel_initializer=params['INIT_FUNCTION'],
                    recurrent_initializer=params['INNER_INIT'],
                    return_sequences=True,
                    trainable=self.trainable),
                merge_mode='concat',
                name='trg_bidirectional_encoder_' + params['ENCODER_RNN_TYPE']
                )(trg_embedding)


        #######################################################################
        ####      DECODERS                                                 ####
        #######################################################################
        ## Concatenation of the two (src, trg) sentence-level representations
        annotations = concatenate(
                [src_annotations, trg_annotations],
                name='annot_seq_concat'
                )


        #######################################################################
        ####      OUTPUTS OF THE MODEL                                     ####
        #######################################################################
        out_activation = params.get('OUT_ACTIVATION', 'sigmoid')

        output_qe_layer = TimeDistributed(
                Dense(
                    params['WORD_QE_CLASSES'],
                    activation=out_activation
                    ),
                name=self.ids_outputs[0]
                )(annotations)

        # instantiating a Model object
        if self.use_bert:
            inputs=[src_words, src_words_mask, src_words_segids, trg_words, trg_words_mask, trg_words_segids]
        else:
            inputs=[src_words, trg_words]

        self.model = Model(
                inputs=inputs,
                outputs=[output_qe_layer]
                )

