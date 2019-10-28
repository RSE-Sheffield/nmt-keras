# -*- coding: utf-8 -*-
#
# birnn.py
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
"""

from nmt_keras.models.model import *


class BIRNN(QEModel):

    # list of QE task levels supported by the model
    _task_levels = ["word", "sent", "doc"]

    def __init__(self, params, model_type='BiRNN',
            verbose=1, structure_path=None, weights_path=None,
            model_name=None, vocabularies=None, store_path=None,
            set_optimizer=True, clear_dirs=True):
        # define here attributes that are model specific

        # and init from the QEModel class
        super().__init__(params)


    def build(self):
        """
        Defines the model architecture using parameters from self.params,
        and instantiate a Model object accordingly.
        """

        params = self.params

        #######################################################################
        ####      INPUTS OF THE MODEL                                      ####
        #######################################################################
        if self.task_level == "word" or self.task == "sent":
            src_words = Input(name=self.ids_inputs[0],
                    batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]),
                    dtype='int32')

            trg_words = Input(name=self.ids_inputs[1],
                    batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]),
                    dtype='int32')

        elif self.task_level == "doc":
            genreshape = GeneralReshape((None, params['MAX_INPUT_TEXT_LEN']), params)

            src_words = Input(name=self.ids_inputs[0],
                    batch_shape=tuple(
                        [None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]),
                    dtype='int32')
            # Reshape input to 2d to produce sent-level vectors
            src_words = genreshape(src_words)

            trg_words = Input(name=self.ids_inputs[1],
                    batch_shape=tuple(
                        [None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]),
                    dtype='int32')
            # Reshape input to 2d to produce sent-level vectors
            trg_words = genreshape(trg_words)


        #######################################################################
        ####      ENCODERS                                                 ####
        #######################################################################
        ## SOURCE encoder
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'],
                params['SOURCE_TEXT_EMBEDDING_SIZE'],
                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                embeddings_initializer=params['INIT_FUNCTION'],
                trainable=self.trainable,
                mask_zero=True,
                name='src_word_embedding'
                )(src_words)

        src_embedding = Regularize(src_embedding, params,
                trainable=self.trainable, name='src_state')

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

        if self.task_level == "sent":
            # apply attention over words at the sentence-level
            annotations = NonMasking()(annotations)
            annotations = attention_3d_block(annotations, params, 'sent')

        elif self.task_level == "doc":
            annotations = NonMasking()(annotations)
            annotations = attention_3d_block(annotations, params, 'sent')

            # reshape back to 3d input to group sent vectors per doc
            genreshape_out = GeneralReshape(
                    (None, params['SECOND_DIM_SIZE'], params['ENCODER_HIDDEN_SIZE'] * 4),
                    params
                    )
            annotations = genreshape_out(annotations)

            #bi-RNN over doc sentences
            #FIXME: there is no reason to do the BiRNN model manually, switch to Bidirectional Layer
            dec_doc_frw, dec_doc_last_state_frw = GRU(
                    params['DOC_DECODER_HIDDEN_SIZE'],
                    return_sequences=True,
                    return_state=True,
                    name='dec_doc_frw'
                    )(annotations)

            dec_doc_bkw, dec_doc_last_state_bkw = GRU(
                    params['DOC_DECODER_HIDDEN_SIZE'],
                    return_sequences=True,
                    return_state=True,
                    go_backwards=True,
                    name='dec_doc_bkw'
                    )(annotations)

            #TODO: check why this block is useles....
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
            #TODO: end of block

            # we take the last bi-RNN state as doc summary
            dec_doc_last_state_concat = concatenate(
                    [dec_doc_last_state_frw, dec_doc_last_state_bkw],
                    name='dec_doc_last_state_concat'
                    )



        #######################################################################
        ####      OUTPUTS OF THE MODEL                                     ####
        #######################################################################
        out_activation = params.get('OUT_ACTIVATION', 'sigmoid')

        if self.task_level == "word":
            output_qe_layer = TimeDistributed(
                    Dense(
                        params['WORD_QE_CLASSES'],
                        activation=out_activation
                        ),
                    name=self.ids_outputs[0]
                    )(annotations)

        elif self.task_level == "phrase":
            raise NotImplementedError

        elif self.task_level == "sent":
            output_qe_layer = Dense(1,
                    activation=out_activation,
                    name=self.ids_outputs[0]
                    )(annotations)

        elif self.task_level == "doc":
            output_qe_layer = Dense(1,
                    activation=out_activation,
                    name=self.ids_outputs[0]
                    )(dec_doc_last_state_concat)

        # instantiating a Model object
        self.model = Model(
                inputs=[src_words, trg_words],
                outputs=[output_qe_layer]
                )

