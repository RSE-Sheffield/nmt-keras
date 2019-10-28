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
#================================
# POSTECH-inspired Predictor model
#================================
#
## Inputs:
# 1. Sentences in src language (shape: (mini_batch_size, words))
# 2. One-position left-shifted reference sentences to represent the right context (shape: (mini_batch_size, words))
# 3. One-position rigth-shifted reference sentences to represent the left context (shape: (mini_batch_size, words))
#
## Output:
# 1. Machine-translated sentences (shape: (mini_batch_size, output_vocabulary_size))
#
## References
# - Hyun Kim, Hun-Young Jung, Hongseok Kwon, Jong-Hyeok Lee, and Seung-Hoon Na. 2017a. Predictor- estimator: Neural quality estimation based on target word prediction for machine translation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 17(1):3:1-3:22, September.
"""

from .model import QEModel
from .utils import *


class Predictor(QEModel):

    def __init__(self, params, model_type='Predictor',
            verbose=1, structure_path=None, weights_path=None,
            model_name=None, vocabularies=None, store_path=None,
            set_optimizer=True, clear_dirs=True):

        # define here attributes that are model specific

        # and init from the QEModel class
        super().__init__(params)


    def build(self):
        """
        Build the Predictor model and return a "Model" object.
        #TODO: add reference
        """

        params = self.params


        #######################################################################
        ####      INPUTS OF THE MODEL                                      ####
        #######################################################################
        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0],
                batch_shape=tuple([None, None]),
                dtype='int32')

        #######################################################################
        ####      ENCODERS                                                 ####
        #######################################################################
        # 2. Encoder
        # 2.1. Source word embedding
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'],
                params['SOURCE_TEXT_EMBEDDING_SIZE'],
                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                embeddings_initializer=params['INIT_FUNCTION'],
                trainable=self.trainable,
                mask_zero=True,
                name='source_word_embedding')(src_text)

        src_embedding = Regularize(src_embedding, params,
                trainable=self.trainable,
                name='src_embedding')

        # 2.2. BRNN encoder (GRU/LSTM)
        #FIXME: is this block a duplication with the loop below?
        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(
                    eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                        kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                        recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                        bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                        dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                        recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                        kernel_initializer=params['INIT_FUNCTION'],
                        recurrent_initializer=params['INNER_INIT'],
                        return_sequences=True, trainable=self.trainable),
                    merge_mode='concat',
                    name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE']
                    )(src_embedding)
        else:
            annotations = eval(
                    params['ENCODER_RNN_TYPE'])(
                            params['ENCODER_HIDDEN_SIZE'],
                            kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                            recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                            bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                            dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                            recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                            kernel_initializer=params['INIT_FUNCTION'],
                            recurrent_initializer=params['INNER_INIT'],
                            return_sequences=True,
                            trainable=self.trainable,
                            name='encoder_' + params['ENCODER_RNN_TYPE']
                            )(src_embedding)

        annotations = Regularize(annotations, params,
                trainable=self.trainable, name='annotations')
        #FIXME: end of block

        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(
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
                        name='bidirectional_encoder_' + str(n_layer)
                        )(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(
                        params['ENCODER_HIDDEN_SIZE'],
                        kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                        recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                        bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                        dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                        recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                        kernel_initializer=params['INIT_FUNCTION'],
                        recurrent_initializer=params['INNER_INIT'],
                        return_sequences=True,
                        trainable=self.trainable,
                        name='encoder_' + str(n_layer)
                        )(annotations)

                current_annotations = Regularize(current_annotations, params,
                        trainable=self.trainable, name='annotations_' + str(n_layer))

                annotations = Add(trainable=self.trainable)([annotations, current_annotations])


        #######################################################################
        ####      DECODERS                                                 ####
        #######################################################################
        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        next_words = Input(name=self.ids_inputs[1],
                batch_shape=tuple([None, None]),
                dtype='int32')

        next_words_bkw = Input(name=self.ids_inputs[2],
                batch_shape=tuple([None, None]),
                dtype='int32')

        # 3.1.2. Target word embedding
        state_below = Embedding(
                params['OUTPUT_VOCABULARY_SIZE'],
                params['TARGET_TEXT_EMBEDDING_SIZE'],
                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                embeddings_initializer=params['INIT_FUNCTION'],
                trainable=self.trainable,
                mask_zero=True,
                name='target_word_embedding_below'
                )(next_words)

        state_below = Regularize(state_below, params,
                trainable=self.trainable,
                name='state_below')

        state_above = Embedding(
                params['OUTPUT_VOCABULARY_SIZE'],
                params['TARGET_TEXT_EMBEDDING_SIZE'],
                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                embeddings_initializer=params['INIT_FUNCTION'],
                trainable=self.trainable,
                mask_zero=True,
                name='target_word_embedding_above'
                )(next_words_bkw)

        state_above = Regularize(state_above, params,
                trainable=self.trainable,
                name='state_above')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)

        # We may want the padded annotations
        annotations = MaskLayer(trainable=self.trainable)(annotations)

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(
                        params['DECODER_HIDDEN_SIZE'],
                        kernel_initializer=params['INIT_FUNCTION'],
                        kernel_regularizer=l2(params['WEIGHT_DECAY']),
                        bias_regularizer=l2(params['WEIGHT_DECAY']),
                        activation=params['INIT_LAYERS'][n_layer_init],
                        trainable=self.trainable,
                        name='init_layer_%d' % n_layer_init
                        )(ctx_mean)

                ctx_mean = Regularize(ctx_mean, params,
                        trainable=self.trainable,
                        name='ctx' + str(n_layer_init))

            initial_state = Dense(
                    params['DECODER_HIDDEN_SIZE'],
                    kernel_initializer=params['INIT_FUNCTION'],
                    kernel_regularizer=l2(params['WEIGHT_DECAY']),
                    bias_regularizer=l2(params['WEIGHT_DECAY']),
                    activation=params['INIT_LAYERS'][-1],
                    trainable=self.trainable,
                    name='initial_state'
                    )(ctx_mean)

            initial_state = Regularize(initial_state, params,
                    trainable=self.trainable,
                    name='initial_state')

            input_attentional_decoder = [state_below, annotations, initial_state]

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                initial_memory = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_memory',
                                       kernel_initializer=params['INIT_FUNCTION'],
                                       kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                       bias_regularizer=l2(params['WEIGHT_DECAY']),
                                       activation=params['INIT_LAYERS'][-1],
                                       trainable=self.trainable)(ctx_mean)
                initial_memory = Regularize(initial_memory, params, trainable=self.trainable, name='initial_memory')
                input_attentional_decoder.append(initial_memory)
        else:
            # Initialize to zeros vector
            input_attentional_decoder = [state_below, annotations]
            initial_state = ZeroesLayer(
                    params['DECODER_HIDDEN_SIZE'],
                    trainable=self.trainable)(ctx_mean)

            input_attentional_decoder.append(initial_state)

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(
                params['DECODER_HIDDEN_SIZE'],
                att_units=params.get('ATTENTION_SIZE', 0),
                kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                attention_context_wa_regularizer=l2(params['WEIGHT_DECAY']),
                attention_recurrent_regularizer=l2(params['WEIGHT_DECAY']),
                attention_context_regularizer=l2(params['WEIGHT_DECAY']),
                bias_ba_regularizer=l2(params['WEIGHT_DECAY']),
                dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                conditional_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                attention_dropout=params['DROPOUT_P'],
                kernel_initializer=params['INIT_FUNCTION'],
                recurrent_initializer=params['INNER_INIT'],
                attention_context_initializer=params['INIT_ATT'],
                return_sequences=True,
                return_extra_variables=True,
                return_states=True,
                num_inputs=len(input_attentional_decoder),
                trainable=self.trainable,
                name='decoder_Att' + params['DECODER_RNN_TYPE'] + 'Cond'
                )

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h  = rnn_output[0]
        x_att   = rnn_output[1]
        alphas  = rnn_output[2]
        h_state = rnn_output[3]

        trg_enc_frw = eval(params['ENCODER_RNN_TYPE'])(
                params['ENCODER_HIDDEN_SIZE'],
                kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                kernel_initializer=params['INIT_FUNCTION'],
                recurrent_initializer=params['INNER_INIT'],
                return_sequences=True,
                trainable=self.trainable,
                name='enc_trg_frw'
                )
        trg_state_frw = trg_enc_frw(state_below)

        trg_enc_bkw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                kernel_initializer=params['INIT_FUNCTION'],
                recurrent_initializer=params['INNER_INIT'],
                return_sequences=True,
                trainable=self.trainable,
                go_backwards=True,
                name='enc_trg_bkw'
                )
        trg_state_bkw = trg_enc_bkw(state_above)

        trg_state_bkw = Reverse(
                trg_state_bkw._keras_shape[2],
                axes=1,
                trainable=self.trainable,
                name='reverse_trg_state_bkw'
                )(trg_state_bkw)

        # preparing formula 3b
        merged_emb = concatenate(
                [state_below, state_above],
                axis=2,
                trainable=self.trainable,
                name='merged_emb'
                )

        merged_states = concatenate(
                [trg_state_frw, trg_state_bkw],
                axis=2,
                trainable=self.trainable,
                name='merged_states'
                )

        # we replace state before with the concatenation of state before and after
        proj_h = merged_states

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]

        shared_Lambda_Permute = PermuteGeneral((1, 0, 2), trainable=self.trainable)

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(
                    alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG']
                    )(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params,
                trainable=self.trainable,
                shared_layers=True,
                name='proj_h0'
                )

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [merged_states, shared_Lambda_Permute(x_att), initial_state]

            shared_proj_h_list.append(
                    eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
                        params['DECODER_HIDDEN_SIZE'],
                        kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                        recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                        conditional_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                        bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                        dropout=params['RECURRENT_DROPOUT_P'],
                        recurrent_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                        conditional_dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                        kernel_initializer=params['INIT_FUNCTION'],
                        recurrent_initializer=params['INNER_INIT'],
                        return_sequences=True,
                        return_states=True,
                        num_inputs=len(current_rnn_input),
                        trainable=self.trainable,
                        name='decoder_' + params['DECODER_RNN_TYPE'].replace(
                            'Conditional', '') + 'Cond' + str(n_layer)
                        )
                )

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)

            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])

            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params,
                    trainable=self.trainable,
                    shared_layers=True,
                    name='proj_h' + str(n_layer)
                    )

            shared_reg_proj_h_list.append(shared_reg_proj_h)
            proj_h = Add(trainable=self.trainable)([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(
                Dense(
                    params['SKIP_VECTORS_HIDDEN_SIZE'],
                    kernel_initializer=params['INIT_FUNCTION'],
                    kernel_regularizer=l2(params['WEIGHT_DECAY']),
                    bias_regularizer=l2(params['WEIGHT_DECAY']),
                    activation='linear',
                    trainable=self.trainable),
                name='logit_lstm'
                )
        out_layer_mlp = shared_FC_mlp(proj_h)

        shared_FC_ctx = TimeDistributed(
                Dense(
                    params['SKIP_VECTORS_HIDDEN_SIZE'],
                    kernel_initializer=params['INIT_FUNCTION'],
                    kernel_regularizer=l2(params['WEIGHT_DECAY']),
                    bias_regularizer=l2(params['WEIGHT_DECAY']),
                    activation='linear',
                    trainable=self.trainable),
                name='logit_ctx'
                )
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)

        shared_FC_emb = TimeDistributed(
                Dense(
                    params['SKIP_VECTORS_HIDDEN_SIZE'],
                    kernel_initializer=params['INIT_FUNCTION'],
                    kernel_regularizer=l2(params['WEIGHT_DECAY']),
                    bias_regularizer=l2(params['WEIGHT_DECAY']),
                    activation='linear',
                    trainable=self.trainable),
                name='logit_emb'
                )
        out_layer_emb = shared_FC_emb(merged_emb)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(
                out_layer_mlp,
                params,
                shared_layers=True,
                trainable=self.trainable,
                name='out_layer_mlp'
                )

        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(
                out_layer_ctx,
                params,
                shared_layers=True,
                trainable=self.trainable,
                name='out_layer_ctx'
                )

        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(
                out_layer_emb,
                params,
                shared_layers=True,
                trainable=self.trainable,
                name='out_layer_emb'
                )


        # formula 3b addition
        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(
                trainable=self.trainable,
                name='additional_input'
                )
        additional_output = shared_additional_output_merge(
                [out_layer_mlp, out_layer_ctx, out_layer_emb]
                )
        shared_activation_tanh = Activation('tanh', trainable=self.trainable)
        out_layer = shared_activation_tanh(additional_output)


        # 3.6 Optional deep ouput layer
        shared_deep_list = []
        shared_reg_deep_list = []

        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(
                Dense(
                    dimension,
                    activation=activation,
                    kernel_initializer=params['INIT_FUNCTION'],
                    kernel_regularizer=l2(params['WEIGHT_DECAY']),
                    bias_regularizer=l2(params['WEIGHT_DECAY']),
                    trainable=self.trainable),
                name=activation + '_%d' % i)
                )

            out_layer = shared_deep_list[-1](out_layer)

            [out_layer, shared_reg_out_layer] = Regularize(
                    out_layer,
                    params,
                    trainable=self.trainable,
                    shared_layers=True,
                    name='out_layer_' + str(activation) + '_%d' % i
                    )

            shared_reg_deep_list.append(shared_reg_out_layer)

        shared_QE_soft = TimeDistributed(
                Dense(
                    params['QE_VECTOR_SIZE'],
                    use_bias=False,
                    kernel_regularizer=l2(params['WEIGHT_DECAY']),
                    bias_regularizer=l2(params['WEIGHT_DECAY']),
                    name='QE'+params['CLASSIFIER_ACTIVATION'],
                    trainable=self.trainable
                    ),
                name='QE'+self.ids_outputs[0]
                )

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(
                Dense(
                    params['OUTPUT_VOCABULARY_SIZE'],
                    use_bias=False,
                    activation=params['CLASSIFIER_ACTIVATION'],
                    kernel_regularizer=l2(params['WEIGHT_DECAY']),
                    bias_regularizer=l2(params['WEIGHT_DECAY']),
                    name=params['CLASSIFIER_ACTIVATION'],
                    trainable = self.trainable
                    ),
                name=self.ids_outputs[0]
                )

        softoutQE = shared_QE_soft(out_layer)
        softout = shared_FC_soft(softoutQE)

        self.model = Model(
                inputs=[src_text, next_words, next_words_bkw],
                outputs=[softout]
                )

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0.:
            self.model.add_loss(alpha_regularizer)
