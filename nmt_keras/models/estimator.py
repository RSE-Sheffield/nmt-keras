
    #==================================================
    # Word-level QE -- POSTECH-inspired Estimator model
    #==================================================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, words))
    # 2. One-position left-shifted machine-translated sentences to represent the right context (shape: (mini_batch_size, words))
    # 3. One-position rigth-shifted machine-translated sentences to represent the left context (shape: (mini_batch_size, words))
    # 4. Unshifted machine-translated sentences for evaluation (shape: (mini_batch_size, words))
    #
    ## Output:
    # 1. Word quality labels (shape: (mini_batch_size, number_of_qe_labels))
    #
    ## References:
    # - Hyun Kim, Hun-Young Jung, Hongseok Kwon, Jong-Hyeok Lee, and Seung-Hoon Na. 2017a. Predictor- estimator: Neural quality estimation based on target word prediction for machine translation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 17(1):3:1-3:22, September.

    def EstimatorWord(self, params):
        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')

        # 2. Encoder
        # 2.1. Source word embedding
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_embedding')

        # 2.2. BRNN encoder (GRU/LSTM)
        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')

        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable,
                                             name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        #trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')

        next_words_bkw = Input(name=self.ids_inputs[2], batch_shape=tuple([None, None]), dtype='int32')
        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        state_above = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_above',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_bkw)
        state_above = Regularize(state_above, params, trainable=self.trainable, name='state_above')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
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
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'], trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_enc_frw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       name='enc_trg_frw')

        trg_enc_bkw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       go_backwards=True,
                                                       name='enc_trg_bkw')

        trg_state_frw = trg_enc_frw(state_below)
        trg_state_bkw = trg_enc_bkw(state_above)

        trg_state_bkw = Reverse(trg_state_bkw._keras_shape[2], axes=1, trainable=self.trainable,
                                name='reverse_trg_state_bkw')(trg_state_bkw)

        # preparing formula 3b
        merged_emb = concatenate([state_below, state_above], axis=2, trainable=self.trainable, name='merged_emb')
        merged_states = concatenate([trg_state_frw, trg_state_bkw], axis=2, trainable=self.trainable,
                                    name='merged_states')

        # we replace state before with the concatenation of state before and after
        proj_h = merged_states

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2), trainable=self.trainable)

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, trainable=self.trainable, shared_layers=True,
                                                 name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [merged_states, shared_Lambda_Permute(x_att), initial_state]
            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
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
                    'Conditional', '') + 'Cond' + str(n_layer)))

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)
            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, trainable=self.trainable,
                                                             shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add(trainable=self.trainable)([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(merged_emb)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input',
                                                                                      trainable=self.trainable)
        # formula 3b addition
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation_tanh = Activation('tanh', trainable=self.trainable)

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []
        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=self.trainable),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, trainable=self.trainable, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        shared_QE_soft = TimeDistributed(Dense(params['QE_VECTOR_SIZE'],
                                               use_bias=False,
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name='QE' + params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='QE' + 'target_text')

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               use_bias=False,
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='target-text')

        softoutQE = shared_QE_soft(out_layer)
        softout = shared_FC_soft(softoutQE)

        trg_words = Input(name=self.ids_inputs[3], batch_shape=tuple([None, None]), dtype='int32')
        next_words_one_hot = one_hot(trg_words, params)

        # we multiply the weight matrix by one-hot reference vector to make vectors for the words we don't need zeroes
        qv_prep = DenseTranspose(params['QE_VECTOR_SIZE'], shared_FC_soft, self.ids_outputs[0], name='transpose',
                                 trainable=self.trainable_est)

        qv_prep_res = qv_prep(next_words_one_hot)

        # we multiply our matrix with zero values by our quality vectors so that zero vectors do not influence our decisions
        qv = multiply([qv_prep_res, softoutQE], name='qv')

        enc_qe_frw, last_state_frw = GRU(params['QE_VECTOR_SIZE'], trainable=self.trainable_est, return_sequences=True, return_state=True,
                                         name='qe_frw')(qv)
        enc_qe_bkw, last_state_bkw = GRU(params['QE_VECTOR_SIZE'], trainable=self.trainable_est, return_sequences=True, return_state=True,
                                         go_backwards=True, name='qe_bkw')(qv)

        enc_qe_bkw = Reverse(enc_qe_bkw._keras_shape[2], axes=1, trainable=self.trainable_est,
                             name='reverse_enc_qe_bkw')(enc_qe_bkw)

        last_state_concat = concatenate([last_state_frw, last_state_bkw], trainable=self.trainable_est, name='last_state_concat')

        seq_concat = concatenate([enc_qe_frw, enc_qe_bkw], trainable=self.trainable_est, name='seq_concat')

        # uncomment for Post QE
        # fin_seq = concatenate([seq_concat, merged_states])
        #qe_sent = Dense(1, activation='sigmoid', name=self.ids_outputs[0])(last_state_concat)
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        word_qe = TimeDistributed(Dense(params['WORD_QE_CLASSES'], activation=out_activation), trainable=self.trainable_est, name=self.ids_outputs[0])(seq_concat)

        # self.model = Model(inputs=[src_text, next_words, next_words_bkw], outputs=[merged_states,softout, softoutQE])
        # if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0.:
        #     self.model.add_loss(alpha_regularizer)
        self.model = Model(inputs=[src_text, next_words, next_words_bkw, trg_words],
                           outputs=[word_qe])



    #======================================================
    # Phrase-level QE -- POSTECH-inspired Estimator model
    #======================================================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, line_words))
    # 2. Parallel machine-translated sentences (shape: (mini_batch_size, sentence_phrases, words))
    # 3. One-position rigth-shifted machine-translated sentences to represent the left context (shape: (mini_batch_size, sentence_phrases, words))
    # 4. Unshifted machine-translated sentences for evaluation (shape: (mini_batch_size, sentence_phrases, words))
    #
    ## Output:
    # 1. Phrase quality labels (shape: (mini_batch_size, number_of_qe_labels))
    #
    ## References:
    # - Hyun Kim, Hun-Young Jung, Hongseok Kwon, Jong-Hyeok Lee, and Seung-Hoon Na. 2017a. Predictor- estimator: Neural quality estimation based on target word prediction for machine translation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 17(1):3:1-3:22, September.

    def EstimatorPhrase(self, params):

        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')

        # 2. Encoder
        # 2.1. Source word embedding
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_embedding')

        # 2.2. BRNN encoder (GRU/LSTM)
        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')

        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable,
                                             name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        #trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, params['SECOND_DIM_SIZE'], None]), dtype='int32')

        # Reshape phrase inputs to 2d
        trg_reshape = Reshape((-1, ))
        next_words_reshaped = trg_reshape(next_words)

        next_words_bkw = Input(name=self.ids_inputs[2], batch_shape=tuple([None, params['SECOND_DIM_SIZE'], None]), dtype='int32')
        next_words_bkw_reshaped = trg_reshape(next_words_bkw)
        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_reshaped)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        state_above = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_above',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_bkw_reshaped)
        state_above = Regularize(state_above, params, trainable=self.trainable, name='state_above')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
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
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'], trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_enc_frw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       name='enc_trg_frw')

        trg_enc_bkw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       go_backwards=True,
                                                       name='enc_trg_bkw')

        trg_state_frw = trg_enc_frw(state_below)
        trg_state_bkw = trg_enc_bkw(state_above)

        trg_state_bkw = Reverse(trg_state_bkw._keras_shape[2], axes=1, trainable=self.trainable,
                                name='reverse_trg_state_bkw')(trg_state_bkw)

        # preparing formula 3b
        merged_emb = concatenate([state_below, state_above], axis=2, trainable=self.trainable, name='merged_emb')
        merged_states = concatenate([trg_state_frw, trg_state_bkw], axis=2, trainable=self.trainable,
                                    name='merged_states')

        # we replace state before with the concatenation of state before and after
        proj_h = merged_states

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2), trainable=self.trainable)

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, trainable=self.trainable, shared_layers=True,
                                                 name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [merged_states, shared_Lambda_Permute(x_att), initial_state]
            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
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
                    'Conditional', '') + 'Cond' + str(n_layer)))

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)
            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, trainable=self.trainable,
                                                             shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add(trainable=self.trainable)([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(merged_emb)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input',
                                                                                      trainable=self.trainable)
        # formula 3b addition
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation_tanh = Activation('tanh', trainable=self.trainable)

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []
        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=self.trainable),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, trainable=self.trainable, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        shared_QE_soft = TimeDistributed(Dense(params['QE_VECTOR_SIZE'],
                                               use_bias=False,
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name='QE' + params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='QE' + 'target_text')

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               use_bias=False,
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='target-text')

        softoutQE = shared_QE_soft(out_layer)
        softout = shared_FC_soft(softoutQE)

        trg_words = Input(name=self.ids_inputs[3], batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_TRG_INPUT_TEXT_LEN']]), dtype='int32')

        #reshape to 2d
        trg_words_reshaped = trg_reshape(trg_words)
        next_words_one_hot = one_hot(trg_words_reshaped, params)

        # we multiply the weight matrix by one-hot reference vector to make vectors for the words we don't need zeroes
        qv_prep = DenseTranspose(params['QE_VECTOR_SIZE'], shared_FC_soft, self.ids_outputs[0], name='transpose',
                                 trainable=self.trainable_est)

        qv_prep_res = qv_prep(next_words_one_hot)

        # we multiply our matrix with zero values by our quality vectors so that zero vectors do not influence our decisions
        qv = multiply([qv_prep_res, softoutQE], name='qv')

        enc_qe_frw, last_state_frw = GRU(params['QE_VECTOR_SIZE'], trainable=self.trainable_est, return_sequences=True, return_state=True,
                                         name='qe_frw')(qv)
        enc_qe_bkw, last_state_bkw = GRU(params['QE_VECTOR_SIZE'], trainable=self.trainable_est, return_sequences=True, return_state=True,
                                         go_backwards=True, name='qe_bkw')(qv)

        enc_qe_bkw = Reverse(enc_qe_bkw._keras_shape[2], axes=1, trainable=self.trainable_est,
                             name='reverse_enc_qe_bkw')(enc_qe_bkw)

        last_state_concat = concatenate([last_state_frw, last_state_bkw], trainable=self.trainable_est, name='last_state_concat')

        seq_concat = concatenate([enc_qe_frw, enc_qe_bkw], trainable=self.trainable_est, name='seq_concat')

        trg_annotations = seq_concat
        annotations = NonMasking()(trg_annotations)

        # reshape back to 3d
        annotations_reshape = Reshape((params['SECOND_DIM_SIZE'], -1, params['QE_VECTOR_SIZE']*2))
        annotations_reshaped = annotations_reshape(annotations)

        #summarize phrase representations (average by default)
        merge_mode = params.get('WORD_MERGE_MODE', None)
        merge_words = Lambda(mask_aware_mean4d, mask_aware_merge_output_shape4d)
        if merge_mode == 'sum':
            merge_words = Lambda(sum4d, mask_aware_merge_output_shape4d)

        output_annotations = merge_words(annotations_reshaped)
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        phrase_qe = TimeDistributed(Dense(params['PHRASE_QE_CLASSES'], activation=out_activation), trainable=self.trainable_est, name=self.ids_outputs[0])(output_annotations)

        self.model = Model(inputs=[src_text, next_words, next_words_bkw, trg_words],
                           outputs=[phrase_qe])



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


    def EstimatorSent(self, params):
        # 1. Source text input
        src_text = Input(name=self.ids_inputs[0], batch_shape=tuple([None, None]), dtype='int32')

        # 2. Encoder
        # 2.1. Source word embedding
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable,name='src_embedding')

        # 2.2. BRNN encoder (GRU/LSTM)
        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True, trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')

        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable, name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        #trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        next_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        next_words_bkw = Input(name=self.ids_inputs[2], batch_shape=tuple([None, None]), dtype='int32')

        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        state_above = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_above',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_bkw)
        state_above = Regularize(state_above, params, trainable=self.trainable, name='state_above')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
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
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'],trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_enc_frw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                     kernel_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     recurrent_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     bias_regularizer=l2(
                                                                         params['RECURRENT_WEIGHT_DECAY']),
                                                                     dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                     recurrent_dropout=params[
                                                                         'RECURRENT_DROPOUT_P'],
                                                                     kernel_initializer=params['INIT_FUNCTION'],
                                                                     recurrent_initializer=params['INNER_INIT'],
                                                                     return_sequences=True,
                                                                     trainable=self.trainable,
                                                                     name='enc_trg_frw')

        trg_enc_bkw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                         kernel_regularizer=l2(
                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                         recurrent_regularizer=l2(
                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                         bias_regularizer=l2(
                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                         recurrent_dropout=params[
                                                             'RECURRENT_DROPOUT_P'],
                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                         recurrent_initializer=params['INNER_INIT'],
                                                         return_sequences=True,
                                                         trainable=self.trainable,
                                                         go_backwards=True,
                                                         name='enc_trg_bkw')

        trg_state_frw = trg_enc_frw(state_below)
        trg_state_bkw = trg_enc_bkw(state_above)

        trg_state_bkw = Reverse(trg_state_bkw._keras_shape[2], axes=1, trainable=self.trainable, name='reverse_trg_state_bkw')(trg_state_bkw)

        # preparing formula 3b
        merged_emb = concatenate([state_below, state_above], axis=2, trainable=self.trainable, name='merged_emb')
        merged_states = concatenate([trg_state_frw, trg_state_bkw], axis=2, trainable=self.trainable, name='merged_states')

        # we replace state before with the concatenation of state before and after
        proj_h = merged_states

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2),trainable=self.trainable)

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, trainable=self.trainable, shared_layers=True, name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [merged_states, shared_Lambda_Permute(x_att), initial_state]
            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
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
                    'Conditional', '') + 'Cond' + str(n_layer)))

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)
            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, trainable=self.trainable, shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add(trainable=self.trainable)([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(merged_emb)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, trainable=self.trainable, name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, trainable=self.trainable, name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, trainable=self.trainable, name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input',
                                                                                      trainable=self.trainable)

        # formula 3b addition
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation_tanh = Activation('tanh', trainable=self.trainable)

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []

        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=self.trainable),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, trainable=self.trainable, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        shared_QE_soft = TimeDistributed(Dense(params['QE_VECTOR_SIZE'],
                                               use_bias=False,
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name='QE'+params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='QE'+'target_text')

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               use_bias=False,
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION'],
                                               trainable = self.trainable),
                                         name='target-text')

        softoutQE = shared_QE_soft(out_layer)
        softout = shared_FC_soft(softoutQE)

        trg_words = Input(name=self.ids_inputs[3], batch_shape=tuple([None, None]), dtype='int32')

        next_words_one_hot = one_hot(trg_words, params)

        # we multiply the weight matrix by one-hot reference vector to make vectors for the words we don't need zeroes
        qv_prep = DenseTranspose(params['QE_VECTOR_SIZE'], shared_FC_soft, self.ids_outputs[0], name='transpose',
                                 trainable=self.trainable_est)

        qv_prep_res = qv_prep(next_words_one_hot)

        # we multiply our matrix with zero values by our quality vectors so that zero vectors do not influence our decisions
        qv = multiply([qv_prep_res, softoutQE], name='qv')

        enc_qe_frw, last_state_frw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         name='qe_frw', trainable=self.trainable_est)(qv)
        enc_qe_bkw, last_state_bkw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         go_backwards=True, name='qe_bkw', trainable=self.trainable_est)(qv)

        enc_qe_bkw = Reverse(enc_qe_bkw._keras_shape[2], axes=1, trainable=self.trainable_est,
                             name='reverse_enc_qe_bkw')(enc_qe_bkw)

        last_state_concat = concatenate([last_state_frw, last_state_bkw], trainable=self.trainable_est, name='last_state_concat')

        seq_concat = concatenate([enc_qe_frw, enc_qe_bkw], trainable=self.trainable_est, name='seq_concat')

        # uncomment for Post QE
        # fin_seq = concatenate([seq_concat, merged_states])
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        qe_sent = Dense(1, activation=out_activation, trainable=self.trainable_est, name=self.ids_outputs[0])(last_state_concat)
        #word_qe = TimeDistributed(Dense(params['WORD_QE_CLASSES'], activation='sigmoid'), name=self.ids_outputs[2])(
        #    seq_concat)

        # self.model = Model(inputs=[src_text, next_words, next_words_bkw], outputs=[merged_states,softout, softoutQE])
        # if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0.:
        #     self.model.add_loss(alpha_regularizer)
        self.model = Model(inputs=[src_text, next_words, next_words_bkw, trg_words],
                           outputs=[qe_sent])


    #===========================================
    # Document-level QE --POSTECH-inspired model
    #===========================================
    #
    ## Inputs:
    # 1. Documents in src language (shape: (mini_batch_size, doc_lines, words))
    # 2. Machine-translated documents with one-position left-shifted sentences to represent the right context (shape: (mini_batch_size, doc_lines, words))
    # 3. Machine-translated documents with one-position rigth-shifted sentences to represent the left context (shape: (mini_batch_size, doc_lines, words))
    # 4. Machine-translated documents with unshifted sentences for evaluation (shape: (mini_batch_size, doc_lines,words))
    #
    ## Output:
    # 1. Document quality scores (shape: (mini_batch_size,))
    #
    ## Summary of the model:
    # A hierarchical neural architecture that generalizes sentence-level representations to the      document level.
    # Sentence-level representations are created as by a POSTECH-inspired sentence-level QE model.
    # Those representations are inputted in a doc-level bi-directional RNN.
    # The last hidden state of ths RNN is taken as the summary of an entire document.
    # Doc-level representations are then directly used for making classification decisions.
    #
    ## References
    # - Hyun Kim, Hun-Young Jung, Hongseok Kwon, Jong-Hyeok Lee, and Seung-Hoon Na. 2017a. Predictor- estimator: Neural quality estimation based on target word prediction for machine translation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 17(1):3:1-3:22, September.
    # - Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. 2016. Hierarchical attention networks for document classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1480-1489, San Diego, California, June. Association for Computational Linguistics.
    # - Jiwei Li, Thang Luong, and Dan Jurafsky. 2015. A hierarchical neural autoencoder for paragraphs and docu- ments. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1106-1115, Beijing, China, July. Association for Computational Linguistics.

    def EstimatorDoc(self, params):
        src_text = Input(name=self.ids_inputs[0],
                         batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')

        # Reshape input to 2d to produce sent-level vectors. Reshaping to (None, None) is necessary for compatibility with pre-trained Predictors.
        genreshape = GeneralReshape((None, None), params)
        src_text_in = genreshape(src_text)
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text_in)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_embedding')

        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')

        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable,
                                             name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        next_words = Input(name=self.ids_inputs[1],
                           batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        next_words_in = genreshape(next_words)

        next_words_bkw = Input(name=self.ids_inputs[2],
                               batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]),
                               dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        next_words_bkw_in = genreshape(next_words_bkw)
        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_in)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        state_above = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_above',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_bkw_in)
        state_above = Regularize(state_above, params, trainable=self.trainable, name='state_above')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
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
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'], trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_enc_frw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       name='enc_trg_frw')

        trg_enc_bkw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       go_backwards=True,
                                                       name='enc_trg_bkw')

        trg_state_frw = trg_enc_frw(state_below)
        trg_state_bkw = trg_enc_bkw(state_above)

        trg_state_bkw = Reverse(trg_state_bkw._keras_shape[2], axes=1, trainable=self.trainable,
                                name='reverse_trg_state_bkw')(trg_state_bkw)

        # preparing formula 3b
        merged_emb = concatenate([state_below, state_above], axis=2, trainable=self.trainable, name='merged_emb')
        merged_states = concatenate([trg_state_frw, trg_state_bkw], axis=2, trainable=self.trainable,
                                    name='merged_states')

        # we replace state before with the concatenation of state before and after
        proj_h = merged_states

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2), trainable=self.trainable)

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, trainable=self.trainable, shared_layers=True,
                                                 name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [merged_states, shared_Lambda_Permute(x_att), initial_state]
            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
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
                    'Conditional', '') + 'Cond' + str(n_layer)))

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)
            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, trainable=self.trainable,
                                                             shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add(trainable=self.trainable)([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(merged_emb)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input',
                                                                                      trainable=self.trainable)

        # formula 3b addition
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation_tanh = Activation('tanh', trainable=self.trainable)

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []

        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=self.trainable),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, trainable=self.trainable, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        shared_QE_soft = TimeDistributed(Dense(params['QE_VECTOR_SIZE'],
                                               use_bias=False,
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name='QE' + params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='QE' + 'target_text')

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               use_bias=False,
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='target-text')

        softoutQE = shared_QE_soft(out_layer)
        softout = shared_FC_soft(softoutQE)

        trg_words = Input(name=self.ids_inputs[3],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        trg_words_in = genreshape(trg_words)

        next_words_one_hot = one_hot(trg_words_in, params)

        # we multiply the weight matrix by one-hot reference vector to make vectors for the words we don't need zeroes
        qv_prep = DenseTranspose(params['QE_VECTOR_SIZE'], shared_FC_soft, self.ids_outputs[0], name='transpose',
                                 trainable=self.trainable_est)

        qv_prep_res = qv_prep(next_words_one_hot)

        # we multiply our matrix with zero values by our quality vectors so that zero vectors do not influence our decisions
        qv = multiply([qv_prep_res, softoutQE], name='qv')

        enc_qe_frw, last_state_frw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         name='qe_frw', trainable=self.trainable_est)(qv)
        enc_qe_bkw, last_state_bkw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         go_backwards=True, name='qe_bkw', trainable=self.trainable_est)(qv)

        enc_qe_bkw = Reverse(enc_qe_bkw._keras_shape[2], axes=1, trainable=self.trainable_est,
                             name='reverse_enc_qe_bkw')(enc_qe_bkw)

        enc_qe_concat = concatenate([enc_qe_frw, enc_qe_bkw], name='enc_qe_concat')
        last_state_concat = concatenate([last_state_frw, last_state_bkw], trainable=self.trainable_est,
                                        name='last_state_concat')

        #reshape back to 3d input to group sent vectors per doc
        genreshape_out = GeneralReshape((None, params['SECOND_DIM_SIZE'], params['QE_VECTOR_SIZE'] * 2), params)

        last_state_concat = genreshape_out(last_state_concat)

        #bi-RNN over doc sentences
        dec_doc_frw, dec_doc_last_state_frw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True,
                                                  return_state=True,
                                                  name='dec_doc_frw')(last_state_concat)
        dec_doc_bkw, dec_doc_last_state_bkw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True,
                                                  return_state=True,
                                                  go_backwards=True, name='dec_doc_bkw')(last_state_concat)

        dec_doc_bkw = Reverse(dec_doc_bkw._keras_shape[2], axes=1,
                              name='dec_reverse_doc_bkw')(dec_doc_bkw)

        dec_doc_seq_concat = concatenate([dec_doc_frw, dec_doc_bkw], trainable=self.trainable_est,
                                         name='dec_doc_seq_concat')

        #we take the last bi-RNN state as doc summary
        dec_doc_last_state_concat = concatenate([dec_doc_last_state_frw, dec_doc_last_state_bkw],
                                                name='dec_doc_last_state_concat')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        qe_doc = Dense(1, activation=out_activation, name=self.ids_outputs[0])(dec_doc_last_state_concat)
        self.model = Model(inputs=[src_text, next_words, next_words_bkw, trg_words],
                           outputs=[qe_doc])


    #=====================================================================
    # Document-level QE with Attention mechanism -- POSTECH-inspired model
    #=====================================================================
    #
    ## Inputs:
    # 1. Documents in src language (shape: (mini_batch_size, doc_lines, words))
    # 2. Machine-translated documents with one-position left-shifted sentences to represent the right context (shape: (mini_batch_size, doc_lines, words))
    # 3. Machine-translated documents with one-position rigth-shifted sentences to represent the left context (shape: (mini_batch_size, doc_lines, words))
    # 4. Machine-translated documents with unshifted sentences for evaluation (shape: (mini_batch_size, doc_lines, doc_lines,words))
    #
    ## Output:
    # 1. Document quality scores (shape: (mini_batch_size,))
    #
    ## Summary of the model:
    # A hierarchical neural architecture that generalizes sentence-level representations to the      document level.
    # Sentence-level representations are created as by a POSTECH-inspired sentence-level QE model.
    # Those representations are inputted in a doc-level bi-directional RNN.
    # A document representation is a weighted sum of its sentences.
    # We apply the following attention function computing a normalized weight for each hidden state of an RNN h_j: alpha_j = exp(W_a*h_j)/sum_k exp(W_a*h_k)
    # The resulting document vector is thus a weighted sum of sentence vectors:
    # v = sum_j alpha_j*h_j
    # Doc-level representations are then directly used for making classification decisions.
    #
    ## References
    # - Hyun Kim, Hun-Young Jung, Hongseok Kwon, Jong-Hyeok Lee, and Seung-Hoon Na. 2017a. Predictor- estimator: Neural quality estimation based on target word prediction for machine translation. ACM Trans. Asian Low-Resour. Lang. Inf. Process., 17(1):3:1-3:22, September.
    # - Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. 2016. Hierarchical attention networks for document classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1480-1489, San Diego, California, June. Association for Computational Linguistics.
    # - Jiwei Li, Thang Luong, and Dan Jurafsky. 2015. A hierarchical neural autoencoder for paragraphs and docu- ments. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1106-1115, Beijing, China, July. Association for Computational Linguistics.

    def EstimatorDocAtt(self, params):
        src_text = Input(name=self.ids_inputs[0],
                         batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')

        # Reshape input to 2d to produce sent-level vectors. Reshaping to (None, None) is necessary for compatibility with pre-trained Predictors
        genreshape = GeneralReshape((None, None), params)
        src_text_in = genreshape(src_text)

        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['SOURCE_TEXT_EMBEDDING_SIZE'],
                                  name='source_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_text_in)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_embedding')

        if params['BIDIRECTIONAL_ENCODER']:
            annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                         kernel_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         recurrent_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         bias_regularizer=l2(
                                                                             params['RECURRENT_WEIGHT_DECAY']),
                                                                         dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                         recurrent_dropout=params[
                                                                             'RECURRENT_DROPOUT_P'],
                                                                         kernel_initializer=params['INIT_FUNCTION'],
                                                                         recurrent_initializer=params['INNER_INIT'],
                                                                         return_sequences=True,
                                                                         trainable=self.trainable),
                                        name='bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)
        else:
            annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                           kernel_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           recurrent_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           bias_regularizer=l2(params['RECURRENT_WEIGHT_DECAY']),
                                                           dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                           recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                           kernel_initializer=params['INIT_FUNCTION'],
                                                           recurrent_initializer=params['INNER_INIT'],
                                                           return_sequences=True,
                                                           name='encoder_' + params['ENCODER_RNN_TYPE'],
                                                           trainable=self.trainable)(src_embedding)
        annotations = Regularize(annotations, params, trainable=self.trainable, name='annotations')
        # 2.3. Potentially deep encoder
        for n_layer in range(1, params['N_LAYERS_ENCODER']):
            if params['BIDIRECTIONAL_DEEP_ENCODER']:
                current_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                                     kernel_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     recurrent_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     bias_regularizer=l2(
                                                                                         params[
                                                                                             'RECURRENT_WEIGHT_DECAY']),
                                                                                     dropout=params[
                                                                                         'RECURRENT_INPUT_DROPOUT_P'],
                                                                                     recurrent_dropout=params[
                                                                                         'RECURRENT_DROPOUT_P'],
                                                                                     kernel_initializer=params[
                                                                                         'INIT_FUNCTION'],
                                                                                     recurrent_initializer=params[
                                                                                         'INNER_INIT'],
                                                                                     return_sequences=True,
                                                                                     trainable=self.trainable,
                                                                                     ),
                                                    merge_mode='concat',
                                                    name='bidirectional_encoder_' + str(n_layer))(annotations)
            else:
                current_annotations = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                                       kernel_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       recurrent_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       bias_regularizer=l2(
                                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                                       recurrent_dropout=params['RECURRENT_DROPOUT_P'],
                                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                                       recurrent_initializer=params['INNER_INIT'],
                                                                       return_sequences=True,
                                                                       trainable=self.trainable,
                                                                       name='encoder_' + str(n_layer))(annotations)
            current_annotations = Regularize(current_annotations, params, trainable=self.trainable,
                                             name='annotations_' + str(n_layer))
            annotations = Add(trainable=self.trainable)([annotations, current_annotations])

        # 3. Decoder
        # 3.1.1. Previously generated words as inputs for training -> Teacher forcing
        # trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        next_words = Input(name=self.ids_inputs[1],
                           batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        next_words_in = genreshape(next_words)

        next_words_bkw = Input(name=self.ids_inputs[2],
                               batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]),
                               dtype='int32')
        next_words_bkw_in = genreshape(next_words_bkw)

        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_in)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

        state_above = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_above',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(next_words_bkw_in)
        state_above = Regularize(state_above, params, trainable=self.trainable, name='state_above')

        # 3.2. Decoder's RNN initialization perceptrons with ctx mean
        ctx_mean = MaskedMean(trainable=self.trainable)(annotations)
        annotations = MaskLayer(trainable=self.trainable)(annotations)  # We may want the padded annotations

        if len(params['INIT_LAYERS']) > 0:
            for n_layer_init in range(len(params['INIT_LAYERS']) - 1):
                ctx_mean = Dense(params['DECODER_HIDDEN_SIZE'], name='init_layer_%d' % n_layer_init,
                                 kernel_initializer=params['INIT_FUNCTION'],
                                 kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                 bias_regularizer=l2(params['WEIGHT_DECAY']),
                                 activation=params['INIT_LAYERS'][n_layer_init],
                                 trainable=self.trainable
                                 )(ctx_mean)
                ctx_mean = Regularize(ctx_mean, params, trainable=self.trainable, name='ctx' + str(n_layer_init))

            initial_state = Dense(params['DECODER_HIDDEN_SIZE'], name='initial_state',
                                  kernel_initializer=params['INIT_FUNCTION'],
                                  kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                  bias_regularizer=l2(params['WEIGHT_DECAY']),
                                  activation=params['INIT_LAYERS'][-1],
                                  trainable=self.trainable
                                  )(ctx_mean)
            initial_state = Regularize(initial_state, params, trainable=self.trainable, name='initial_state')
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
            initial_state = ZeroesLayer(params['DECODER_HIDDEN_SIZE'], trainable=self.trainable)(ctx_mean)
            input_attentional_decoder.append(initial_state)
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                input_attentional_decoder.append(initial_state)

        # 3.3. Attentional decoder
        sharedAttRNNCond = eval('Att' + params['DECODER_RNN_TYPE'] + 'Cond')(params['DECODER_HIDDEN_SIZE'],
                                                                             att_units=params.get('ATTENTION_SIZE', 0),
                                                                             kernel_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             recurrent_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             conditional_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             bias_regularizer=l2(
                                                                                 params['RECURRENT_WEIGHT_DECAY']),
                                                                             attention_context_wa_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_recurrent_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             attention_context_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             bias_ba_regularizer=l2(
                                                                                 params['WEIGHT_DECAY']),
                                                                             dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             recurrent_dropout=params[
                                                                                 'RECURRENT_DROPOUT_P'],
                                                                             conditional_dropout=params[
                                                                                 'RECURRENT_INPUT_DROPOUT_P'],
                                                                             attention_dropout=params['DROPOUT_P'],
                                                                             kernel_initializer=params['INIT_FUNCTION'],
                                                                             recurrent_initializer=params['INNER_INIT'],
                                                                             attention_context_initializer=params[
                                                                                 'INIT_ATT'],
                                                                             return_sequences=True,
                                                                             return_extra_variables=True,
                                                                             return_states=True,
                                                                             num_inputs=len(input_attentional_decoder),
                                                                             name='decoder_Att' + params[
                                                                                 'DECODER_RNN_TYPE'] + 'Cond',
                                                                             trainable=self.trainable)

        rnn_output = sharedAttRNNCond(input_attentional_decoder)
        proj_h = rnn_output[0]
        x_att = rnn_output[1]
        alphas = rnn_output[2]
        h_state = rnn_output[3]

        trg_enc_frw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       name='enc_trg_frw')

        trg_enc_bkw = eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
                                                       kernel_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       recurrent_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       bias_regularizer=l2(
                                                           params['RECURRENT_WEIGHT_DECAY']),
                                                       dropout=params['RECURRENT_INPUT_DROPOUT_P'],
                                                       recurrent_dropout=params[
                                                           'RECURRENT_DROPOUT_P'],
                                                       kernel_initializer=params['INIT_FUNCTION'],
                                                       recurrent_initializer=params['INNER_INIT'],
                                                       return_sequences=True,
                                                       trainable=self.trainable,
                                                       go_backwards=True,
                                                       name='enc_trg_bkw')

        trg_state_frw = trg_enc_frw(state_below)
        trg_state_bkw = trg_enc_bkw(state_above)

        trg_state_bkw = Reverse(trg_state_bkw._keras_shape[2], axes=1, trainable=self.trainable,
                                name='reverse_trg_state_bkw')(trg_state_bkw)

        # preparing formula 3b
        merged_emb = concatenate([state_below, state_above], axis=2, trainable=self.trainable, name='merged_emb')
        merged_states = concatenate([trg_state_frw, trg_state_bkw], axis=2, trainable=self.trainable,
                                    name='merged_states')

        # we replace state before with the concatenation of state before and after
        proj_h = merged_states

        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memory = rnn_output[4]
        shared_Lambda_Permute = PermuteGeneral((1, 0, 2), trainable=self.trainable)

        if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0:
            alpha_regularizer = AlphaRegularizer(alpha_factor=params['DOUBLE_STOCHASTIC_ATTENTION_REG'])(alphas)

        [proj_h, shared_reg_proj_h] = Regularize(proj_h, params, trainable=self.trainable, shared_layers=True,
                                                 name='proj_h0')

        # 3.4. Possibly deep decoder
        shared_proj_h_list = []
        shared_reg_proj_h_list = []

        h_states_list = [h_state]
        if 'LSTM' in params['DECODER_RNN_TYPE']:
            h_memories_list = [h_memory]

        for n_layer in range(1, params['N_LAYERS_DECODER']):
            current_rnn_input = [merged_states, shared_Lambda_Permute(x_att), initial_state]
            shared_proj_h_list.append(eval(params['DECODER_RNN_TYPE'].replace('Conditional', '') + 'Cond')(
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
                    'Conditional', '') + 'Cond' + str(n_layer)))

            if 'LSTM' in params['DECODER_RNN_TYPE']:
                current_rnn_input.append(initial_memory)
            current_rnn_output = shared_proj_h_list[-1](current_rnn_input)
            current_proj_h = current_rnn_output[0]
            h_states_list.append(current_rnn_output[1])
            if 'LSTM' in params['DECODER_RNN_TYPE']:
                h_memories_list.append(current_rnn_output[2])
            [current_proj_h, shared_reg_proj_h] = Regularize(current_proj_h, params, trainable=self.trainable,
                                                             shared_layers=True,
                                                             name='proj_h' + str(n_layer))
            shared_reg_proj_h_list.append(shared_reg_proj_h)

            proj_h = Add(trainable=self.trainable)([proj_h, current_proj_h])

        # 3.5. Skip connections between encoder and output layer
        shared_FC_mlp = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_lstm')
        out_layer_mlp = shared_FC_mlp(proj_h)
        shared_FC_ctx = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_ctx')
        out_layer_ctx = shared_FC_ctx(x_att)
        out_layer_ctx = shared_Lambda_Permute(out_layer_ctx)
        shared_FC_emb = TimeDistributed(Dense(params['SKIP_VECTORS_HIDDEN_SIZE'],
                                              kernel_initializer=params['INIT_FUNCTION'],
                                              kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                              bias_regularizer=l2(params['WEIGHT_DECAY']),
                                              activation='linear',
                                              trainable=self.trainable),
                                        name='logit_emb')
        out_layer_emb = shared_FC_emb(merged_emb)

        [out_layer_mlp, shared_reg_out_layer_mlp] = Regularize(out_layer_mlp, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_mlp')
        [out_layer_ctx, shared_reg_out_layer_ctx] = Regularize(out_layer_ctx, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_ctx')
        [out_layer_emb, shared_reg_out_layer_emb] = Regularize(out_layer_emb, params,
                                                               shared_layers=True, trainable=self.trainable,
                                                               name='out_layer_emb')

        shared_additional_output_merge = eval(params['ADDITIONAL_OUTPUT_MERGE_MODE'])(name='additional_input',
                                                                                      trainable=self.trainable)

        # formula 3b addition
        additional_output = shared_additional_output_merge([out_layer_mlp, out_layer_ctx, out_layer_emb])
        shared_activation_tanh = Activation('tanh', trainable=self.trainable)

        out_layer = shared_activation_tanh(additional_output)

        shared_deep_list = []
        shared_reg_deep_list = []

        # 3.6 Optional deep ouput layer
        for i, (activation, dimension) in enumerate(params['DEEP_OUTPUT_LAYERS']):
            shared_deep_list.append(TimeDistributed(Dense(dimension, activation=activation,
                                                          kernel_initializer=params['INIT_FUNCTION'],
                                                          kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                                          bias_regularizer=l2(params['WEIGHT_DECAY']),
                                                          trainable=self.trainable),
                                                    name=activation + '_%d' % i))
            out_layer = shared_deep_list[-1](out_layer)
            [out_layer, shared_reg_out_layer] = Regularize(out_layer,
                                                           params, trainable=self.trainable, shared_layers=True,
                                                           name='out_layer_' + str(activation) + '_%d' % i)
            shared_reg_deep_list.append(shared_reg_out_layer)

        shared_QE_soft = TimeDistributed(Dense(params['QE_VECTOR_SIZE'],
                                               use_bias=False,
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name='QE' + params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='QE' + 'target_text')

        # 3.7. Output layer: Softmax
        shared_FC_soft = TimeDistributed(Dense(params['OUTPUT_VOCABULARY_SIZE'],
                                               use_bias=False,
                                               activation=params['CLASSIFIER_ACTIVATION'],
                                               kernel_regularizer=l2(params['WEIGHT_DECAY']),
                                               bias_regularizer=l2(params['WEIGHT_DECAY']),
                                               name=params['CLASSIFIER_ACTIVATION'],
                                               trainable=self.trainable),
                                         name='target-text')

        softoutQE = shared_QE_soft(out_layer)
        softout = shared_FC_soft(softoutQE)

        trg_words = Input(name=self.ids_inputs[3],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        trg_words_in = genreshape(trg_words)

        next_words_one_hot = one_hot(trg_words_in, params)

        # we multiply the weight matrix by one-hot reference vector to make vectors for the words we don't need zeroes
        qv_prep = DenseTranspose(params['QE_VECTOR_SIZE'], shared_FC_soft, self.ids_outputs[0], name='transpose',
                                 trainable=self.trainable_est)

        qv_prep_res = qv_prep(next_words_one_hot)

        # we multiply our matrix with zero values by our quality vectors so that zero vectors do not influence our decisions
        qv = multiply([qv_prep_res, softoutQE], name='qv')

        enc_qe_frw, last_state_frw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         name='qe_frw', trainable=self.trainable_est)(qv)
        enc_qe_bkw, last_state_bkw = GRU(params['QE_VECTOR_SIZE'], return_sequences=True, return_state=True,
                                         go_backwards=True, name='qe_bkw', trainable=self.trainable_est)(qv)

        enc_qe_bkw = Reverse(enc_qe_bkw._keras_shape[2], axes=1, trainable=self.trainable_est,
                             name='reverse_enc_qe_bkw')(enc_qe_bkw)

        enc_qe_concat = concatenate([enc_qe_frw, enc_qe_bkw], name='enc_qe_concat')
        last_state_concat = concatenate([last_state_frw, last_state_bkw], trainable=self.trainable_est,
                                        name='last_state_concat')

        # reshape back to 3d input to group sent vectors per doc
        genreshape_out = GeneralReshape((None, params['SECOND_DIM_SIZE'], params['QE_VECTOR_SIZE'] * 2), params)

        last_state_concat = genreshape_out(last_state_concat)

        #bi-RNN over doc sentences
        dec_doc_frw, dec_doc_last_state_frw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True,
                                                  return_state=True,
                                                  name='dec_doc_frw')(last_state_concat)
        dec_doc_bkw, dec_doc_last_state_bkw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True,
                                                  return_state=True,
                                                  go_backwards=True, name='dec_doc_bkw')(last_state_concat)

        dec_doc_bkw = Reverse(dec_doc_bkw._keras_shape[2], axes=1,
                              name='dec_reverse_doc_bkw')(dec_doc_bkw)

        dec_doc_seq_concat = concatenate([dec_doc_frw, dec_doc_bkw], trainable=self.trainable_est,
                                         name='dec_doc_seq_concat')

        dec_doc_last_state_concat = concatenate([dec_doc_last_state_frw, dec_doc_last_state_bkw],
                                                name='dec_doc_last_state_concat')

        dec_doc_seq_concat  = NonMasking()(dec_doc_seq_concat)

        # apply doc-level attention over sentences
      attention_mul = attention_3d_block(dec_doc_seq_concat, params, 'doc')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')

        qe_doc = Dense(1, activation=out_activation, name=self.ids_outputs[0])(attention_mul)

        self.model = Model(inputs=[src_text, next_words, next_words_bkw, trg_words],
                           outputs=[qe_doc])
