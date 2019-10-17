from quest.models.model import QEModel
from quest.models.utils import 


class BiRNN(QEModel):

    def __init__(self):
        super().__init__()

        pass

    def build():
        """
        Build the BiRNN QE model and return a "Model" object.
        #TODO: add reference
        """
        pass

    #=============================
    # Word-level QE -- BiRNN model
    #=============================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, line_words))
    # 2. Parallel machine-translated sentences (shape: (mini_batch_size, line_words))
    #
    ## Output:
    # 1. Word quality labels (shape: (mini_batch_size, number_of_qe_labels))
    #
    ## Summary of the model:
    # The sentence-level representations of both the SRC and the MT are created using two bi-directional RNNs.
    # Those representations are then concatenated at the word level, and used for making classification decisions.

    def EncWord(self, params):
        src_words = Input(name=self.ids_inputs[0],
                          batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')

        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='src_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_words)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_state')

        src_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
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
                                        name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)

        trg_words = Input(name=self.ids_inputs[1],
                          batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')

        trg_embedding = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='target_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(trg_words)
        trg_embedding = Regularize(trg_embedding, params, trainable=self.trainable, name='state')

        trg_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
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
                                    merge_mode='concat')(trg_embedding)

        annotations = concatenate([src_annotations, trg_annotations], name='anot_seq_concat')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')

        word_qe = TimeDistributed(Dense(params['WORD_QE_CLASSES'], activation=out_activation), name=self.ids_outputs[0])(annotations)

        self.model = Model(inputs=[src_words, trg_words],
                           outputs=[word_qe])


    #===============================================================================
    # Word-level QE -- simplified RNN POSTECH model inspired by Jhaveri et al., 2018
    #===============================================================================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, words))
    # 2. Machine-translated sentences (shape: (mini_batch_size, words))
    #
    ## Output:
    # 1. Word quality labels (shape: (mini_batch_size, number_of_qe_labels))
    #
    ## References:
    # - Nisarg Jhaveri, Manish Gupta, and Vasudeva Varman. 2018. Translation quality estimation for indian languages. In Proceedings of th 21st International Conference of the European Association for Machine Transla- tion (EAMT).
    
    def EncWordAtt(self, params):
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
        trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')

        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(trg_words)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

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
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')

        word_qe = TimeDistributed(Dense(params['WORD_QE_CLASSES'], activation=out_activation), trainable=self.trainable, name=self.ids_outputs[0])(proj_h)

        # self.model = Model(inputs=[src_text, next_words, next_words_bkw], outputs=[merged_states,softout, softoutQE])
        # if params['DOUBLE_STOCHASTIC_ATTENTION_REG'] > 0.:
        #     self.model.add_loss(alpha_regularizer)
        self.model = Model(inputs=[src_text, trg_words],
                           outputs=[word_qe])



    #=============================
    # Phrase-level QE -- BiRNN model
    #=============================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, line_words))
    # 2. Parallel machine-translated sentences (shape: (mini_batch_size, sentence_phrases, words))
    #
    ## Output:
    # 1. Phrase quality labels (shape: (mini_batch_size, number_of_qe_labels))
    #
    ## Summary of the model:
    # The encoder encodes the source, the decoder at each timestep produces an output representation taking into account the previously produced representations, as well as the sum of source word representations weighted by the attention mechanism.
    # The resulting word-level representations are summarized (sum or average) into phrase-level representations used to make classification decisions.


    def EncPhraseAtt(self, params):
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
        # trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, None]), dtype='int32')
        trg_words = Input(name=self.ids_inputs[1], batch_shape=tuple([None, params['SECOND_DIM_SIZE'], None]), dtype='int32')
        trg_reshape = Reshape((-1,))
        # reshape MT input to 2d
        trg_words_reshaped = trg_reshape(trg_words)

        # 3.1.2. Target word embedding
        state_below = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                name='target_word_embedding_below',
                                embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                embeddings_initializer=params['INIT_FUNCTION'],
                                trainable=self.trainable,
                                mask_zero=True)(trg_words_reshaped)
        state_below = Regularize(state_below, params, trainable=self.trainable, name='state_below')

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

        trg_annotations = proj_h
        annotations = NonMasking()(trg_annotations)
        # reshape back to 3d
        annotations_reshape = Reshape((params['SECOND_DIM_SIZE'], -1, params['DECODER_HIDDEN_SIZE']))
        annotations_reshaped = annotations_reshape(annotations)
        #summarize phrase representations (average by default)
        merge_mode = params.get('WORD_MERGE_MODE', None)
        merge_words = Lambda(mask_aware_mean4d, mask_aware_merge_output_shape4d)
        if merge_mode == 'sum':
            merge_words = Lambda(sum4d, mask_aware_merge_output_shape4d)
            
        output_annotations = merge_words(annotations_reshaped)
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        phrase_qe = TimeDistributed(Dense(params['PHRASE_QE_CLASSES'], activation=out_activation), trainable=self.trainable,
                                  name=self.ids_outputs[0])(output_annotations)

        self.model = Model(inputs=[src_text, trg_words],
                           outputs=[phrase_qe])



    #=================================
    # Sentence-level QE -- BiRNN model
    #=================================
    #
    ## Inputs:
    # 1. Sentences in src language (shape: (mini_batch_size, line_words))
    # 2. Parallel machine-translated documents (shape: (mini_batch_size, line_words))
    #
    ## Output:
    # 1. Sentence quality scores (shape: (mini_batch_size,))
    #
    ## Summary of the model:
    # The sententence-level representations of both the SRC and the MT are created using two bi-directional RNNs.
    # Those representations are then concatenated at the word level, and the sentence representation is a weighted sum of its words.
    # We apply the following attention function computing a normalized weight for each hidden state of an RNN h_j: 
    #       alpha_j = exp(W_a*h_j)/sum_k exp(W_a*h_k)
    # The resulting sentence vector is thus a weighted sum of word vectors:
    #       v = sum_j alpha_j*h_j
    # Sentence vectors are then directly used for making classification decisions.

    def EncSent(self, params):
        src_words = Input(name=self.ids_inputs[0],
                          batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='src_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_words)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_state')

        src_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
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
                                        name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)

        trg_words = Input(name=self.ids_inputs[1],
                          batch_shape=tuple([None, params['MAX_INPUT_TEXT_LEN']]), dtype='int32')

        trg_embedding = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='target_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(trg_words)
        trg_embedding = Regularize(trg_embedding, params, trainable=self.trainable, name='state')

        trg_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
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
                                    merge_mode='concat')(trg_embedding)

        annotations = concatenate([src_annotations, trg_annotations], name='anot_seq_concat')
        annotations = NonMasking()(annotations)
        # apply attention over words at the sentence-level
        annotations = attention_3d_block(annotations, params, 'sent')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        qe_sent = Dense(1, activation=out_activation, name=self.ids_outputs[0])(annotations)

        self.model = Model(inputs=[src_words, trg_words],
                           outputs=[qe_sent])




    #=================================
    # Document-level QE -- BiRNN model
    #=================================
    #
    ## Inputs:
    # 1. Documents in src language (shape: (mini_batch_size, doc_lines, words))
    # 2. Parallel machine-translated documents (shape: (mini_batch_size, doc_lines, words))
    #
    ## Output:
    # 1. Document quality scores (shape: (mini_batch_size,))
    #
    ## Summary of the model:
    # A hierarchical neural architecture that generalizes sentence-level representations to the document level.
    # The sentence-level representations of both the SRC and the MT are created using two bi-directional RNNs.
    # Those representations are then concatenated at the word level.
    # A sentence representation is a weighted sum of its words.
    # We apply the following attention function computing a normalized weight for each hidden state of an RNN h_j:
    #       alpha_j = exp(W_a*h_j)/sum_k exp(W_a*h_k)
    # The resulting sentence vector is thus a weighted sum of word vectors:
    #       v = sum_j alpha_j*h_j
    # Sentence vectors are inputted in a doc-level bi-directional RNN.
    # The last hidden state of ths RNN is taken as the summary of an entire document.
    # Doc-level representations are then directly used for making classification decisions.
    #
    ## References
    # - Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. 2016. Hierarchical attention networks for document classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1480-1489, San Diego, California, June. Association for Computational Linguistics.
    # - Jiwei Li, Thang Luong, and Dan Jurafsky. 2015. A hierarchical neural autoencoder for paragraphs and docu- ments. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1106-1115, Beijing, China, July. Association for Computational Linguistics.

    def EncDoc(self, params):
        src_words = Input(name=self.ids_inputs[0],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        genreshape = GeneralReshape((None, params['MAX_INPUT_TEXT_LEN']), params)
        src_words_in = genreshape(src_words)

        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='src_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_words_in)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_state')

        src_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
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
                                        name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)

        trg_words = Input(name=self.ids_inputs[1],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        trg_words_in = genreshape(trg_words)

        trg_embedding = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='target_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(trg_words_in)
        trg_embedding = Regularize(trg_embedding, params, trainable=self.trainable, name='state')

        trg_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
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
                                    merge_mode='concat')(trg_embedding)

        annotations = concatenate([src_annotations, trg_annotations], name='anot_seq_concat')
        annotations = NonMasking()(annotations)
        # apply sent-level attention over words
        annotations = attention_3d_block(annotations, params, 'sent')
        # reshape back to 3d input to group sent vectors per doc
        genreshape_out = GeneralReshape((None, params['SECOND_DIM_SIZE'], params['ENCODER_HIDDEN_SIZE'] * 4), params)
        annotations = genreshape_out(annotations)

        # bi-RNN over doc sentences
        dec_doc_frw, dec_doc_last_state_frw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True, return_state=True,
                                         name='dec_doc_frw')(annotations)
        dec_doc_bkw, dec_doc_last_state_bkw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True, return_state=True,
                                         go_backwards=True, name='dec_doc_bkw')(annotations)

        dec_doc_bkw = Reverse(dec_doc_bkw._keras_shape[2], axes=1,
                            name='dec_reverse_doc_bkw')(dec_doc_bkw)

        dec_doc_seq_concat = concatenate([dec_doc_frw, dec_doc_bkw], trainable=self.trainable_est, name='dec_doc_seq_concat')

        # we take the last bi-RNN state as doc summary
        dec_doc_last_state_concat = concatenate([dec_doc_last_state_frw, dec_doc_last_state_bkw], name='dec_doc_last_state_concat')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')
        qe_doc = Dense(1, activation=out_activation, name=self.ids_outputs[0])(dec_doc_last_state_concat)

        self.model = Model(inputs=[src_words, trg_words],
                           outputs=[qe_doc])



    #=============================================================================
    # Document-level QE with Attention mechanism -- BiRNN model Doc QE + Attention
    #=============================================================================
    #
    ## Inputs:
    # 1. Documents in src language (shape: (mini_batch_size, doc_lines, words))
    # 2. Parallel machine-translated documents (shape: (mini_batch_size, doc_lines, words))
    #
    ## Output:
    # 1. Document quality scores (shape: (mini_batch_size,))
    #
    ## Summary of the model:
    # A hierarchical neural architecture that generalizes sentence-level representations to the      document level.
    # The sentence-level representations of both the SRC and the MT are created using two bi-directional RNNs.
    # Those representations are then concatenated at the word level.
    # A sentence representation is a weighted sum of its words.
    # We apply the following attention function computing a normalized weight for each hidden state of an RNN h_j:
    #       alpha_j = exp(W_a*h_j)/sum_k exp(W_a*h_k)
    # The resulting sentence vector is thus a weighted sum of word vectors:
    #       v = sum_j alpha_j*h_j
    # Sentence vectors are inputted in a doc-level bi-directional RNN.
    # A document representation is a weighted sum of its sentences. We apply the attention function as described above.
    # Doc-level representations are then directly used for making classification decisions.
    #
    ## References
    # - Zichao Yang, Diyi Yang, Chris Dyer, Xiaodong He, Alex Smola, and Eduard Hovy. 2016. Hierarchical attention networks for document classification. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 1480-1489, San Diego, California, June. Association for Computational Linguistics.
    # - Jiwei Li, Thang Luong, and Dan Jurafsky. 2015. A hierarchical neural autoencoder for paragraphs and docu- ments. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1106-1115, Beijing, China, July. Association for Computational Linguistics.

    def EncDocAtt(self, params):
        src_words = Input(name=self.ids_inputs[0],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        genreshape = GeneralReshape((None, params['MAX_INPUT_TEXT_LEN']), params)
        src_words_in = genreshape(src_words)

        src_embedding = Embedding(params['INPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='src_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(src_words_in)
        src_embedding = Regularize(src_embedding, params, trainable=self.trainable, name='src_state')

        src_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
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
                                        name='src_bidirectional_encoder_' + params['ENCODER_RNN_TYPE'],
                                        merge_mode='concat')(src_embedding)

        trg_words = Input(name=self.ids_inputs[1],
                          batch_shape=tuple([None, params['SECOND_DIM_SIZE'], params['MAX_INPUT_TEXT_LEN']]), dtype='int32')
        # Reshape input to 2d to produce sent-level vectors
        trg_words_in = genreshape(trg_words)

        trg_embedding = Embedding(params['OUTPUT_VOCABULARY_SIZE'], params['TARGET_TEXT_EMBEDDING_SIZE'],
                                  name='target_word_embedding',
                                  embeddings_regularizer=l2(params['WEIGHT_DECAY']),
                                  embeddings_initializer=params['INIT_FUNCTION'],
                                  trainable=self.trainable,
                                  mask_zero=True)(trg_words_in)
        trg_embedding = Regularize(trg_embedding, params, trainable=self.trainable, name='state')

        trg_annotations = Bidirectional(eval(params['ENCODER_RNN_TYPE'])(params['ENCODER_HIDDEN_SIZE'],
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
                                    merge_mode='concat')(trg_embedding)

        annotations = concatenate([src_annotations, trg_annotations], name='anot_seq_concat')
        annotations = NonMasking()(annotations)
        # apply sent-level attention over words
        annotations = attention_3d_block(annotations, params, 'sent')
        # reshape back to 3d input to group sent vectors per doc
        genreshape_out = GeneralReshape((None, params['SECOND_DIM_SIZE'], params['ENCODER_HIDDEN_SIZE'] * 4), params)
        annotations = genreshape_out(annotations)

        #bi-RNN over doc sentences
        dec_doc_frw, dec_doc_last_state_frw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True, return_state=True,
                                         name='dec_doc_frw')(annotations)
        dec_doc_bkw, dec_doc_last_state_bkw = GRU(params['DOC_DECODER_HIDDEN_SIZE'], return_sequences=True, return_state=True,
                                         go_backwards=True, name='dec_doc_bkw')(annotations)

        dec_doc_bkw = Reverse(dec_doc_bkw._keras_shape[2], axes=1,
                            name='dec_reverse_doc_bkw')(dec_doc_bkw)

        dec_doc_seq_concat = concatenate([dec_doc_frw, dec_doc_bkw], trainable=self.trainable_est, name='dec_doc_seq_concat')

        dec_doc_last_state_concat = concatenate([dec_doc_last_state_frw, dec_doc_last_state_bkw], name='dec_doc_last_state_concat')
        dec_doc_seq_concat = NonMasking()(dec_doc_seq_concat)

        # apply attention over doc sentences
        attention_mul = attention_3d_block(dec_doc_seq_concat, params, 'doc')
        out_activation=params.get('OUT_ACTIVATION', 'sigmoid')

        qe_doc = Dense(1, activation=out_activation, name=self.ids_outputs[0])(attention_mul)

        self.model = Model(inputs=[src_words, trg_words],
                           outputs=[qe_doc])

