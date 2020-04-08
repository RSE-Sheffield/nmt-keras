
def train(config, changes={}):
    """
    Handles QE model training.
    :param config: Either a path to a YAML or pkl config file or a dictionary of parameters.
    :param dataset: Optional path to a previously built pkl dataset.
    :param changes: Optional dictionary of parameters to overwrite config.
    """
    parameters = setparameters(user_config_path=config)

    if changes:
        parameters.update(changes2dict(changes))

    if parameters.get('SEED') is not None:
        print('Setting deepQuest seed to', parameters['SEED'])
        import numpy.random
        numpy.random.seed(parameters['SEED'])
        import random
        random.seed(parameters['SEED'])
    
    if parameters.get('GPU_ID') is not None:
        n_gpus = set_gpu_id(str(parameters.get('GPU_ID')))
        parameters.update({'N_GPUS': n_gpus, 'GPU_ID': parameters.get('GPU_ID')})

    import deepquest.train
    train.main(parameters)

def predict(model, dataset, save_path=None, evalset=None, changes={}):
    """
    Predicts QE scores on a dataset using a pre-trained model.
    :param model: Model file (.h5) to use.
    :param dataset: Dataset file (.pkl) to use.
    :param save_path: Optinal directory path to save predictions to. Default = STORE_PATH
    :param evalset: Optional set to evaluate on. Default = 'test'
    :param changes: Optional dictionary of parameters to overwrite config.
    """
    import deepquest.predict
    predict.main(model, dataset, save_path, evalset, changes2dict(changes))

def score(files):
    """
    Evaluate a set of predictions with regard to a reference set.
    :param files: List of paths to two text files containing predictions and references.
    """
    import deepquest.score
    score.main(files)

def set_gpu_id(gpuid):
    """
    Sets the environment variable CUDA_VISIBLE_DEVICES to control which GPUs are used for training.
    Must only be called from deepquest/__main__.py, ie before import of deepquest module
    :param gpuid: String of comma-separated integers refering to GPU devices.
    """
    import os
    import re
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpu_list = re.split(r"\W+", gpuid)
    gpustr = ",".join(gpu_list)
    gpucount = len(gpu_list)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpustr

    return gpucount


def changes2dict(changes_list):
    import ast
    if changes_list:
        changes_dict = {}
        try:
            for arg in changes_list:
                try:
                    k, v = arg.split('=')
                    if '_' in v:
                        changes_dict[k] = v
                    else:
                        if k == 'GPU_ID':
                            changes_dict[k] = str(v)
                        else:
                            changes_dict[k] = ast.literal_eval(v)
                except ValueError:
                    print('Ignoring command line arg: "%s"' % str(arg))
        except ValueError:
            print("Error processing arguments: {!r}".format(arg))
            exit(2)
        return changes_dict
    else:
        return {}


def setparameters(user_config_path):
    """
    This function sets the overall parameters to used by the current running instance of dq.
    """
    import codecs
    import logging
    logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
    logger = logging.getLogger(__name__)
    try:

        parameters = default_params() # load the default parameters (BiRNN by default)

        if user_config_path.endswith('.yml'):
            import yaml
            with codecs.open(user_config_path, 'r', encoding='utf-8') as fh_user:
                user_parameters = yaml.load(fh_user, Loader=yaml.FullLoader)
            parameters.update(user_parameters)
            del user_parameters

        elif user_config_path.endswith('.pkl'):
            from keras_wrapper.extra.read_write import pkl2dict
            parameters = update_parameters(parameters, pkl2dict(user_config_path))
        
        parameters = add_dependent_params(parameters) # add some parameters that depend on others

    except Exception as exception:
        logger.exception("Exception occured: {}".format(exception))

    return parameters

def default_params(model='BiRNN'):
    """
    Loads the default hyperparameters
    Parameters:

    model (str): type of model in default param options (currently only BiRNN)

    :return parameters: Dictionary of loaded parameters
    """
    if model.lower() == 'birnn':
        # Input data params
        TASK_NAME = 'BiRNN'                           # Task name
        # DATASET_NAME = TASK_NAME                        # Dataset name
        SRC_LAN = 'src'                                  # Language of the source text
        TRG_LAN = 'mt'                                  # Language of the target text

        DATA_DIR = 'data/'                                  # Data directory
        MODEL_DIRECTORY = 'trained_models/'                # Trained models will be stored here

        # SRC_LAN or TRG_LAN will be added to the file names
        TEXT_FILES = {'train': 'train.',        # Data files
                    'val': 'dev.',
                    'test': 'test.'}
        
        STOP_METRIC = 'pearson'                        # Metric for the stop

        # Model parameters
        MODEL_TYPE = 'EncSent'                 # Model to train. See model_zoo() for the supported architectures

        # Tensorboard params
        TENSORBOARD = False

        # Dataset class parameters
        INPUTS_IDS_DATASET = ['source_text', 'target_text']     # Corresponding inputs of the dataset
        #OUTPUTS_IDS_DATASET_FULL = ['target_text', 'word_qe', 'sent_hter']                   # Corresponding outputs of the dataset
        OUTPUTS_IDS_DATASET = ['sent_qe']
        INPUTS_IDS_MODEL = ['source_text', 'target_text']       # Corresponding inputs of the built model
        #OUTPUTS_IDS_MODEL_FULL = ['target_text','word_qe', 'sent_hter']                     # Corresponding outputs of the built model
        OUTPUTS_IDS_MODEL = ['sent_qe']
        WORD_QE_CLASSES = 5
        PRED_SCORE='hter'

        # Evaluation params
        METRICS = ['qe_metrics']                            # Metric used for evaluating the model
        #KERAS_METRICS = ['pearson_corr', 'mae', 'rmse']
        EVAL_ON_SETS = 'val'                        # Possible values: 'train', 'val' and 'test' (external evaluator)
        NO_REF = False
        #EVAL_ON_SETS_KERAS = ['val']                       #  Possible values: 'train', 'val' and 'test' (Keras' evaluator). Untested.
        EVAL_ON_SETS_KERAS = []
        START_EVAL_ON_EPOCH = 1                      # First epoch to start the model evaluation
        EVAL_EACH_EPOCHS = True                       # Select whether evaluate between N epochs or N updates
        EVAL_EACH = 1                                 # Sets the evaluation frequency (epochs or updates)

        MULTI_TASK = False

        # Search parameters
        SAMPLING = 'max_likelihood'                   # Possible values: multinomial or max_likelihood (recommended)
        TEMPERATURE = 1                               # Multinomial sampling parameter
        BEAM_SEARCH = False                            # Switches on-off the beam search procedure
        BEAM_SIZE = 6                                 # Beam size (in case of BEAM_SEARCH == True)
        OPTIMIZED_SEARCH = True                       # Compute annotations only a single time per sample
        SEARCH_PRUNING = False                        # Apply pruning strategies to the beam search method.
        MAXLEN_GIVEN_X = True                         # Generate translations of similar length to the source sentences
        MAXLEN_GIVEN_X_FACTOR = 2                     # The hypotheses will have (as maximum) the number of words of the
                                                    # source sentence * LENGTH_Y_GIVEN_X_FACTOR
        MINLEN_GIVEN_X = True                         # Generate translations of similar length to the source sentences
        MINLEN_GIVEN_X_FACTOR = 3                     # The hypotheses will have (as minimum) the number of words of the
                                                    # source sentence / LENGTH_Y_GIVEN_X_FACTOR

        # Apply length and coverage decoding normalizations.
        # See Section 7 from Wu et al. (2016) (https://arxiv.org/abs/1609.08144)
        LENGTH_PENALTY = False                        # Apply length penalty
        LENGTH_NORM_FACTOR = 0.2                      # Length penalty factor
        COVERAGE_PENALTY = False                      # Apply source coverage penalty
        COVERAGE_NORM_FACTOR = 0.2                    # Coverage penalty factor

        # Alternative (simple) length normalization.
        NORMALIZE_SAMPLING = False                    # Normalize hypotheses scores according to their length:
        ALPHA_FACTOR = .6                             # Normalization according to |h|**ALPHA_FACTOR

        # Unknown words treatment
        POS_UNK = True                                # Enable POS_UNK strategy for unknown words
        HEURISTIC = 0                                 # Heuristic to follow:
                                                    #     0: Replace the UNK by the correspondingly aligned source
                                                    #     1: Replace the UNK by the translation (given by an external
                                                    #        dictionary) of the correspondingly aligned source
                                                    #     2: Replace the UNK by the translation (given by an external
                                                    #        dictionary) of the correspondingly aligned source only if it
                                                    #        starts with a lowercase. Otherwise, copies the source word.
        ALIGN_FROM_RAW = True                         # Align using the full vocabulary or the short_list
        
        # Word representation params
        TOKENIZATION_METHOD = 'tokenize_none'         # Select which tokenization we'll apply.
                                                    # See Dataset class (from stager_keras_wrapper) for more info.

        DETOKENIZATION_METHOD = 'detokenize_none'     # Select which de-tokenization method we'll apply

        APPLY_DETOKENIZATION = False                  # Wheter we apply a detokenization method

        TOKENIZE_HYPOTHESES = True  		          # Whether we tokenize the hypotheses using the
                                                    # previously defined tokenization method
        TOKENIZE_REFERENCES = True                    # Whether we tokenize the references using the
                                                    # previously defined tokenization method

        # Input image parameters
        DATA_AUGMENTATION = False                     # Apply data augmentation on input data (still unimplemented for text inputs)

        # Text parameters
        FILL = 'end'                                  # Whether we pad the 'end' or the 'start' of the sentence with 0s
        PAD_ON_BATCH = False                           # Whether we take as many timesteps as the longest sequence of
                                                    # the batch or a fixed size (MAX_OUTPUT_TEXT_LEN)
        # Input text parameters
        INPUT_VOCABULARY_SIZE = 30000                     # Size of the input vocabulary. Set to 0 for using all,
                                                    # otherwise it will be truncated to these most frequent words.
        MIN_OCCURRENCES_INPUT_VOCAB = 0               # Minimum number of occurrences allowed for the words in the input vocabulary.
                                                    # Set to 0 for using them all.
        MAX_INPUT_TEXT_LEN = 70                       # Maximum length of the input sequence

        # Output text parameters
        OUTPUT_VOCABULARY_SIZE = 30000                    # Size of the input vocabulary. Set to 0 for using all,
                                                    # otherwise it will be truncated to these most frequent words.
        MIN_OCCURRENCES_OUTPUT_VOCAB = 0              # Minimum number of occurrences allowed for the words in the output vocabulary.
        MAX_OUTPUT_TEXT_LEN = 70                      # Maximum length of the output sequence
                                                    # set to 0 if we want to use the whole answer as a single class
        MAX_OUTPUT_TEXT_LEN_TEST = MAX_OUTPUT_TEXT_LEN * 3  # Maximum length of the output sequence during test time

        # Optimizer parameters (see model.compile() function)
        LOSS = ['mse']
    
        #predictor activation
        CLASSIFIER_ACTIVATION = 'softmax'

        OPTIMIZER = 'Adadelta'                            # Optimizer
        LR = 1.0                                    # Learning rate. Recommended values - Adam 0.001 - Adadelta 1.0
        CLIP_C = 1.                                   # During training, clip L2 norm of gradients to this value (0. means deactivated)
        CLIP_V = 0.                                   # During training, clip absolute value of gradients to this value (0. means deactivated)
        SAMPLE_WEIGHTS = {'word_qe': {'BAD': 3}}                        # Select whether we use a weights matrix (mask) for the data outputs
        # Learning rate annealing
        LR_DECAY = None                               # Frequency (number of epochs or updates) between LR annealings. Set to None for not decay the learning rate
        LR_GAMMA = 0.8                                # Multiplier used for decreasing the LR
        LR_REDUCE_EACH_EPOCHS = False                 # Reduce each LR_DECAY number of epochs or updates
        LR_START_REDUCTION_ON_EPOCH = 0               # Epoch to start the reduction
        LR_REDUCER_TYPE = 'exponential'               # Function to reduce. 'linear' and 'exponential' implemented.
        LR_REDUCER_EXP_BASE = 0.5                     # Base for the exponential decay
        LR_HALF_LIFE = 5000                           # Factor for exponenital decay

        # Training parameters
        MAX_EPOCH = 500 # Stop when computed this number of epochs
        EPOCH_PER_UPDATE = 1
        EPOCH_PER_PRED = 5
        EPOCH_PER_EST_SENT = 10
        EPOCH_PER_EST_WORD = 10
        
        #to use on real data
        BATCH_SIZE = 50

        HOMOGENEOUS_BATCHES = False                   # Use batches with homogeneous output lengths (Dangerous!!)
        JOINT_BATCHES = 4                             # When using homogeneous batches, get this number of batches to sort
        PARALLEL_LOADERS = 1                          # Parallel data batch loaders
        EPOCHS_FOR_SAVE = 1                           # Number of epochs between model saves
        WRITE_VALID_SAMPLES = True                    # Write valid samples in file
        SAVE_EACH_EVALUATION = True                   # Save each time we evaluate the model

        # Early stop parameters
        EARLY_STOP = True                             # Turns on/off the early stop protocol
        PATIENCE = 5                        # We'll stop if the val STOP_METRIC does not improve after this
                                                    # number of evaluations

        

        # only Predictor
        #MODEL_TYPE = 'Predictor'

        #parameters for Predictor
        ENCODER_RNN_TYPE = 'GRU'                     # Encoder's RNN unit type ('LSTM' and 'GRU' supported)
        DECODER_RNN_TYPE = 'ConditionalGRU'          # Decoder's RNN unit type
                                                    # ('LSTM', 'GRU', 'ConditionalLSTM' and 'ConditionalGRU' supported)
        # Initializers (see keras/initializations.py).
        INIT_FUNCTION = 'glorot_uniform'              # General initialization function for matrices.
        INNER_INIT = 'orthogonal'                     # Initialization function for inner RNN matrices.
        INIT_ATT = 'glorot_uniform'                   # Initialization function for attention mechism matrices

        SOURCE_TEXT_EMBEDDING_SIZE = 300              # Source language word embedding size.
        SRC_PRETRAINED_VECTORS = None                 # Path to pretrained vectors (e.g.: DATA_ROOT_PATH + '/DATA/word2vec.%s.npy' % SRC_LAN)
                                                    # Set to None if you don't want to use pretrained vectors.
                                                    # When using pretrained word embeddings. this parameter must match with the word embeddings size
        SRC_PRETRAINED_VECTORS_TRAINABLE = True       # Finetune or not the target word embedding vectors.

        TARGET_TEXT_EMBEDDING_SIZE = 300               # Source language word embedding size.
        TRG_PRETRAINED_VECTORS = None                 # Path to pretrained vectors. (e.g. DATA_ROOT_PATH + '/DATA/word2vec.%s.npy' % TRG_LAN)
                                                    # Set to None if you don't want to use pretrained vectors.
                                                    # When using pretrained word embeddings, the size of the pretrained word embeddings must match with the word embeddings size.
        TRG_PRETRAINED_VECTORS_TRAINABLE = True       # Finetune or not the target word embedding vectors.

        # Encoder configuration
        ENCODER_HIDDEN_SIZE = 50                      # For models with RNN encoder
        BIDIRECTIONAL_ENCODER = True                  # Use bidirectional encoder
        N_LAYERS_ENCODER = 1                          # Stack this number of encoding layers
        BIDIRECTIONAL_DEEP_ENCODER = True             # Use bidirectional encoder in all encoding layers

        # Decoder configuration
        DECODER_HIDDEN_SIZE = 500                      # For models with RNN decoder
        N_LAYERS_DECODER = 1                          # Stack this number of decoding layers.
        ADDITIONAL_OUTPUT_MERGE_MODE = 'Add'          # Merge mode for the skip-connections (see keras.layers.merge.py)
        ATTENTION_SIZE = DECODER_HIDDEN_SIZE
        # Skip connections size
        SKIP_VECTORS_HIDDEN_SIZE = TARGET_TEXT_EMBEDDING_SIZE

        #QE vector config
        QE_VECTOR_SIZE = 75

        # Fully-Connected layers for initializing the first RNN state
        #       Here we should only specify the activation function of each layer
        #       (as they have a potentially fixed size)
        #       (e.g INIT_LAYERS = ['tanh', 'relu'])
        INIT_LAYERS = ['tanh']

        # Additional Fully-Connected layers applied before softmax.
        #       Here we should specify the activation function and the output dimension
        #       (e.g DEEP_OUTPUT_LAYERS = [('tanh', 600), ('relu', 400), ('relu', 200)])
        DEEP_OUTPUT_LAYERS = [('linear', TARGET_TEXT_EMBEDDING_SIZE)]

        # Regularizers
        WEIGHT_DECAY = 1e-4                           # L2 regularization
        RECURRENT_WEIGHT_DECAY = 0.                   # L2 regularization in recurrent layers

        DROPOUT_P = 0.                                # Percentage of units to drop (0 means no dropout)
        RECURRENT_INPUT_DROPOUT_P = 0.                # Percentage of units to drop in input cells of recurrent layers
        RECURRENT_DROPOUT_P = 0.                      # Percentage of units to drop in recurrent layers

        USE_NOISE = True                              # Use gaussian noise during training
        NOISE_AMOUNT = 0.01                           # Amount of noise

        USE_BATCH_NORMALIZATION = True                # If True it is recommended to deactivate Dropout
        BATCH_NORMALIZATION_MODE = 1                  # See documentation in Keras' BN

        USE_PRELU = False                             # use PReLU activations as regularizer
        USE_L2 = False                                # L2 normalization on the features

        DOUBLE_STOCHASTIC_ATTENTION_REG = 0.0         # Doubly stochastic attention (Eq. 14 from arXiv:1502.03044)

        SAMPLING_SAVE_MODE = 'listoflists'                        # 'list': Store in a text file, one sentence per line.
        VERBOSE = 1                                        # Verbosity level
        RELOAD = 0                                         # If 0 start training from scratch, otherwise the model
                                                        # Saved on epoch 'RELOAD' will be used
        RELOAD_EPOCH = True                                # Select whether we reload epoch or update number

        REBUILD_DATASET = True                             # Build again or use stored instance
        
        # Extra parameters for special trainings
        TRAIN_ON_TRAINVAL = False                          # train the model on both training and validation sets combined
        FORCE_RELOAD_VOCABULARY = False                    # force building a new vocabulary from the training samples
                                                        # applicable if RELOAD > 1

    # ================================================ #
    parameters = locals().copy()
    return parameters

def add_dependent_params(parameters):
    """
    Take the 'static' parameters and calculate any remaining parameters which depend on others.
    Parameters:

    parameters (dict): dict of parameters

    :return parameters: Dictionary of parameters
    """
    import os

    parameters['DATASET_NAME'] = parameters['TASK_NAME']
    parameters['DATA_ROOT_PATH'] = os.path.join(
        parameters['DATA_DIR'], parameters['DATASET_NAME'])
    parameters['MAPPING'] = os.path.join(parameters['DATA_ROOT_PATH'], 'mapping.%s_%s.pkl' % (
        parameters['SRC_LAN'], parameters['TRG_LAN']))
    parameters['BPE_CODES_PATH'] = os.path.join(
        parameters['DATA_ROOT_PATH'], '/training_codes.joint')
    parameters['MODEL_NAME'] = parameters['TASK_NAME'] + '_' + \
        parameters['SRC_LAN'] + parameters['TRG_LAN'] + \
        '_' + parameters['MODEL_TYPE']
    parameters['STORE_PATH'] = os.path.join(
        parameters['MODEL_DIRECTORY'], parameters['MODEL_NAME'])
    parameters['DATASET_STORE_PATH'] = parameters['STORE_PATH']
    max_src_in_len=parameters.get('MAX_SRC_INPUT_TEXT_LEN', None)
    if max_src_in_len == None:
        parameters['MAX_SRC_INPUT_TEXT_LEN'] = parameters['MAX_INPUT_TEXT_LEN']
    max_trg_in_len=parameters.get('MAX_TRG_INPUT_TEXT_LEN', None)
    if max_trg_in_len == None:
        parameters['MAX_TRG_INPUT_TEXT_LEN'] = parameters['MAX_INPUT_TEXT_LEN']

    return parameters
