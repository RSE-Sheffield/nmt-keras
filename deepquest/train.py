# -*- coding: utf-8 -*-
from __future__ import print_function
import ast
import os

from timeit import default_timer as timer
import yaml

from keras.utils import CustomObjectScope
from keras_wrapper.cnn_model import updateModel
from keras_wrapper.dataset import loadDataset, saveDataset
from keras_wrapper.extra.callbacks import EarlyStopping
from keras_wrapper.extra.read_write import pkl2dict, dict2pkl
from nmt_keras.nmt_keras import check_params

import deepquest.qe_models as modFactory
from deepquest.utils.callbacks import PrintPerformanceMetricOnEpochEndOrEachNUpdates
from deepquest.utils.logs import logger_setup
from deepquest import compare_params
from deepquest.data_engine.prepare_data import build_dataset, update_dataset_from_file, keep_n_captions, preprocessDoc

logger, logging = logger_setup('train')

def train_model(params, weights_dict=None, load_dataset=None, trainable_pred=True, trainable_est=True, weights_path=None):
    """
    Training function. Sets the training parameters from params. Build or loads the model and launches the training.
    :param params: Dictionary of network hyperparameters.
    :param load_dataset: Load dataset from file or build it from the parameters.
    :return: None
    """
    check_params(params)

    if params['RELOAD'] > 0:
        logger.info('Resuming training from epoch ' + str(params['RELOAD']))
        # Load data
        if load_dataset is None:
            if params['REBUILD_DATASET']:
                logging.info('Rebuilding dataset.')

                pred_vocab = params.get('PRED_VOCAB', None)
                if pred_vocab is not None:
                    dataset_voc = loadDataset(params['PRED_VOCAB'])
                    dataset = build_dataset(params, dataset_voc.vocabulary,
                                            dataset_voc.vocabulary_len)
                else:
                    dataset = build_dataset(params)
            else:
                logging.info('Updating dataset.')
                dataset = loadDataset(params['DATASET_STORE_PATH'] + '/Dataset_' + params['DATASET_NAME']
                                      + '_' + params['SRC_LAN'] + params['TRG_LAN'] + '.pkl')

                for split, filename in params['TEXT_FILES'].iteritems():
                    dataset = update_dataset_from_file(dataset,
                                                       params['DATA_ROOT_PATH'] + '/' +
                                                       filename +
                                                       params['SRC_LAN'],
                                                       params,
                                                       splits=list([split]),
                                                       output_text_filename=params['DATA_ROOT_PATH'] + '/' + filename +
                                                       params['TRG_LAN'],
                                                       remove_outputs=False,
                                                       compute_state_below=True,
                                                       recompute_references=True)
                    dataset.name = params['DATASET_NAME'] + '_' + \
                        params['SRC_LAN'] + params['TRG_LAN']
                saveDataset(dataset, params['DATASET_STORE_PATH'])

        else:
            logging.info('Reloading and using dataset.')
            dataset = loadDataset(load_dataset)
    else:
        # Load data
        if load_dataset is None:
            pred_vocab = params.get('PRED_VOCAB', None)
            if pred_vocab is not None:
                dataset_voc = loadDataset(params['PRED_VOCAB'])
                # for the testing pharse handle model vocab differences
                #dataset_voc.vocabulary['target_text'] = dataset_voc.vocabulary['target']
                #dataset_voc.vocabulary_len['target_text'] = dataset_voc.vocabulary_len['target']
                dataset = build_dataset(
                    params, dataset_voc.vocabulary, dataset_voc.vocabulary_len)
            else:
                if 'doc_qe' in params['OUTPUTS_IDS_MODEL']:
                    params = preprocessDoc(params)
                elif 'estimatordoc' in params['MODEL_TYPE'].lower():
                    raise Exception('Translation_Model model_type "' +
                                    params['MODEL_TYPE'] + '" is not implemented.')
                dataset = build_dataset(params)
                if params['NO_REF']:
                    keep_n_captions(dataset, repeat=1, n=1,
                                    set_names=params['EVAL_ON_SETS'])
        else:
            dataset = loadDataset(load_dataset)
    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    #params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['OUTPUTS_IDS_DATASET_FULL'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len['target_text']

    if params['RELOAD_EPOCH'] == True and params['RELOAD'] > 0:
        compare_params(params, pkl2dict(os.path.join(params['STORE_PATH'], 'config.pkl')), ignore=['RELOAD', 'RELOAD_EPOCH'])
        logger.info('Resuming training from epoch ' + str(params['RELOAD']))

    # Build model
    try:
        qe_model = modFactory.get(params['MODEL_TYPE'], params)

        # Define the inputs and outputs mapping from our Dataset instance to our model
        params['INPUTS_IDS_DATASET'] = qe_model.ids_inputs
        inputMapping = dict()
        for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
            pos_source = dataset.ids_inputs.index(id_in)
            id_dest = qe_model.ids_inputs[i]
            inputMapping[id_dest] = pos_source
        qe_model.setInputsMapping(inputMapping)

        params['OUTPUTS_IDS_DATASET'] = qe_model.ids_outputs
        outputMapping = dict()
        for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']):
            pos_target = dataset.ids_outputs.index(id_out)
            id_dest = qe_model.ids_outputs[i]
            outputMapping[id_dest] = pos_target
        qe_model.setOutputsMapping(outputMapping)

        if not params['RELOAD']:
            # if we don't reload, it means
            # we build a new model that needs
            # an optimizer
            qe_model.setOptimizer()
        else:
            # otherwise we just reload the weights
            # from the files containing the model
            from keras.utils import CustomObjectScope
            # includes all layers and everything defined in deepquest.qe_models.utils
            import deepquest.qe_models.layers as layers
            with CustomObjectScope(vars(layers)):
                qe_model = updateModel(
                    qe_model, params['STORE_PATH'], params['RELOAD'], reload_epoch=params['RELOAD_EPOCH'])
            qe_model.setParams(params)
            qe_model.setOptimizer()
            params['EPOCH_OFFSET'] = params['RELOAD'] if params['RELOAD_EPOCH'] else \
                int(params['RELOAD'] *
                    params['BATCH_SIZE'] / dataset.len_train)

    except AttributeError as error:
        logging.error("Error occured: {}".format(error))

    except Exception as exception:
        logging.exception("Exception occured: {}".format(exception))

    # Store configuration as pkl
    dict2pkl(params, params['STORE_PATH'] + '/config')

    # Callbacks
    callbacks = buildCallbacks(params, qe_model, dataset)

    # Training
    total_start_time = timer()

    logger.debug('Starting training!')
    training_params = {'n_epochs': params['MAX_EPOCH'],
                       'batch_size': params['BATCH_SIZE'],
                       'homogeneous_batches': params['HOMOGENEOUS_BATCHES'],
                       'maxlen': params['MAX_OUTPUT_TEXT_LEN'],
                       'joint_batches': params['JOINT_BATCHES'],
                       # LR decay parameters
                       'lr_decay': params.get('LR_DECAY', None),
                       'reduce_each_epochs': params.get('LR_REDUCE_EACH_EPOCHS', True),
                       'start_reduction_on_epoch': params.get('LR_START_REDUCTION_ON_EPOCH', 0),
                       'lr_gamma': params.get('LR_GAMMA', 0.9),
                       'lr_reducer_type': params.get('LR_REDUCER_TYPE', 'linear'),
                       'lr_reducer_exp_base': params.get('LR_REDUCER_EXP_BASE', 0),
                       'lr_half_life': params.get('LR_HALF_LIFE', 50000),
                       'epochs_for_save': params['EPOCHS_FOR_SAVE'],
                       'verbose': params['VERBOSE'],
                       'eval_on_sets': params['EVAL_ON_SETS_KERAS'],
                       'n_parallel_loaders': params['PARALLEL_LOADERS'],
                       'extra_callbacks': callbacks,
                       'reload_epoch': params['RELOAD'],
                       'epoch_offset': params.get('EPOCH_OFFSET', 0),
                       'data_augmentation': params['DATA_AUGMENTATION'],
                       # early stopping parameters
                       'patience': params.get('PATIENCE', 0),
                       'metric_check': None,
                       'eval_on_epochs': params.get('EVAL_EACH_EPOCHS', True),
                       'each_n_epochs': params.get('EVAL_EACH', 1),
                       'start_eval_on_epoch': params.get('START_EVAL_ON_EPOCH', 0),
                       'n_gpus': params.get('N_GPUS', 1),
                       }

    # if tensorboard is on, then add tensorboard params to training params
    if params.get('TENSORBOARD', False):
        tensorboard_training_params = {'tensorboard': params.get('TENSORBOARD', False),
                                       'tensorboard_params': {'log_dir': params.get('LOG_DIR', 'tensorboard_logs'),
                                                              'histogram_freq': params.get('HISTOGRAM_FREQ', 0),
                                                              'batch_size': params.get('TENSORBOARD_BATCH_SIZE', params['BATCH_SIZE']),
                                                              'write_graph': params.get('WRITE_GRAPH', True),
                                                              'write_grads': params.get('WRITE_GRADS', False),
                                                              'write_images': params.get('WRITE_IMAGES', False),
                                                              'embeddings_freq': params.get('EMBEDDINGS_FREQ', 0),
                                                              'embeddings_layer_names': params.get('EMBEDDINGS_LAYER_NAMES', None),
                                                              'embeddings_metadata': params.get('EMBEDDINGS_METADATA', None),
                                                              'label_word_embeddings_with_vocab': params.get(
                                           'LABEL_WORD_EMBEDDINGS_WITH_VOCAB', False),
            'word_embeddings_labels': params.get('WORD_EMBEDDINGS_LABELS', None),
        }
        }
        training_params.update(tensorboard_training_params)

    if weights_dict is not None:
        for layer in qe_model.model.layers:
            if layer.name in weights_dict:
                layer.set_weights(weights_dict[layer.name])

    qe_model.trainNet(dataset, training_params)
    if weights_dict is not None:
        for layer in qe_model.model.layers:
            weights_dict[layer.name] = layer.get_weights()

    total_end_time = timer()
    time_difference = total_end_time - total_start_time
    logging.info('In total is {0:.2f}s = {1:.2f}m'.format(
        time_difference, time_difference / 60.0))


def buildCallbacks(params, model, dataset):
    """
    Builds the selected set of callbacks run during the training of the model.

    :param params: Dictionary of network hyperparameters.
    :param model: Model instance on which to apply the callback.
    :param dataset: Dataset instance on which to apply the callback.
    :return:
    """

    callbacks = []

    if params['METRICS'] or params['SAMPLE_ON_SETS']:
        # Evaluate training
        extra_vars = {'language': params.get('TRG_LAN', 'en'),
                      'n_parallel_loaders': params['PARALLEL_LOADERS'],
                      'tokenize_f': eval('dataset.' + params.get('TOKENIZATION_METHOD', 'tokenize_none')),
                      'detokenize_f': eval('dataset.' + params.get('DETOKENIZATION_METHOD', 'detokenize_none')),
                      'apply_detokenization': params.get('APPLY_DETOKENIZATION', False),
                      'tokenize_hypotheses': params.get('TOKENIZE_HYPOTHESES', True),
                      'tokenize_references': params.get('TOKENIZE_REFERENCES', True)
                      }

        input_text_id = params['INPUTS_IDS_DATASET'][0]
        vocab_x = dataset.vocabulary[input_text_id]['idx2words']
        vocab_y = dataset.vocabulary[params['INPUTS_IDS_DATASET']
                                     [1]]['idx2words']
        if params['BEAM_SEARCH']:
            extra_vars['beam_size'] = params.get('BEAM_SIZE', 6)
            extra_vars['state_below_index'] = params.get(
                'BEAM_SEARCH_COND_INPUT', -1)
            extra_vars['maxlen'] = params.get('MAX_OUTPUT_TEXT_LEN_TEST', 30)
            extra_vars['optimized_search'] = params.get(
                'OPTIMIZED_SEARCH', True)
            extra_vars['model_inputs'] = params['INPUTS_IDS_MODEL']
            extra_vars['model_outputs'] = params['OUTPUTS_IDS_MODEL']
            extra_vars['dataset_inputs'] = params['INPUTS_IDS_DATASET']
            extra_vars['dataset_outputs'] = params['OUTPUTS_IDS_DATASET']
            extra_vars['search_pruning'] = params.get('SEARCH_PRUNING', False)
            extra_vars['normalize_probs'] = params.get(
                'NORMALIZE_SAMPLING', False)
            extra_vars['alpha_factor'] = params.get('ALPHA_FACTOR', 1.)
            extra_vars['coverage_penalty'] = params.get(
                'COVERAGE_PENALTY', False)
            extra_vars['length_penalty'] = params.get('LENGTH_PENALTY', False)
            extra_vars['length_norm_factor'] = params.get(
                'LENGTH_NORM_FACTOR', 0.0)
            extra_vars['coverage_norm_factor'] = params.get(
                'COVERAGE_NORM_FACTOR', 0.0)
            extra_vars['pos_unk'] = params['POS_UNK']
            extra_vars['output_max_length_depending_on_x'] = params.get(
                'MAXLEN_GIVEN_X', True)
            extra_vars['output_max_length_depending_on_x_factor'] = params.get(
                'MAXLEN_GIVEN_X_FACTOR', 3)
            extra_vars['output_min_length_depending_on_x'] = params.get(
                'MINLEN_GIVEN_X', True)
            extra_vars['output_min_length_depending_on_x_factor'] = params.get(
                'MINLEN_GIVEN_X_FACTOR', 2)

            if params['POS_UNK']:
                extra_vars['heuristic'] = params['HEURISTIC']
                if params['HEURISTIC'] > 0:
                    extra_vars['mapping'] = dataset.mapping

        if params['METRICS']:
            for s in params['EVAL_ON_SETS']:
                extra_vars[s] = dict()
                extra_vars[s]['references'] = dataset.extra_variables[s][params['OUTPUTS_IDS_DATASET'][0]]
            callback_metric = PrintPerformanceMetricOnEpochEndOrEachNUpdates(model,
                                                                             dataset,
                                                                             gt_id=[
                                                                                 params['OUTPUTS_IDS_DATASET'][0]],
                                                                             metric_name=params['METRICS'],
                                                                             set_name=params['EVAL_ON_SETS'],
                                                                             batch_size=params['BATCH_SIZE'],
                                                                             each_n_epochs=params['EVAL_EACH'],
                                                                             extra_vars=extra_vars,
                                                                             reload_epoch=params['RELOAD'],
                                                                             is_text=True,
                                                                             input_text_id=input_text_id,
                                                                             index2word_y=vocab_y,
                                                                             # index2word_y=dataset.vocabulary[params['OUTPUTS_IDS_DATASET'][0]]['idx2words'],
                                                                             index2word_x=vocab_x,
                                                                             sampling_type=params['SAMPLING'],
                                                                             beam_search=params['BEAM_SEARCH'],
                                                                             save_path=model.model_path,
                                                                             start_eval_on_epoch=params[
                                                                                 'START_EVAL_ON_EPOCH'],
                                                                             write_samples=True,
                                                                             write_type=params['SAMPLING_SAVE_MODE'],
                                                                             eval_on_epochs=params['EVAL_EACH_EPOCHS'],
                                                                             save_each_evaluation=params[
                                                                                 'SAVE_EACH_EVALUATION'],
                                                                             verbose=params['VERBOSE'],
                                                                             no_ref=params['NO_REF'],
                                                                             metric_check=params['STOP_METRIC'],
                                                                             want_to_minimize=True if params['STOP_METRIC'].lower() in ['rmse','mae'] else False)

            callbacks.append(callback_metric)

        # Early stopper
        if params['EARLY_STOP']:
            callback_early_stop = EarlyStopping(model,
                                                patience=params['PATIENCE'],
                                                metric_check=params['STOP_METRIC'],
                                                check_split=params['EVAL_ON_SETS'][0],
                                                want_to_minimize=True if params['STOP_METRIC'].lower() in ['rmse','mae'] else False,
                                                eval_on_epochs=params['EVAL_EACH_EPOCHS'],
                                                each_n_epochs=params['EVAL_EACH'],
                                                start_eval_on_epoch=params['START_EVAL_ON_EPOCH'])
            callbacks.append(callback_early_stop)

    return callbacks


def save_random_states(write_path, user_seed=None):
    import codecs
    import numpy as np
    import random
    # import json
    import pickle

    write_path = os.path.join(write_path, 'assets')

    if not os.path.exists(write_path):
        os.makedirs(write_path)

    with codecs.open(os.path.join(write_path, 'np_states.pkl'), 'wb') as fh:
        pickle.dump(np.random.get_state(), fh)

    with codecs.open(os.path.join(write_path, 'random_states.pkl'), 'wb') as fh:
        pickle.dump(random.getstate(), fh)

    # n = list(numpy.random.get_state())
    # n[1] = n[1].tolist()
    #
    # p = list(random.getstate())
    #
    # data = {'user_seed': user_seed,
    #         'numpy_rand_state': n,
    #         'python_rand_state': p
    #         }
    #
    # with open(os.path.join(write_path, 'random_states.json'), 'w') as outfile:
    #     json.dump(data, outfile)


def main(parameters):
    """
    Handles QE model training.
    :param config: Either a path to a YAML or pkl config file or a dictionary of parameters.
    :param dataset: Optional path to a previously built pkl dataset.
    :param changes: Optional dictionary of parameters to overwrite config.
    """
    
    logger.info(parameters)

    # check if model already exists
    if not os.path.exists(parameters['STORE_PATH']):
        # if model doesn't already exist
        # write out initial parameters to pkl
        os.makedirs(parameters['STORE_PATH'])
        dataset = None
        parameters['RELOAD'] = 0
        parameters['RELOAD_EPOCH'] = False
    else:
        # if model already exists check that only RELOAD or RELOAD_EPOCH differ
        logger.info('Model ' + parameters['STORE_PATH'] + ' already exists.')
        # logger.info('Loading trained model config.pkl from ' + prev_config)
        if parameters['RELOAD_EPOCH'] != True or parameters['RELOAD'] == 0:
            logger.info(
                'Specify RELOAD_EPOCH=True and RELOAD>0 in your config to resume training an existing model. ')
            return
        # compare_params(parameters, pkl2dict(os.path.join(parameters['STORE_PATH'], 'config.pkl')), ignore=['RELOAD', 'RELOAD_EPOCH'])
        # logger.info('Resuming training from epoch ' + str(parameters['RELOAD']))

        # if there is a pre-trained model and dataset is not specified earlier, set the path to load the existing dataset
        dataset = parameters['DATASET_STORE_PATH'] + '/Dataset_' + parameters['DATASET_NAME'] + \
            '_' + parameters['SRC_LAN'] + parameters['TRG_LAN'] + '.pkl'


    check_params(parameters) # nmt-keras' check_params function

    save_random_states(parameters['STORE_PATH'], user_seed=parameters.get('SEED'))

    if parameters['MULTI_TASK']:

        total_epochs = parameters['MAX_EPOCH']
        epoch_per_update = parameters['EPOCH_PER_UPDATE']

        weights_dict = dict()
        for i in range(total_epochs):

            for output in parameters['OUTPUTS_IDS_DATASET_FULL']:

                trainable_est = True
                trainable_pred = True

                if i > 0 and 'PRED_WEIGHTS' in parameters:
                    del parameters['PRED_WEIGHTS']
                    #parameters['PRED_WEIGHTS'] = os.getcwd()+'/trained_models/'+parameters['MODEL_NAME']+'/epoch_'+str(parameters['EPOCH_PER_MODEL'])+'_weights.h5'

                parameters['OUTPUTS_IDS_DATASET'] = [output]
                parameters['OUTPUTS_IDS_MODEL'] = [output]

                if output == 'target_text' and i > 0:

                    parameters['MODEL_TYPE'] = 'Predictor'
                    parameters['MODEL_NAME'] = 'Predictor'
                    parameters['EPOCH_PER_MODEL'] = parameters['EPOCH_PER_PRED']
                    parameters['LOSS'] = 'categorical_crossentropy'

                elif output == 'sent_hter':

                    parameters['MODEL_TYPE'] = 'EstimatorSent'
                    parameters['MODEL_NAME'] = 'EstimatorSent'
                    parameters['EPOCH_PER_MODEL'] = parameters['EPOCH_PER_EST_SENT']
                    parameters['LOSS'] = 'mse'
                    if i == 0:
                        trainable_pred = False

                elif output == 'word_qe':

                    parameters['MODEL_TYPE'] = 'EstimatorWord'
                    parameters['MODEL_NAME'] = 'EstimatorWord'
                    parameters['EPOCH_PER_MODEL'] = parameters['EPOCH_PER_EST_WORD']
                    parameters['LOSS'] = 'categorical_crossentropy'
                    if i == 0:
                        trainable_pred = False

                else:
                    continue

                for j in range(epoch_per_update):

                    logging.info('Running training task for ' +
                                 parameters['MODEL_NAME'])
                    parameters['MAX_EPOCH'] = parameters['EPOCH_PER_MODEL']

                    train_model(parameters, weights_dict, load_dataset=dataset, trainable_est=trainable_est,
                                trainable_pred=trainable_pred, weights_path=parameters.get('PRED_WEIGHTS', None))

                    flag = True
    else:

        logging.info('Running training task.')
        train_model(parameters,weights_dict=None, load_dataset=dataset, trainable_est=True, trainable_pred=True,
                    weights_path=parameters.get('PRED_WEIGHTS', None))

    logger.info('Done!')
