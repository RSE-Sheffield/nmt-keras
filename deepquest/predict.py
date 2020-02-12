import ast
import os
import sys

import pickle

from keras.utils import CustomObjectScope
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.dataset import loadDataset

import deepquest.qe_models as modFactory
from deepquest.utils.prepare_data import build_dataset, update_dataset_from_file, keep_n_captions
from deepquest.utils import evaluation
from deepquest.utils.callbacks import *

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def apply_NMT_model(params, dataset, model, save_path):
    """
    Sample from a previously trained model.

    :param params: Dictionary of network hyperparameters.
    :return: None
    """
    params['PRED_VOCAB'] = dataset
    dataset = loadDataset(params['PRED_VOCAB'])
    # dataset = build_dataset(params, dataset_voc.vocabulary, dataset_voc.vocabulary_len)
    if 'test' in params['EVAL_ON_SETS'] and len(dataset.ids_inputs) != len(dataset.types_inputs['test']):
        dataset.ids_inputs = dataset.ids_inputs[1:4]

    params['INPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len[params['INPUTS_IDS_DATASET'][0]]
    params['OUTPUT_VOCABULARY_SIZE'] = dataset.vocabulary_len['target_text']

    # Load model
    nmt_model = loadModel(model, -1, full_path=True)

    nmt_model.model_path = save_path

    inputMapping = dict()
    for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
        pos_source = dataset.ids_inputs.index(id_in)
        id_dest = nmt_model.ids_inputs[i]
        inputMapping[id_dest] = pos_source
    nmt_model.setInputsMapping(inputMapping)

    outputMapping = dict()
    for i, id_out in enumerate(params['OUTPUTS_IDS_DATASET']):
        pos_target = dataset.ids_outputs.index(id_out)
        id_dest = nmt_model.ids_outputs[i]
        outputMapping[id_dest] = pos_target
    nmt_model.setOutputsMapping(outputMapping)
    nmt_model.setOptimizer()

    for s in params['EVAL_ON_SETS']:
        # Evaluate training
        extra_vars = {'language': params.get('TRG_LAN', 'en'),
                      'n_parallel_loaders': params['PARALLEL_LOADERS'],
                      'tokenize_f': eval('dataset.' + params['TOKENIZATION_METHOD']),
                      'detokenize_f': eval('dataset.' + params['DETOKENIZATION_METHOD']),
                      'apply_detokenization': params['APPLY_DETOKENIZATION'],
                      'tokenize_hypotheses': params['TOKENIZE_HYPOTHESES'],
                      'tokenize_references': params['TOKENIZE_REFERENCES']}

        extra_vars[s] = dict()
        # True when we should score against a reference

        # add the test split reference to the dataset
        path_list = os.path.join(params['DATA_ROOT_PATH'], s + '.' + params['PRED_SCORE'])
        if dataset.ids_outputs[0] == 'word_qe':
            out_type = 'text'
        else:
            out_type = 'real'

        if not params.get('NO_REF', False):
            if not dataset.loaded_raw_test[1] and s == 'test':
                dataset.setRawOutput(path_list, set_name=s, type='file-name', id='raw_'+id_out, overwrite_split=False,
                                     add_additional=False)
            if not dataset.loaded_test[1] and s == 'test':
                dataset.setOutput(path_list, set_name=s, type=out_type, id=id_out, repeat_set=1, overwrite_split=False,
                                  add_additional=False, sample_weights=False, label_smoothing=0.,
                                  tokenization='tokenize_none', max_text_len=0, offset=0, fill='end', min_occ=0,  # 'text'
                                  pad_on_batch=True, words_so_far=False, build_vocabulary=False, max_words=0,  # 'text'
                                  bpe_codes=None, separator='@@', use_unk_class=False,  # 'text'
                                  associated_id_in=None, num_poolings=None,  # '3DLabel' or '3DSemanticLabel'
                                  sparse=False,  # 'binary'
                                  )
            keep_n_captions(dataset, repeat=1, n=1, set_names=params['EVAL_ON_SETS'])

        input_text_id = params['INPUTS_IDS_DATASET'][0]
        vocab_x = dataset.vocabulary[input_text_id]['idx2words']
        vocab_y = dataset.vocabulary[params['INPUTS_IDS_DATASET'][1]]['idx2words']

        callbacks = buildCallbacks(params, nmt_model, dataset)
        metrics = callbacks.evaluate(
            params['RELOAD'], counter_name='epoch' if params['EVAL_EACH_EPOCHS'] else 'update')


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
        vocab_y = dataset.vocabulary[params['INPUTS_IDS_DATASET'][1]]['idx2words']
        if params['BEAM_SEARCH']:
            extra_vars['beam_size'] = params.get('BEAM_SIZE', 6)
            extra_vars['state_below_index'] = params.get('BEAM_SEARCH_COND_INPUT', -1)
            extra_vars['maxlen'] = params.get('MAX_OUTPUT_TEXT_LEN_TEST', 30)
            extra_vars['optimized_search'] = params.get('OPTIMIZED_SEARCH', True)
            extra_vars['model_inputs'] = params['INPUTS_IDS_MODEL']
            extra_vars['model_outputs'] = params['OUTPUTS_IDS_MODEL']
            extra_vars['dataset_inputs'] = params['INPUTS_IDS_DATASET']
            extra_vars['dataset_outputs'] = params['OUTPUTS_IDS_DATASET']
            extra_vars['search_pruning'] = params.get('SEARCH_PRUNING', False)
            extra_vars['normalize_probs'] = params.get('NORMALIZE_SAMPLING', False)
            extra_vars['alpha_factor'] = params.get('ALPHA_FACTOR', 1.)
            extra_vars['coverage_penalty'] = params.get('COVERAGE_PENALTY', False)
            extra_vars['length_penalty'] = params.get('LENGTH_PENALTY', False)
            extra_vars['length_norm_factor'] = params.get('LENGTH_NORM_FACTOR', 0.0)
            extra_vars['coverage_norm_factor'] = params.get('COVERAGE_NORM_FACTOR', 0.0)
            extra_vars['pos_unk'] = params['POS_UNK']
            extra_vars['output_max_length_depending_on_x'] = params.get('MAXLEN_GIVEN_X', True)
            extra_vars['output_max_length_depending_on_x_factor'] = params.get(
                'MAXLEN_GIVEN_X_FACTOR', 3)
            extra_vars['output_min_length_depending_on_x'] = params.get('MINLEN_GIVEN_X', True)
            extra_vars['output_min_length_depending_on_x_factor'] = params.get(
                'MINLEN_GIVEN_X_FACTOR', 2)

            if params['POS_UNK']:
                extra_vars['heuristic'] = params['HEURISTIC']
                if params['HEURISTIC'] > 0:
                    extra_vars['mapping'] = dataset.mapping

        if params['METRICS']:
            for s in params['EVAL_ON_SETS']:
                extra_vars[s] = dict()
                if not params.get('NO_REF', False):
                    extra_vars[s]['references'] = dataset.extra_variables[s][params['OUTPUTS_IDS_DATASET'][0]]
            callback_metric = EvalPerformance(model,
                                              dataset,
                                              gt_id=[params['OUTPUTS_IDS_DATASET'][0]],
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
                                              no_ref=params['NO_REF'])

            callbacks = callback_metric

    return callbacks


def check_params(params):
    """
    Checks some typical parameters and warns if something wrong was specified.
    :param params: Model instance on which to apply the callback.
    :return: None
    """
    if params['POS_UNK']:
        assert params['OPTIMIZED_SEARCH'], 'Unknown words replacement requires ' \
                                           'to use the optimized search ("OPTIMIZED_SEARCH" parameter).'
    if params['COVERAGE_PENALTY']:
        assert params['OPTIMIZED_SEARCH'], 'The application of "COVERAGE_PENALTY" requires ' \
                                           'to use the optimized search ("OPTIMIZED_SEARCH" parameter).'
    if params['SRC_PRETRAINED_VECTORS'] and params['SRC_PRETRAINED_VECTORS'][:-1] != '.npy':
        warnings.warn('It seems that the pretrained word vectors provided for the target text are not in npy format.'
                      'You should preprocess the word embeddings with the "utils/preprocess_*_word_vectors.py script.')

    if params['TRG_PRETRAINED_VECTORS'] and params['TRG_PRETRAINED_VECTORS'][:-1] != '.npy':
        warnings.warn('It seems that the pretrained word vectors provided for the target text are not in npy format.'
                      'You should preprocess the word embeddings with the "utils/preprocess_*_word_vectors.py script.')


def main(model, dataset, save_path=None, evalset=None, changes={}):
    """
    Predicts QE scores on a dataset using a pre-trained model.
    :param model: Model file (.h5) to use.
    :param dataset: Dataset file (.pkl) to use.
    :param save_path: Optinal directory path to save predictions to. Default = STORE_PATH
    :param evalset: Optional set to evaluate on. Default = 'test'
    :param changes: Optional dictionary of parameters to overwrite config.
    """
    parameters = pickle.load(open(os.path.join(os.path.split(model)[0], 'config.pkl'), 'rb'))

    parameters.update(changes)

    if evalset is None:
        parameters['EVAL_ON_SETS'] = ['test']
    else:
        parameters['EVAL_ON_SETS'] = [evalset]

    if save_path is None:
        save_path = parameters['STORE_PATH']

    logging.info('Running prediction.')

    # NMT Keras expects model path to appear without the .h5
    if model.endswith(".h5"):
        model = model[:-3]
    # Directory containing model
    model_dir, file_name = os.path.split(model)
    _, parameters["MODEL_NAME"] = os.path.split(model_dir)
    parameters["STORE_PATH"] = "trained_models/" + parameters["MODEL_NAME"]
    del parameters["TASK_NAME"]
    assert file_name.startswith("epoch_")
    parameters["RELOAD"] = file_name.replace("epoch_", "")

    # from nmt_keras import model_zoo
    from keras.utils import CustomObjectScope
    import deepquest.qe_models.utils as layers  # includes all layers and everything defined in deepquest.qe_models.utils
    with CustomObjectScope(vars(layers)):
        apply_NMT_model(parameters, dataset, model, save_path)

    logging.info('Done.')
