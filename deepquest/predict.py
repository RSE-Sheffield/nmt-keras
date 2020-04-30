import ast
import os
import sys

import pickle

from keras.utils import CustomObjectScope
from keras_wrapper.cnn_model import loadModel
from keras_wrapper.dataset import loadDataset

import deepquest.qe_models as modFactory
from deepquest.data_engine.dataset import Dataset
from deepquest.data_engine.prepare_data import build_dataset, update_dataset_from_file, keep_n_captions
# from deepquest.utils import evaluation
from deepquest.utils.callbacks import *

logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)



def apply_QE_model(model, dataset, predict_parameters):

    """
    Apply the QE model on the test set of the dataset.
    (We previously place the data for which we want predictions in the test set of the dataset.)

    :param model: QE model to produce the predictions with
    :param dataset: Dataset instance on which the predictions will be produced
    :param predict_parameters: parameters used by predictNet() to predict

    :return: predictions
    """
    predictions = ()

    try:
        # input_list = []
        # for input_name, pos in model.inputsMapping.items():
        #     ds_input_pos = dataset.ids_inputs.index(input_name)
        #     input_list.append(
        #             np.array(dataset.getX('test', 0, dataset.len_test)[ds_input_pos])
        #             )
        #
        # if len(input_list) != len(model.model.input_names):
        #     raise Exception('The number of inputs in input_list does not match \
        #             the number of inputs of the model!')

        # predictions = model.predict(inputs)
        raw_predictions = model.predictNet(dataset, predict_parameters)
        # raw_predictions = model.model.predict(input_list)
        if predict_parameters['word-level']:
            # we retrieve the threshold to determine whether a token is OK or BAD
            # fom the evaluation during training of the model
            # (see evaluate() in callbacks.py)
            threshold = model.getLog('val', 'threshold')[-1]
            # print("threshold is {}".format(threshold))
            predictions = {}
            target_text = dataset.X_test['target_text']
            # print(raw_predictions[predict_parameters['predict_on_sets'][0]])
            for i in range(len(target_text)):
                pred = []
                # we use the length of the target sent to retrieve the number
                # of labels that we need to produce, otherwise padding to 70...
                mt_sent = target_text[i].split(' ')
                # logging.info('y_init[i]: %s' % y_init[i])
                for j in range(len(mt_sent)):
                    # pred_word = raw_predictions[i][j]
                    pred_word = raw_predictions[predict_parameters['predict_on_sets'][0]][i][j]
                    # print(pred_word)
                    if pred_word[dataset.vocabulary['word_qe']['words2idx']['OK']] >= threshold:
                        pred.append('OK')
                    else:
                        pred.append('BAD')
            # logging.info('y_pred: %s' % y_pred)
                predictions[i] = np.array(pred)

        else:
            predictions = raw_predictions[predict_parameters['predict_on_sets'][0]].reshape(
                    raw_predictions[predict_parameters['predict_on_sets'][0]].shape[0]
                    )

            if len(predictions) != dataset.len_test:
                raise Exception('The number of predictions ({}) does not match the size \
                        of the test set ({})!'.format(len(predictions, dataset.len_test)))

    except ValueError as e:
        logger.error("ValueError exception occurred: {}".format(e))
        sys.exit(1)

    except Exception as e:
        logger.error("Exception occurred: {}".format(e))
        sys.exit(1)

    return predictions


def main(parameters):
    """
    Predict on unseen data
    """
    try:
        model_path = parameters.get('LOAD_MODEL', None)
        src_file2predict = parameters.get('TEST_SRC_FILE', None)
        trg_file2predict = parameters.get('TEST_TRG_FILE', None)
        feature_file = parameters.get('FEAT_FILE', None)

        if not model_path:
            raise ValueError('Model to load not specificied, can\'t predict without it!')
        if not src_file2predict:
            raise ValueError('Sentences in source language are missing, can\'t predict without them!')
        if not trg_file2predict:
            raise ValueError('Sentences in target language are missing, can\'t predict without them!')

    except ValueError as e:
        print('Error occurred: {}'.format(e))
        sys.exit(1)

    save_path = parameters.get('PRED_PATH', None)
    if save_path is None:
        save_path = os.path.join(parameters['STORE_PATH'], 'predictions')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, 'predictions.txt')

    logger.info('Loading pre-trained QE model...')

    try:
        # includes all layers and everything defined in deepquest.qe_models.utils
        import deepquest.qe_models.layers as layers
        with CustomObjectScope(vars(layers)):
            print(model_path)
            qe_model = loadModel(model_path, -1, full_path=True)

    except Exception as e:
        logger.error('Exception occurred while loading pre-trained QE model: {}'.format(e))
        sys.exit(1)

    logger.info('QE model loaded...')
    logger.info('Processing dataset...')

    try:
        # Loading the dataset/vocab used to train the pre-trained model
        # information is given from the config.pkl file used as default params
        training_ds = pickle.load(open(os.path.join(
            parameters['DATASET_STORE_PATH'],
            'Dataset_{}_{}{}.pkl'.format(
                parameters['DATASET_NAME'],
                parameters['SRC_LAN'], parameters['TRG_LAN'])
            ),
            'rb')
            )

        # We create a Dataset object with the (unseen) data we wish to
        # predict the quality for 
        # note: no need to save it
        # name = params['DATASET_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN']
        predict_ds = Dataset('prediction_tmp', '/', silence=False)
        predict_ds.vocabulary = training_ds.vocabulary
        predict_ds.vocabulary_len = training_ds.vocabulary_len

        # We check whether we are doing word-level QE as the predictions would require
        # a little extra post-processing
        word_level = False
        if 'word' in qe_model.params['MODEL_TYPE'].lower():
            word_level = True

        # Getting informatiom from both the pre-trained model and
        # the used dataset, to know how to process the unseen data
        use_bert = False
        if 'bert' in parameters['TOKENIZATION_METHOD'].lower():
            use_bert = True

        inputs_offsets = {}
        for input_name in qe_model.model.input_names:
            # if BERT is used, we skip those entries as they are processed 
            # automatically whitin the preprocessText() function of the Dataset class
            if use_bert and 'mask' in input_name or 'segids' in input_name:
                continue
            # retrieving offset information from the used dataset 
            # (offset is used e.g. by Predictor model)
            inputs_offsets[input_name] = training_ds.text_offset[input_name]

        # Addint RAW input (similarly to build_dataset()
        # ds.setRawInput(src_file2predict,
        #         'test',
        #         type='file-name',
        #         id='raw_source_text')

        # Adding the input source sentences
        predict_ds.setInput(src_file2predict,
                'test',
                type=parameters.get('INPUTS_TYPES_DATASET', ['text', 'text'])[0],
                id='source_text',
                tokenization=parameters.get('TOKENIZATION_METHOD', 'tokenize_none'),
                build_vocabulary=False,
                pad_on_batch=parameters.get('PAD_ON_BATCH', True),
                offset=inputs_offsets['source_text'],
                fill=parameters.get('FILL', 'end'),
                max_text_len=parameters.get('MAX_INPUT_TEXT_LEN', 100),
                max_words=parameters.get('INPUT_VOCABULARY_SIZE', 0),
                min_occ=parameters.get('MIN_OCCURRENCES_INPUT_VOCAB', 0),
                bpe_codes=parameters.get('BPE_CODES_PATH', None),
                overwrite_split=True)

        # Adding the input target sentences
        predict_ds.setInput(trg_file2predict,
                'test',
                type=parameters.get('INPUTS_TYPES_DATASET', ['text', 'text'])[0],
                id='target_text',
                tokenization=parameters.get('TOKENIZATION_METHOD', 'tokenize_none'),
                build_vocabulary=False,
                pad_on_batch=parameters.get('PAD_ON_BATCH', True),
                offset=inputs_offsets['target_text'],
                fill=parameters.get('FILL', 'end'),
                max_text_len=parameters.get('MAX_INPUT_TEXT_LEN', 100),
                max_words=parameters.get('INPUT_VOCABULARY_SIZE', 0),
                min_occ=parameters.get('MIN_OCCURRENCES_INPUT_VOCAB', 0),
                bpe_codes=parameters.get('BPE_CODES_PATH', None),
                overwrite_split=True)

        # TODO: considering when the QE model as more than 2 inputs (e.g. Predictor-Estimator)

        # TODO: considering extra input, such as visual/text features (e.g. multimodalQE)
        # Adding extra features (if any)
        # if feature_file:
        #     predict_ds.setInput(src_file2predict,
        #             'test',
        #             type=parameters.get('INPUTS_TYPES_DATASET', ['text', 'text'])[0],
        #             id='source_text',
        #             tokenization=parameters.get('TOKENIZATION_METHOD', 'tokenize_none'),
        #             build_vocabulary=False,
        #             pad_on_batch=parameters.get('PAD_ON_BATCH', True),
        #             fill=parameters.get('FILL', 'end'),
        #             max_text_len=parameters.get('MAX_INPUT_TEXT_LEN', 100),
        #             max_words=parameters.get('INPUT_VOCABULARY_SIZE', 0),
        #             min_occ=parameters.get('MIN_OCCURRENCES_INPUT_VOCAB', 0),
        #             bpe_codes=parameters.get('BPE_CODES_PATH', None),
        #             overwrite_split=True)

        # Note: no need to set an output!
        # We pretend that there is no ref and
        # we leave evaluation to the scoring pipeline

        # We make sure to associate the inputs IDs of the model to
        # the right inputs IDs of the new dataset (unseen data)
        parameters['INPUTS_IDS_DATASET'] = qe_model.ids_inputs
        inputMapping = dict()
        for i, id_in in enumerate(parameters['INPUTS_IDS_DATASET']):
            pos_source = predict_ds.ids_inputs.index(id_in)
            id_dest = qe_model.ids_inputs[i]
            inputMapping[id_dest] = pos_source
        qe_model.setInputsMapping(inputMapping)

    except Exception as e:
        logger.error('Exception occurred when processing dataset: {}'.format(e))
        sys.exit(1)

    logger.info('Processing dataset done.')
    logger.info('Predicting estimates on data.')

    predict_parameters = {
            'predict_on_sets': ['test'],
            'batch_size': parameters.get('BATCH_SIZE', 50),
            'verbose': parameters.get('VERBOSE', 0),
            'word-level': word_level,
            # 'model_name': 'model' # name of the attribute where the model for prediction is stored
            }

    predictions = apply_QE_model(qe_model, predict_ds, predict_parameters)

    logger.info('Predicting estimates DONE.')
    logger.info('Saving predictions in {}'.format(save_file))

    with codecs.open(save_file, 'w+', encoding='utf-8') as fh_pred:
        if word_level:
            for i in range(len(predictions)):
                fh_pred.write(' '.join(predictions[i]) + '\n')
        else:
            for pred in predictions:
                fh_pred.write(str(pred) + '\n')


    logger.info('Predictions saved.')

