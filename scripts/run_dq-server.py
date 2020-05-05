#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# run_dq-server.py
#
# Copyright (C) 2020 Frederic Blain (feedoo) <f.blain@sheffield.ac.uk>
#
# Licensed under the "THE BEER-WARE LICENSE" (Revision 42):
# Fred (feedoo) Blain wrote this file. As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a tomato juice or coffee in return
#

"""
-- USAGE

# Start the server:
    `python run_dq_server.py -c [config]`

# Submita a request via Python:
    `python query_dq-server.py`

(Inspired by https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html)
"""

# import the necessary packages
import io
import os
import sys
import codecs
import flask
import pickle
import json
import numpy as np
import random

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

params = None
dataset = None
model = None
graph = None
inputs_offsets = None
use_bert = False


def load_model(model_path):
    import deepquest.qe_models.layers as layers
    from keras.utils import CustomObjectScope
    from keras_wrapper.cnn_model import loadModel

    global model

    # load the pre-trained Keras model (here we are using a model
    # pre-trained on ImageNet and provided by Keras, but you can
    # substitute in your own networks just as easily)
    try:
        model2load = os.path.join(model_path, 'best_model')
        with CustomObjectScope(vars(layers)):
            model = loadModel(model2load, -1, full_path=True)
            for layer in model.model.layers: layer.trainable = False

    except Exception as e:
        logger.error('Exception occurred when loading the model: {}'.format(e))
        sys.exit(1)


def load_dataset():
    from deepquest.data_engine.dataset import Dataset

    global dataset
    global use_bert

    try:
        # Loading the dataset/vocab used to train the pre-trained model
        # information is given from the config.pkl file used as default params
        # params['DATASET_STORE_PATH'],
        training_ds = pickle.load(open(os.path.join(
            params['LOAD_MODEL'],
            'Dataset_{}_{}{}.pkl'.format(
                params['DATASET_NAME'],
                params['SRC_LAN'], params['TRG_LAN'])
            ),
            'rb')
            )

        # We create a Dataset object with the (unseen) data we wish to
        # predict the quality for 
        # note: no need to save it
        dataset = Dataset('prediction_tmp', '/', silence=False)
        dataset.vocabulary = training_ds.vocabulary
        dataset.vocabulary_len = training_ds.vocabulary_len

        # Getting informatiom from both the pre-trained model and
        # the used dataset, to know how to process the unseen data
        if 'bert' in params['TOKENIZATION_METHOD'].lower():
            use_bert = True

        global inputs_offsets
        inputs_offsets = {}
        for input_name in model.model.input_names:
            # if BERT is used, we skip those entries as they are processed 
            # automatically whitin the preprocessText() function of the Dataset class
            if use_bert and 'mask' in input_name or 'segids' in input_name:
                continue
            # retrieving offset information from the used dataset 
            # (offset is used e.g. by Predictor model)
            inputs_offsets[input_name] = training_ds.text_offset[input_name]

    except Exception as e:
        logger.error('Exception occurred when processing dataset: {}'.format(e))
        sys.exit(1)


def prepare_dataset(src_txt, tgt_txt):

    global dataset

    if isinstance(src_txt, str):
        src_txt = [src_txt]

    if isinstance(tgt_txt, str):
        tgt_txt = [tgt_txt]

    try:
        # Adding the input source sentence(s)
        dataset.setInput(src_txt,
                'test',
                type=params.get('INPUTS_TYPES_DATASET', ['text', 'text'])[0],
                id='source_text',
                tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                build_vocabulary=False,
                pad_on_batch=params.get('PAD_ON_BATCH', True),
                offset=inputs_offsets['source_text'],
                fill=params.get('FILL', 'end'),
                max_text_len=params.get('MAX_INPUT_TEXT_LEN', 100),
                max_words=params.get('INPUT_VOCABULARY_SIZE', 0),
                min_occ=params.get('MIN_OCCURRENCES_INPUT_VOCAB', 0),
                bpe_codes=params.get('BPE_CODES_PATH', None),
                overwrite_split=True)

        # Adding the input target sentence(s)
        dataset.setInput(tgt_txt,
                'test',
                type=params.get('INPUTS_TYPES_DATASET', ['text', 'text'])[0],
                id='target_text',
                tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                build_vocabulary=False,
                pad_on_batch=params.get('PAD_ON_BATCH', True),
                offset=inputs_offsets['target_text'],
                fill=params.get('FILL', 'end'),
                max_text_len=params.get('MAX_INPUT_TEXT_LEN', 100),
                max_words=params.get('INPUT_VOCABULARY_SIZE', 0),
                min_occ=params.get('MIN_OCCURRENCES_INPUT_VOCAB', 0),
                bpe_codes=params.get('BPE_CODES_PATH', None),
                overwrite_split=True)

        # We make sure to associate the inputs IDs of the model to
        # the right inputs IDs of the new dataset (unseen data)
        params['INPUTS_IDS_DATASET'] = model.ids_inputs
        inputMapping = dict()
        for i, id_in in enumerate(params['INPUTS_IDS_DATASET']):
            pos_source = dataset.ids_inputs.index(id_in)
            id_dest = model.ids_inputs[i]
            inputMapping[id_dest] = pos_source
        model.setInputsMapping(inputMapping)

    except Exception as e:
        logger.error('Exception occurred when preparing dataset: {}'.format(e))
        sys.exit(1)


@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    data = {"success": False}

    if flask.request.method == "POST":
        try:
            request = flask.request.get_json()

            with graph.as_default():
                # print(model.model.summary())
                logger.info("NEW REQUEST -- {}".format(request))

                data["predictions"] = []

                ## process the input we want to predict
                to_predict = json.loads(request)
                if 'src' in to_predict:
                    src_txt = to_predict['src']
                else:
                    raise Exception("No 'src' in request: {}".format(request))

                if 'tgt' in to_predict:
                    tgt_txt = to_predict['tgt']
                else:
                    raise Exception("No 'tgt' in request: {}".format(request))

                prepare_dataset(src_txt, tgt_txt)

                predict_parameters = {
                        'predict_on_sets': ['test'],
                        'batch_size': params.get('BATCH_SIZE', 50),
                        'verbose': params.get('VERBOSE', 0),
                }
                raw_predictions = model.predictNet(dataset, predict_parameters)

                # We check whether we are doing word-level QE as the predictions would require
                # a little extra post-processing
                # TODO: dirty, to change at some point
                if 'word' in model.params['MODEL_TYPE'].lower():
                    data["predlevel"] = "word"

                    # we retrieve the threshold to determine whether a token is OK or BAD
                    # fom the evaluation during training of the model
                    # (see evaluate() in callbacks.py)
                    threshold = model.getLog('val', 'threshold')[-1]

                    predictions = {}
                    if use_bert:
                        target_text = dataset.X_test['target_text_tok']
                    else:
                        target_text = dataset.X_test['target_text']

                    for i in range(len(target_text)):
                        pred = []
                        # we use the length of the target sent to retrieve the number
                        # of labels that we need to produce, otherwise padding to 70...
                        if use_bert:
                            # we do not consider '[CLS]' nor '[SEP]'
                            mt_sent = target_text[i].split(' ')[1:-1]
                        else:
                            mt_sent = target_text[i].split(' ')
                        for j in range(len(mt_sent)):
                            if use_bert:
                                if mt_sent[j].startswith('##'):
                                    continue
                                # we do not consider '[CLS]' nor '[SEP]' and beyond
                                pred_word = raw_predictions[predict_parameters['predict_on_sets'][0]][i][1:len(mt_sent)+1][j]
                            else:
                                pred_word = raw_predictions[predict_parameters['predict_on_sets'][0]][i][j]
                            if pred_word[dataset.vocabulary['word_qe']['words2idx']['OK']] >= threshold:
                                pred.append('OK')
                            else:
                                pred.append('BAD')
                        predictions[i] = pred

                    data["predictions"] = predictions

                # SENTENCE-LEVEL (Doc-level is not supported yet)
                else:
                    predictions = raw_predictions[predict_parameters['predict_on_sets'][0]].reshape(
                            raw_predictions[predict_parameters['predict_on_sets'][0]].shape[0]
                            )
                    data["predictions"] = predictions.tolist()
                    data["predlevel"] = "sentence"

                # indicate that the request was a success
                data["success"] = True

        except ValueError as e:
            logger.error("ValueError exception occurred: {}".format(e))
            sys.exit(1)

        except Exception as e:
            logger.error("Exception occurred: {}".format(e))
            sys.exit(1)

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


# if __name__ == "__main__":
def main(config):
    global graph
    global params

    from deepquest.utils.utils import setparameters

    # params = setparameters(sys.argv[1])
    params = setparameters(config)

    logger.info("loading training numpy & random states...")
    assets_dir = params.get('LOAD_MODEL') + '/assets/'
    if os.path.exists(assets_dir):
        numpy_states = os.path.join(assets_dir, 'np_states.pkl')
        if os.path.isfile(numpy_states):
            with codecs.open(numpy_states, 'rb') as fh:
                np.random.set_state(pickle.load(fh))

        random_states = os.path.join(assets_dir, 'random_states.pkl')
        if os.path.isfile(random_states):
            with codecs.open(random_states, 'rb') as fh:
                random.setstate(pickle.load(fh))
    elif params.get('SEED', None):
        seed = params.get('SEED', None)
        logger.info("fixing seed ({}) from given config...".format(seed))
        np.random.seed(seed)
        random.seed(seed)
    else:
        logger.info("No given seed, not previous state to reload, using random seed.")

    import tensorflow as tf
    tf.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    graph = tf.get_default_graph()

    model_path = params.get('LOAD_MODEL', None)
    if model_path:
        logger.info(("* Loading QE model and Flask starting server..."
            "please wait until server has fully started"))
        load_model(model_path)
        load_dataset()
        app.run()
    else:
        logger.error("No model path given... use the 'LOAD_MODEL' in config")
        sys.exit(2)


if __name__ == "__main__":
    # decreasing Tensorflow verbosity
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from deepquest.utils.logs import logger_setup
    logger, logging = logger_setup('server')

    import plac
    plac.call(main)

