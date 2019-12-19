import logging

from keras_wrapper.dataset import Dataset, saveDataset, loadDataset

# Do not do this at home.
# A bug in multimodal_keras_wrapper (see
# https://github.com/MarcBS/multimodal_keras_wrapper/pull/221)
# means that process of real-valued labels is incorrect.
# We patch the Dataset class here.
# This will not be necessary when multimodal_keras_wrapper is fixed.

def dataset_preprocessreal_fixed(filename):
    """
    A version of multimodal_keras_wrapper dataset.Dataset's
    processReal method that works with reals.
    """

    import codecs

    with codecs.open(filename, 'r', encoding="utf-8") as lines:
        labels = []
        for line in lines:
            labels.append(float(line))
        return labels

Dataset.preprocessReal = staticmethod(dataset_preprocessreal_fixed)


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def update_dataset_from_file(ds,
                             input_text_filename,
                             params,
                             splits=None,
                             output_text_filename=None,
                             remove_outputs=False,
                             compute_state_below=False,
                             recompute_references=False):
    """
    Updates the dataset instance from a text file according to the given params.
    Used for sampling

    :param ds: Dataset instance
    :param input_text_filename: Source language sentences
    :param params: Parameters for building the dataset
    :param splits: Splits to sample
    :param output_text_filename: Target language sentences
    :param remove_outputs: Remove outputs from dataset (if True, will ignore the output_text_filename parameter)
    :param compute_state_below: Compute state below input (shifted target text for professor teaching)
    :param recompute_references: Whether we should rebuild the references of the dataset or not.

    :return: Dataset object with the processed data
    """

    if splits is None:
        splits = ['val']

    if output_text_filename is None:
        recompute_references = False

    for split in splits:
        if split == 'train':
            output_type = params.get('OUTPUTS_TYPES_DATASET', ['dense-text'] if 'sparse' in params['LOSS'] else ['text'])[0]
        else:
            # Type of val/test outuput is always 'text' or 'dense-text'
            output_type = 'dense-text' if 'sparse' in params['LOSS'] else 'text'
        # if sampling: output_type = "real"

        if remove_outputs:
            ds.removeOutput(split,
                            id=params['OUTPUTS_IDS_DATASET'][0])
            recompute_references = False

        elif output_text_filename is not None:
            ds.setOutput(output_text_filename,
                         split,
                         type=output_type,
                         id=params['OUTPUTS_IDS_DATASET'][0],
                         tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                         build_vocabulary=False,
                         pad_on_batch=params.get('PAD_ON_BATCH', True),
                         fill=params.get('FILL', 'end'),
                         sample_weights=params.get('SAMPLE_WEIGHTS', True),
                         max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 100),
                         max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                         min_occ=params.get('MIN_OCCURRENCES_OUTPUT_VOCAB', 0),
                         bpe_codes=params.get('BPE_CODES_PATH', None),
                         label_smoothing=params.get('LABEL_SMOOTHING', 0.),
                         overwrite_split=True)

        # INPUT DATA
        ds.setInput(input_text_filename,
                    split,
                    type=params.get('INPUTS_TYPES_DATASET', ['text', 'text'])[0],
                    id=params['INPUTS_IDS_DATASET'][0],
                    tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                    build_vocabulary=False,
                    pad_on_batch=params.get('PAD_ON_BATCH', True),
                    fill=params.get('FILL', 'end'),
                    max_text_len=params.get('MAX_INPUT_TEXT_LEN', 100),
                    max_words=params.get('INPUT_VOCABULARY_SIZE', 0),
                    min_occ=params.get('MIN_OCCURRENCES_INPUT_VOCAB', 0),
                    bpe_codes=params.get('BPE_CODES_PATH', None),
                    overwrite_split=True)
        if compute_state_below and output_text_filename is not None:
            # INPUT DATA
            ds.setInput(output_text_filename,
                        split,
                        type=params.get('INPUTS_TYPES_DATASET', ['text', 'text'])[1],
                        id=params['INPUTS_IDS_DATASET'][1],
                        pad_on_batch=params.get('PAD_ON_BATCH', True),
                        tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                        build_vocabulary=False,
                        offset=1,
                        fill=params.get('FILL', 'end'),
                        max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 100),
                        max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                        min_occ=params.get('MIN_OCCURRENCES_OUTPUT_VOCAB', 0),
                        bpe_codes=params.get('BPE_CODES_PATH', None),
                        overwrite_split=True)


            # Avoid if sampling
            # """
            ds.setInput(output_text_filename,
                        split,
                        type='text',
                        id=params['INPUTS_IDS_DATASET'][2],
                        pad_on_batch=params.get('PAD_ON_BATCH', True),
                        tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                        build_vocabulary=False,
                        offset=-1,
                        fill=params.get('FILL', 'end'),
                        max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 100),
                        max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                        min_occ=params.get('MIN_OCCURRENCES_OUTPUT_VOCAB', 0),
                        bpe_codes=params.get('BPE_CODES_PATH', None),
                        overwrite_split=True)
            # """


        else:
            # ds.setInput(None,
            #             split,
            #             type='ghost',
            #             id=params['INPUTS_IDS_DATASET'][-1],
            #             required=False,
            #             overwrite_split=True)
            ds.setInput(output_text_filename,
                        split,
                        type='text',
                        id=params['INPUTS_IDS_DATASET'][1],
                        pad_on_batch=params.get('PAD_ON_BATCH', True),
                        tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                        build_vocabulary=False,
                        offset=1,
                        fill=params.get('FILL', 'end'),
                        max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 100),
                        max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                        min_occ=params.get('MIN_OCCURRENCES_OUTPUT_VOCAB', 0),
                        bpe_codes=params.get('BPE_CODES_PATH', None),
                        overwrite_split=True)

            ds.setInput(output_text_filename,
                        split,
                        type='text',
                        id=params['INPUTS_IDS_DATASET'][2],
                        pad_on_batch=params.get('PAD_ON_BATCH', True),
                        tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                        build_vocabulary=False,
                        offset=-1,
                        fill=params.get('FILL', 'end'),
                        max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 100),
                        max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                        min_occ=params.get('MIN_OCCURRENCES_OUTPUT_VOCAB', 0),
                        bpe_codes=params.get('BPE_CODES_PATH', None),
                        overwrite_split=True)

        if params['ALIGN_FROM_RAW']:
            ds.setRawInput(input_text_filename,
                           split,
                           type='file-name',
                           id='raw_' + params['INPUTS_IDS_DATASET'][0],
                           overwrite_split=True)

        # If we had multiple references per sentence
        if recompute_references:
            keep_n_captions(ds, repeat=1, n=1, set_names=params['EVAL_ON_SETS'])

    return ds


def build_dataset(params, vocabulary=dict(), vocabulary_len=dict()):
    """
    Builds (or loads) a Dataset instance.
    :param params: Parameters specifying Dataset options
    :return: Dataset object
    """

    if params['REBUILD_DATASET']:  # We build a new dataset instance
        if params['VERBOSE'] > 0:
            silence = False
            logger.info('Building ' + params['DATASET_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + ' dataset')
        else:
            silence = True

        base_path = params['DATA_ROOT_PATH']
        name = params['DATASET_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN']

        ds = Dataset(name, base_path, silence=silence)
        ds.vocabulary = vocabulary
        ds.vocabulary_len = vocabulary_len

        # OUTPUT DATA
        # Load the train, val and test splits of the target language sentences (outputs). The files include a sentence per line.

        if params['MODEL_TYPE']=='Predictor':
            if 'PRED_VOCAB' in params:
                ds.setOutput(base_path + '/' + params['TEXT_FILES']['train'] + params['TRG_LAN'],
                         'train',
                         type=params.get('OUTPUTS_TYPES_DATASET', ['dense-text'] if 'sparse' in params['LOSS'] else ['text'])[0],
                         id=params['OUTPUTS_IDS_DATASET'][0],
                         tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                         # if you want new vocabulary set build_vocabulary to True
                         build_vocabulary=params['OUTPUTS_IDS_DATASET'][0],
                         pad_on_batch=params.get('PAD_ON_BATCH', True),
                         sample_weights=params.get('SAMPLE_WEIGHTS', True),
                         fill=params.get('FILL', 'end'),
                         max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 70),
                         max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                         min_occ=params.get('MIN_OCCURRENCES_OUTPUT_VOCAB', 0),
                         bpe_codes=params.get('BPE_CODES_PATH', None),
                         label_smoothing=params.get('LABEL_SMOOTHING', 0.))
            else:
                ds.setOutput(base_path + '/' + params['TEXT_FILES']['train'] + params['TRG_LAN'],
                         'train',
                         type=params.get('OUTPUTS_TYPES_DATASET', ['dense-text'] if 'sparse' in params['LOSS'] else ['text'])[0],
                         id=params['OUTPUTS_IDS_DATASET'][0],
                         tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                         # if you want new vocabulary set build_vocabulary to True
                         build_vocabulary=True,
                         pad_on_batch=params.get('PAD_ON_BATCH', True),
                         sample_weights=params.get('SAMPLE_WEIGHTS', True),
                         fill=params.get('FILL', 'end'),
                         max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 70),
                         max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                         min_occ=params.get('MIN_OCCURRENCES_OUTPUT_VOCAB', 0),
                         bpe_codes=params.get('BPE_CODES_PATH', None),
                         label_smoothing=params.get('LABEL_SMOOTHING', 0.))


        elif params['MODEL_TYPE']=='EstimatorSent' or params['MODEL_TYPE']=='EncSent' or 'EstimatorDoc' in params['MODEL_TYPE'] or 'EncDoc' in params['MODEL_TYPE']:

            ds.setOutput(base_path + '/' + params['TEXT_FILES']['train'] + params['PRED_SCORE'],
                         'train',
                         type='real',
                         id=params['OUTPUTS_IDS_DATASET'][0],
                         tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                         build_vocabulary=False,
                         pad_on_batch=params.get('PAD_ON_BATCH', False),
                         sample_weights=params.get('SAMPLE_WEIGHTS', False),
                         fill=params.get('FILL', 'end'),
                         max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 70),
                         max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                         min_occ=params.get('MIN_OCCURRENCES_OUTPUT_VOCAB', 0),
                         bpe_codes=params.get('BPE_CODES_PATH', None),
                         label_smoothing=params.get('LABEL_SMOOTHING', 0.))

        elif params['MODEL_TYPE'] == 'EstimatorWord' or params['MODEL_TYPE'] == 'EncWord' or params['MODEL_TYPE'] == 'EncWordAtt' or params['MODEL_TYPE'] == 'EncPhraseAtt' or params['MODEL_TYPE'] == 'EstimatorPhrase':

            ds.setOutput(base_path + '/' + params['TEXT_FILES']['train'] + params['PRED_SCORE'],
                         'train',
                         type='text',
                         id=params['OUTPUTS_IDS_DATASET'][0],
                         tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                         build_vocabulary=True,
                         pad_on_batch=params.get('PAD_ON_BATCH', True),
                         sample_weights=params.get('SAMPLE_WEIGHTS', False),
                         fill=params.get('FILL', 'end'),
                         max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 70),
                         max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                         min_occ=params.get('MIN_OCCURRENCES_OUTPUT_VOCAB', 0),
                         bpe_codes=params.get('BPE_CODES_PATH', None),
                         label_smoothing=params.get('LABEL_SMOOTHING', 0.))


        if params.get('ALIGN_FROM_RAW', True) and not params.get('HOMOGENEOUS_BATCHES', False):
            ds.setRawOutput(base_path + '/' + params['TEXT_FILES']['train'] + params['TRG_LAN'],
                            'train',
                            type='file-name',
                            id='raw_' + params['OUTPUTS_IDS_DATASET'][0])

        val_test_list = params.get('EVAL_ON_SETS', ['val'])
        no_ref = params.get('NO_REF', False)
        # if no_ref:
        #     val_test_list = []
        for split in val_test_list:
            if params['TEXT_FILES'].get(split) is not None:
                if params['MODEL_TYPE'] == 'Predictor':

                    ds.setOutput(base_path + '/' + params['TEXT_FILES'][split] + params['TRG_LAN'],
                                 split,
                                 type='text',
                                 id=params['OUTPUTS_IDS_DATASET'][0],
                                 pad_on_batch=params.get('PAD_ON_BATCH', True),
                                 tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                                 sample_weights=params.get('SAMPLE_WEIGHTS', True),
                                 max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 70),
                                 max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                                 bpe_codes=params.get('BPE_CODES_PATH', None),
                                 label_smoothing=0.)

                elif params['MODEL_TYPE'] == 'EstimatorSent' or params['MODEL_TYPE'] == 'EncSent' or 'EstimatorDoc' in params['MODEL_TYPE'] or 'EncDoc' in params['MODEL_TYPE']:

                    ds.setOutput(base_path + '/' + params['TEXT_FILES'][split] + params['PRED_SCORE'],
                                 split,
                                 type='real',
                                 id=params['OUTPUTS_IDS_DATASET'][0],
                                 pad_on_batch=params.get('PAD_ON_BATCH', True),
                                 tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                                 sample_weights=params.get('SAMPLE_WEIGHTS', False),
                                 max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 70),
                                 max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                                 bpe_codes=params.get('BPE_CODES_PATH', None),
                                 label_smoothing=0.)

                elif params['MODEL_TYPE'] == 'EstimatorWord' or params['MODEL_TYPE'] == 'EncWord' or params['MODEL_TYPE'] == 'EncWordAtt' or params['MODEL_TYPE'] == 'EncPhraseAtt' or params['MODEL_TYPE'] == 'EstimatorPhrase':

                    ds.setOutput(base_path + '/' + params['TEXT_FILES'][split] + params['PRED_SCORE'],
                                 split,
                                 type='text',
                                 id=params['OUTPUTS_IDS_DATASET'][0],
                                 pad_on_batch=params.get('PAD_ON_BATCH', True),
                                 tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                                 sample_weights=params.get('SAMPLE_WEIGHTS', False),
                                 max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 70),
                                 max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                                 bpe_codes=params.get('BPE_CODES_PATH', None),
                                 label_smoothing=0.)


                if params.get('ALIGN_FROM_RAW', True) and not params.get('HOMOGENEOUS_BATCHES', False):
                    ds.setRawOutput(base_path + '/' + params['TEXT_FILES'][split] + params['TRG_LAN'],
                                    split,
                                    type='file-name',
                                    id='raw_' + params['OUTPUTS_IDS_DATASET'][0])

        # INPUT DATA
        # We must ensure that the 'train' split is the first (for building the vocabulary)
        max_src_in_len=params.get('MAX_SRC_INPUT_TEXT_LEN', None)
        if max_src_in_len == None:
            params['MAX_SRC_INPUT_TEXT_LEN'] = params['MAX_INPUT_TEXT_LEN']

        max_trg_in_len=params.get('MAX_TRG_INPUT_TEXT_LEN', None)
        if max_trg_in_len == None:
            params['MAX_TRG_INPUT_TEXT_LEN'] = params['MAX_INPUT_TEXT_LEN']
    
        data_type_src = params.get('INPUTS_TYPES_DATASET', ['text', 'text'])[0]
        data_type_trg = data_type_src

        if 'EstimatorDoc' in params['MODEL_TYPE'] or 'EncDoc' in params['MODEL_TYPE']:
            data_type_src = 'text'
            data_type_trg = 'text'
       

        # here we set to doc meaning just the 3d input
        if params['MODEL_TYPE'] == 'EstimatorPhrase' or params['MODEL_TYPE'] == 'EncPhraseAtt':
            data_type_trg = 'doc'



        ext = params['TRG_LAN']
        target_dict='target_text'

        #if params['MODEL_TYPE'] != 'Predictor':
        #    ext = 'mt'
        
        for split in ['train', 'val', 'test']:
            if params['TEXT_FILES'].get(split) is not None:
                if split == 'train':
                    build_vocabulary = True
                else:
                    build_vocabulary = False
                if 'PRED_VOCAB' in params:

                    ds.setInput(base_path + '/' + params['TEXT_FILES'][split] + params['SRC_LAN'],
                            split,
                            type=data_type_src,
                            id=params['INPUTS_IDS_DATASET'][0],
                            pad_on_batch=params.get('PAD_ON_BATCH', True),
                            tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                            build_vocabulary=params['INPUTS_IDS_DATASET'][0],
                            fill=params.get('FILL', 'end'),
                            max_text_len=params.get('MAX_SRC_INPUT_TEXT_LEN', 70),
                            max_words=params.get('INPUT_VOCABULARY_SIZE', 0),
                            min_occ=params.get('MIN_OCCURRENCES_INPUT_VOCAB', 0),
                            bpe_codes=params.get('BPE_CODES_PATH', None))
                else:

                    ds.setInput(base_path + '/' + params['TEXT_FILES'][split] + params['SRC_LAN'],
                            split,
                            type=data_type_src,
                            id=params['INPUTS_IDS_DATASET'][0],
                            pad_on_batch=params.get('PAD_ON_BATCH', True),
                            tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                            build_vocabulary=build_vocabulary,
                            fill=params.get('FILL', 'end'),
                            max_text_len=params.get('MAX_SRC_INPUT_TEXT_LEN', 70),
                            max_words=params.get('INPUT_VOCABULARY_SIZE', 0),
                            min_occ=params.get('MIN_OCCURRENCES_INPUT_VOCAB', 0),
                            bpe_codes=params.get('BPE_CODES_PATH', None))

                if len(params['INPUTS_IDS_DATASET']) == 2:
                    if 'PRED_VOCAB' not in params and 'train' in split:

                        ds.setInput(base_path + '/' + params['TEXT_FILES'][split] + ext,
                                    split,
                                    type=data_type_trg,
                                    id=params['INPUTS_IDS_DATASET'][1],
                                    required=False,
                                    tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                                    pad_on_batch=params.get('PAD_ON_BATCH', True),
                                    build_vocabulary=build_vocabulary,
                                    offset=0,
                                    fill=params.get('FILL', 'end'),
                                    max_text_len=params.get('MAX_TRG_INPUT_TEXT_LEN', 3),
                                    max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                                    bpe_codes=params.get('BPE_CODES_PATH', None))


                    else:
                        # ds.setInput(None,
                        #             split,
                        #             type='ghost',
                        #             id=params['INPUTS_IDS_DATASET'][-1],
                        #             required=False)

                        ds.setInput(base_path + '/' + params['TEXT_FILES'][split] + ext,
                                    split,
                                    type=data_type_trg,
                                    id=params['INPUTS_IDS_DATASET'][1],
                                    required=False,
                                    tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                                    pad_on_batch=params.get('PAD_ON_BATCH', True),
                                    build_vocabulary=target_dict,
                                    offset=0,
                                    fill=params.get('FILL', 'end'),
                                    max_text_len=params.get('MAX_TRG_INPUT_TEXT_LEN', 3),
                                    max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                                    bpe_codes=params.get('BPE_CODES_PATH', None))


                if len(params['INPUTS_IDS_DATASET']) > 2:
                    if 'PRED_VOCAB' not in params and 'train' in split:

                        ds.setInput(base_path + '/' + params['TEXT_FILES'][split] + ext,
                                    split,
                                    type=data_type_trg,
                                    id=params['INPUTS_IDS_DATASET'][1],
                                    required=False,
                                    tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                                    pad_on_batch=params.get('PAD_ON_BATCH', True),
                                    build_vocabulary=build_vocabulary,
                                    offset=1,
                                    fill=params.get('FILL', 'end'),
                                    max_text_len=params.get('MAX_TRG_INPUT_TEXT_LEN', 3),
                                    max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                                    bpe_codes=params.get('BPE_CODES_PATH', None))

                        ds.setInput(base_path + '/' + params['TEXT_FILES'][split] + ext,
                                    split,
                                    type=data_type_trg,
                                    id=params['INPUTS_IDS_DATASET'][2],
                                    required=False,
                                    tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                                    pad_on_batch=params.get('PAD_ON_BATCH', True),
                                    build_vocabulary=target_dict,
                                    offset=-1,
                                    fill=params.get('FILL', 'end'),
                                    max_text_len=params.get('MAX_TRG_INPUT_TEXT_LEN', 3),
                                    max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                                    bpe_codes=params.get('BPE_CODES_PATH', None))

                        ds.setInput(base_path + '/' + params['TEXT_FILES'][split] + ext,
                                    split,
                                    type=data_type_trg,
                                    id=params['INPUTS_IDS_DATASET'][3],
                                    required=False,
                                    tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                                    pad_on_batch=params.get('PAD_ON_BATCH', True),
                                    build_vocabulary=target_dict,
                                    offset=0,
                                    fill=params.get('FILL', 'end'),
                                    max_text_len=params.get('MAX_TRG_INPUT_TEXT_LEN', 3),
                                    max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                                    bpe_codes=params.get('BPE_CODES_PATH', None))
                        if params.get('TIE_EMBEDDINGS', False):
                            ds.merge_vocabularies([params['INPUTS_IDS_DATASET'][1], params['INPUTS_IDS_DATASET'][0]])

                    else:
                        # ds.setInput(None,
                        #             split,
                        #             type='ghost',
                        #             id=params['INPUTS_IDS_DATASET'][-1],
                        #             required=False)

                        ds.setInput(base_path + '/' + params['TEXT_FILES'][split] + ext,
                                    split,
                                    type=data_type_trg,
                                    id=params['INPUTS_IDS_DATASET'][1],
                                    required=False,
                                    tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                                    pad_on_batch=params.get('PAD_ON_BATCH', True),
                                    build_vocabulary=target_dict,
                                    offset=1,
                                    fill=params.get('FILL', 'end'),
                                    max_text_len=params.get('MAX_TRG_INPUT_TEXT_LEN', 3),
                                    max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                                    bpe_codes=params.get('BPE_CODES_PATH', None))

                        ds.setInput(base_path + '/' + params['TEXT_FILES'][split] + ext,
                                    split,
                                    type=data_type_trg,
                                    id=params['INPUTS_IDS_DATASET'][2],
                                    required=False,
                                    tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                                    pad_on_batch=params.get('PAD_ON_BATCH', True),
                                    build_vocabulary=target_dict,
                                    offset=-1,
                                    fill=params.get('FILL', 'end'),
                                    max_text_len=params.get('MAX_TRG_INPUT_TEXT_LEN', 3),
                                    max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                                    bpe_codes=params.get('BPE_CODES_PATH', None))

                        ds.setInput(base_path + '/' + params['TEXT_FILES'][split] + ext,
                                    split,
                                    type=data_type_trg,
                                    id=params['INPUTS_IDS_DATASET'][3],
                                    required=False,
                                    tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                                    pad_on_batch=params.get('PAD_ON_BATCH', True),
                                    build_vocabulary=target_dict,
                                    offset=0,
                                    fill=params.get('FILL', 'end'),
                                    max_text_len=params.get('MAX_TRG_INPUT_TEXT_LEN', 3),
                                    max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                                    bpe_codes=params.get('BPE_CODES_PATH', None))

                if params.get('ALIGN_FROM_RAW', True) and not params.get('HOMOGENEOUS_BATCHES', False):
                    ds.setRawInput(base_path + '/' + params['TEXT_FILES'][split] + params['SRC_LAN'],
                                   split,
                                   type='file-name',
                                   id='raw_' + params['INPUTS_IDS_DATASET'][0])
        if params.get('POS_UNK', False):
            if params.get('HEURISTIC', 0) > 0:
                ds.loadMapping(params['MAPPING'])

        # If we had multiple references per sentence
        if not params.get('NO_REF', False):
            keep_n_captions(ds, repeat=1, n=1, set_names=params['EVAL_ON_SETS'])

        # We have finished loading the dataset, now we can store it for using it in the future
        saveDataset(ds, params['DATASET_STORE_PATH'])

    else:
        # We can easily recover it with a single line
        ds = loadDataset(params['DATASET_STORE_PATH'] + '/Dataset_' + params['DATASET_NAME'] + '_' + params['SRC_LAN'] + params['TRG_LAN'] + '.pkl')

        # If we had multiple references per sentence
        keep_n_captions(ds, repeat=1, n=1, set_names=params['EVAL_ON_SETS'])

    return ds


def keep_n_captions(ds, repeat, n=1, set_names=None):
    """
    Keeps only n captions per image and stores the rest in dictionaries for a later evaluation
    :param ds: Dataset object
    :param repeat: Number of input samples per output
    :param n: Number of outputs to keep.
    :param set_names: Set name.
    :return:
    """

    if set_names is None:
        set_names = ['val', 'test']
    for s in set_names:
        logger.info('Keeping ' + str(n) + ' captions per input on the ' + str(s) + ' set.')

        ds.extra_variables[s] = dict()
        n_samples = getattr(ds, 'len_' + s)
        # Process inputs
        for id_in in ds.ids_inputs:
            new_X = []
            if id_in in ds.optional_inputs:
                try:
                    X = getattr(ds, 'X_' + s)
                    for i in range(0, n_samples, repeat):
                        for j in range(n):
                            new_X.append(X[id_in][i + j])
                    getattr(ds, 'X ' + s)[id_in] = new_X
                except Exception:
                    pass
            else:
                X = getattr(ds, 'X_' + s)
                for i in range(0, n_samples, repeat):
                    for j in range(n):
                        new_X.append(X[id_in][i + j])
                aux_list = getattr(ds, 'X_' + s)
                aux_list[id_in] = new_X
                setattr(ds, 'X_' + s, aux_list)
                del aux_list
        # Process outputs
        for id_out in ds.ids_outputs:
            new_Y = []
            Y = getattr(ds, 'Y_' + s)
            dict_Y = dict()
            count_samples = 0
            for i in range(0, n_samples, repeat):
                dict_Y[count_samples] = []
                for j in range(repeat):
                    if j < n:
                        new_Y.append(Y[id_out][i + j])
                    dict_Y[count_samples].append(Y[id_out][i + j])
                count_samples += 1

            aux_list = getattr(ds, 'Y_' + s)
            aux_list[id_out] = new_Y
            setattr(ds, 'Y_' + s, aux_list)
            del aux_list

            # store dictionary with img_pos -> [cap1, cap2, cap3, ..., capN]
            ds.extra_variables[s][id_out] = dict_Y

        new_len = len(new_Y)
        setattr(ds, 'len_' + s, new_len)

        logger.info('Samples reduced to ' + str(new_len) + ' in ' + s + ' set.')
