import logging

from deepquest.data_engine.dataset import Dataset
from keras_wrapper.dataset import saveDataset, loadDataset

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)

def preprocessDoc(params):
    """
    Preprocesses document level data for compatability with keras_wrapper by concatenating document sentences and padding.

    the param['TEXT_FILES'] dict will be updated so that the file location corresponding to each split is updated to point to the preprocessed files.

    the params dict() contains other keys that are used by this function.

    :return: params with text file names ('TEXT_FILES') modified to point to the preprocessed files.
    """

    import os
    from shutil import copyfile

    doc_size = params['SECOND_DIM_SIZE']

    for split in ['train', 'val', 'test']:
        for ext in [params['SRC_LAN'], params['TRG_LAN'], params['PRED_SCORE']]:
            if ext == params['PRED_SCORE'] and split != 'test':
                scores_file = params['DATA_ROOT_PATH'] + '/' + params['TEXT_FILES'][split] + ext
                filename, file_extension = os.path.splitext(scores_file)
                write_path = filename + '_docProcess' + file_extension
                copyfile(scores_file, write_path)
            elif ext == params['PRED_SCORE'] and split == 'test' and not params['NO_REF']:
                scores_file = params['DATA_ROOT_PATH'] + '/' + params['TEXT_FILES'][split] + ext
                filename, file_extension = os.path.splitext(scores_file)
                write_path = filename + '_docProcess' + file_extension
                copyfile(scores_file, write_path)
            elif ext != params['PRED_SCORE']:
                if not params['TEXT_FILES'].get(split):
                    continue
                annotations_list = os.path.join(params['DATA_ROOT_PATH'], params['TEXT_FILES'][split] + ext)

                sentences = []
                sentences2d = []
                if isinstance(annotations_list, str) and os.path.isfile(annotations_list):
                    with open(annotations_list, 'r') as list_:
                        sentences_doc = []
                        sent_counter = 0
                        for line in list_:
                            new_line = line.rstrip('\n')
                            if new_line == '#doc#':
                                for i in range(sent_counter,doc_size):
                                    sentences_doc.append('<pad> <pad> <pad>')
                                sentences2d.append(sentences_doc)
                                sentences_doc = []
                                sent_counter = 0
                            else:
                                if sent_counter < doc_size:
                                    sentences.append(new_line)
                                    sentences_doc.append(new_line)
                                    sent_counter += 1
                elif isinstance(annotations_list, list):
                    sentences = annotations_list
                else:
                    raise Exception(
                        'Wrong type for "annotations_list". It must be a path to a text file with the sentences or a list of sentences. '
                        'It currently is: %s' % (str(annotations_list)))

                filename, file_extension = os.path.splitext(annotations_list)
                write_path = filename + '_docProcess' + file_extension
                with open(write_path, "w") as f:
                    for lines in sentences2d: f.writelines(lines); f.write('\n')
        params['TEXT_FILES'].update({split: os.path.basename(filename) + '_docProcess.'})

    return params



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
                    max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 100),
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


        # INPUT DATA
        data_type_src = params.get('INPUTS_TYPES_DATASET', ['text', 'text'])[0]
        data_type_trg = data_type_src

        if 'estimatordoc' in params['MODEL_TYPE'].lower() or 'encdoc' in params['MODEL_TYPE'].lower():
            data_type_src = 'text'
            data_type_trg = 'text'


        # here we set to doc meaning just the 3d input
        if params['MODEL_TYPE'].lower() == 'estimatorphrase' or params['MODEL_TYPE'].lower() == 'encphraseatt':
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
                            max_text_len=params.get('MAX_INPUT_TEXT_LEN', 70),
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
                            max_text_len=params.get('MAX_INPUT_TEXT_LEN', 70),
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


        # OUTPUT DATA
        # Load the train, val and test splits of the target language sentences (outputs). The files include a sentence per line.

        if params['MODEL_TYPE'].lower()=='predictor':
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


        elif params['MODEL_TYPE'].lower()=='estimatorsent' or params['MODEL_TYPE'].lower()=='encsent' or 'estimatordoc' in params['MODEL_TYPE'].lower() or 'encdoc' in params['MODEL_TYPE'].lower():

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

        elif params['MODEL_TYPE'].lower() == 'estimatorword' or params['MODEL_TYPE'].lower() == 'encword' or params['MODEL_TYPE'].lower() == 'encwordatt' or params['MODEL_TYPE'].lower() == 'encphraseatt' or params['MODEL_TYPE'].lower() == 'estimatorphrase':
            # tok_method = params.get('TOKENIZATION_METHOD', 'tokenize_none'),
            # if 'bert' in params['TOKENIZATION_METHOD'].lower():
            #     tok_method = 'tokenize_none'

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
        # if no_ref:                              # removed as this seems to cause a bug in training ig no_ref=true
        #     val_test_list = []
        for split in val_test_list:
            if params['TEXT_FILES'].get(split) is not None:
                if params['MODEL_TYPE'].lower() == 'predictor':

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

                elif params['MODEL_TYPE'].lower() == 'estimatorsent' or params['MODEL_TYPE'].lower() == 'encsent' or 'estimatordoc' in params['MODEL_TYPE'].lower() or 'encdoc' in params['MODEL_TYPE'].lower():

                    ds.setOutput(base_path + '/' + params['TEXT_FILES'][split] + params['PRED_SCORE'],
                                 split,
                                 type='real',
                                 id=params['OUTPUTS_IDS_DATASET'][0],
                                 pad_on_batch=params.get('PAD_ON_BATCH', True),
                                 tokenization=params.get('TOKENIZATION_METHOD', 'tokenize_none'),                                 sample_weights=params.get('SAMPLE_WEIGHTS', False),
                                 max_text_len=params.get('MAX_OUTPUT_TEXT_LEN', 70),
             max_words=params.get('OUTPUT_VOCABULARY_SIZE', 0),
                                 bpe_codes=params.get('BPE_CODES_PATH', None),
                                 label_smoothing=0.)

                elif params['MODEL_TYPE'].lower() == 'estimatorword' or params['MODEL_TYPE'].lower() == 'encword' or params['MODEL_TYPE'].lower() == 'encwordatt' or params['MODEL_TYPE'].lower() == 'encphraseatt' or params['MODEL_TYPE'].lower() == 'estimatorphrase':
                    # tok_method = params.get('TOKENIZATION_METHOD', 'tokenize_none'),
                    # if 'bert' in params['TOKENIZATION_METHOD'].lower():
                    #     tok_method = 'tokenize_none'

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
