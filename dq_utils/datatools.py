import os
from shutil import copyfile

def preprocessDoc(params):
    """
    Preprocesses document level data for compatability with keras_wrapper by concatenating document sentences and padding.

    the param['TEXT_FILES'] dict will be updated so that the file location corresponding to each split is updated to point to the preprocessed files.

    the params dict() contains other keys that are used by this function.

    :return: params with text file names ('TEXT_FILES') modified to point to the preprocessed files.
    """

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
