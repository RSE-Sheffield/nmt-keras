#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# dataset.py
#
# Copyright (C) 2020 Frederic Blain (feedoo) <f.blain@sheffield.ac.uk>
#
# Licensed under the "THE BEER-WARE LICENSE" (Revision 42):
# Fred (feedoo) Blain wrote this file. As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a tomato juice or coffee in return
#

"""

"""

import os
import sys
import logging

if sys.version_info.major == 3:
    import _pickle as pk
else:
    import cPickle as pk
    from itertools import izip as zip
import codecs
import numpy as np

from keras_wrapper.extra.tokenizers import *
from keras_wrapper.dataset import Dataset as Keras_dataset
from deepquest.data_engine.tokenizers import FullTokenizer

import tensorflow as tf
import tensorflow_hub as hub


logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%d/%m/%Y %H:%M:%S')
logger = logging.getLogger(__name__)


class Dataset(Keras_dataset):

    def __init__(self, name, path, pad_symbol='<pad>', unk_symbol='<unk>', null_symbol='<null>', silence=False):
        # new Dataset attributes w.r.t BERT
        self.bert_hub_module_handle = None
        self.bert_tokenizer = None
        self.bert_vocab_file = None
        self.bert_tokenizer_built = False

        super().__init__(name, path, pad_symbol, unk_symbol, null_symbol, silence)


    def build_bert_tokenizer(self, bert_hub_module_handle):
        """
        Constructs a BERT tokenizer instance.
        :return: None
        """
        with tf.Graph().as_default():
            bert_module = hub.Module(bert_hub_module_handle)
            tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
            with tf.Session() as sess:
                vocab_file, do_lower_case = sess.run([
                    tokenization_info["vocab_file"],
                    tokenization_info["do_lower_case"]]
                    )

        bert_tokenizer = FullTokenizer(
                vocab_file=vocab_file,
                do_lower_case=do_lower_case
                )
        self.bert_tokenizer = bert_tokenizer
        self.bert_vocab_file = vocab_file
        self.bert_tokenizer_built = True


    def tokenize_bert(self, sentence, max_text_len):
        """
        This function is similar to `convert_sentences_to_features`
        :param sentence: sentence to tokenieize
        :param tokenizer: BERT Tokenizer object
        :param max_text_len: max length of sentence *once* tokenized
        :return: tokens, BERT token IDs, mask and segment IDs
        """
        if not self.bert_tokenizer_built:
            self.build_bert_tokenizer(bert_hub_module_handle)
        tokenizer = self.bert_tokenizer

        tokens = ['[CLS]']
        tokens.extend(tokenizer.tokenize(sentence))
        if len(tokens) > max_text_len-1:
            tokens = tokens[:max_text_len-1]
        tokens.append('[SEP]')

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)

        #Zero Mask till seq_length
        zero_mask = [0] * (max_text_len-len(tokens))
        input_ids.extend(zero_mask)
        input_mask.extend(zero_mask)
        segment_ids.extend(zero_mask)

        # loadText in keras_wrapper requires string, not list
        # tokens = ' '.join(tokens)
        input_ids =  ' '.join(str(x) for x in input_ids)
        input_mask =  ' '.join(str(x) for x in input_mask)
        segment_ids =  ' '.join(str(x) for x in segment_ids)

        return input_ids, input_mask, segment_ids


    def setInput(self, path_list, set_name, type='raw-image', id='image', repeat_set=1, required=True,
                 overwrite_split=False, normalization_types=None, data_augmentation_types=None,
                 add_additional=False,
                 img_size=None, img_size_crop=None, use_RGB=True,
                 # 'raw-image' / 'video'   (height, width, depth)
                 max_text_len=35, tokenization='tokenize_none', offset=0, fill='end', min_occ=0,  # 'text'
                 pad_on_batch=True, build_vocabulary=False, max_words=0, words_so_far=False,  # 'text'
                 bpe_codes=None, separator='@@', use_unk_class=False,  # 'text'
                 bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1', # multilingual cased BERT model by default
                 feat_len=1024,  # 'image-features' / 'video-features'
                 max_video_len=26,  # 'video'
                 sparse=False,  # 'binary'
                 ):
        """
        Loads a list which can contain all samples from either the 'train', 'val', or
        'test' set splits (specified by set_name).

        # General parameters

        :param use_RGB:
        :param path_list: can either be a path to a text file containing the
                          paths to the images or a python list of paths
        :param set_name: identifier of the set split loaded ('train', 'val' or 'test')
        :param type: identifier of the type of input we are loading
                     (accepted types can be seen in self.__accepted_types_inputs)
        :param id: identifier of the input data loaded
        :param repeat_set: repeats the inputs given (useful when we have more outputs than inputs).
                           Int or array of ints.
        :param required: flag for optional inputs
        :param overwrite_split: indicates that we want to overwrite the data with
                                id that was already declared in the dataset
        :param normalization_types: type of normalization applied to the current input
                                    if we activate the data normalization while loading
        :param data_augmentation_types: type of data augmentation applied to the current
                                        input if we activate the data augmentation while loading
        :param add_additional: adds additional data to an already existent input ID


        # 'raw-image'-related parameters

        :param img_size: size of the input images (any input image will be resized to this)
        :param img_size_crop: size of the cropped zone (when dataAugmentation=False the central crop will be used)


        # 'text'-related parameters

        :param tokenization: type of tokenization applied (must be declared as a method of this class)
                             (only applicable when type=='text').
        :param build_vocabulary: whether a new vocabulary will be built from the loaded data or not
                                 (only applicable when type=='text'). A previously calculated vocabulary will be used
                                 if build_vocabulary is an 'id' from a previously loaded input/output
        :param max_text_len: maximum text length, the rest of the data will be padded with 0s
                            (only applicable if the output data is of type 'text').
        :param max_words: a maximum of 'max_words' words from the whole vocabulary will
                          be chosen by number or occurrences
        :param offset: number of timesteps that the text is shifted to the right
                      (for sequential conditional models, which take as input the previous output)
        :param fill: select whether padding before or after the sequence
        :param min_occ: minimum number of occurrences allowed for the words in the vocabulary. (default = 0)
        :param pad_on_batch: the batch timesteps size will be set to the length of the largest sample +1 if
                            True, max_len will be used as the fixed length otherwise
        :param words_so_far: if True, each sample will be represented as the complete set of words until the point
                            defined by the timestep dimension (e.g. t=0 'a', t=1 'a dog', t=2 'a dog is', etc.)
        :param bpe_codes: Codes used for applying BPE encoding.
        :param separator: BPE encoding separator.

        # 'image-features' and 'video-features'- related parameters

        :param feat_len: size of the feature vectors for each dimension.
                         We must provide a list if the features are not vectors.


        # 'video'-related parameters
        :param max_video_len: maximum video length, the rest of the data will be padded with 0s
                              (only applicable if the input data is of type 'video' or video-features').
        """
        self.__checkSetName(set_name)
        if img_size is None:
            img_size = [256, 256, 3]

        if img_size_crop is None:
            img_size_crop = [227, 227, 3]
        # Insert type and id of input data
        keys_X_set = list(getattr(self, 'X_' + set_name))
        if id not in self.ids_inputs:
            self.ids_inputs.append(id)
        elif id in keys_X_set and not overwrite_split and not add_additional:
            raise Exception('An input with id "' + id + '" is already loaded into the Database.')

        if not required and id not in self.optional_inputs:
            self.optional_inputs.append(id)

        if type not in self.__accepted_types_inputs:
            raise NotImplementedError('The input type "' + type +
                                      '" is not implemented. '
                                      'The list of valid types are the following: ' + str(self.__accepted_types_inputs))
        if self.types_inputs.get(set_name) is None:
            self.types_inputs[set_name] = [type]
        else:
            self.types_inputs[set_name].append(type)

        # Preprocess the input data depending on its type
        if type == 'raw-image':
            data = self.preprocessImages(path_list, id, set_name, img_size, img_size_crop, use_RGB)
        elif type == 'video':
            data = self.preprocessVideos(path_list, id, set_name, max_video_len, img_size, img_size_crop)
        elif type == 'text' or type == 'dense-text':
            if self.max_text_len.get(id) is None:
                self.max_text_len[id] = dict()
            data = self.preprocessText(path_list, id, set_name, tokenization, build_vocabulary, max_text_len,
                                       max_words, offset, fill, min_occ, pad_on_batch, words_so_far,
                                       bpe_codes=bpe_codes, separator=separator, use_unk_class=use_unk_class, bert_hub_module_handle=bert_hub_module_handle)
        elif type == 'text-features':
            if self.max_text_len.get(id) is None:
                self.max_text_len[id] = dict()
            data = self.preprocessTextFeatures(path_list, id, set_name, tokenization, build_vocabulary, max_text_len,
                                               max_words, offset, fill, min_occ, pad_on_batch, words_so_far,
                                               bpe_codes=bpe_codes, separator=separator, use_unk_class=use_unk_class)
        elif type == 'image-features':
            data = self.preprocessFeatures(path_list, id, set_name, feat_len)
        elif type == 'video-features':
            # Check if the chosen data augmentation types exists
            if data_augmentation_types is not None:
                for da in data_augmentation_types:
                    if da not in self.__available_augm_vid_feat:
                        raise NotImplementedError(
                            'The chosen data augmentation type ' + da +
                            ' is not implemented for the type "video-features".')
            self.inputs_data_augmentation_types[id] = data_augmentation_types
            data = self.preprocessVideoFeatures(path_list, id, set_name, max_video_len, img_size, img_size_crop, feat_len)
        elif type == 'categorical':
            if build_vocabulary:
                self.setClasses(path_list, id)
            data = self.preprocessCategorical(path_list, id)
        elif type == 'categorical_raw':
            data = self.preprocessIDs(path_list, id, set_name)
        elif type == 'binary':
            data = self.preprocessBinary(path_list, id, sparse)
        elif type == 'id':
            data = self.preprocessIDs(path_list, id, set_name)
        elif type == 'ghost':
            data = []

        if isinstance(repeat_set, (np.ndarray, np.generic, list)) or repeat_set > 1:
            data = list(np.repeat(data, repeat_set))

        if 'bert' in tokenization.lower():
            # data[0]: tokens, data[1]: mask, data[2]: segids
            self.__setInput(data[0], set_name, type, id, overwrite_split, add_additional)

            # if we use BERT, we create new inputs in the Dataset artificially
            # for both the mask and segids required by the BERT Layer
            self.__setInput(data[1], set_name, type, id + '_mask', overwrite_split, add_additional)
            if id + '_mask' not in self.ids_inputs:
                self.ids_inputs.append(id + '_mask')
            self.types_inputs[set_name].append(type)
            # if not required and (id + '_mask') not in self.optional_inputs:
            self.optional_inputs.append(id + '_mask')

            self.__setInput(data[2], set_name, type, id + '_segids', overwrite_split, add_additional)
            if id + '_segids' not in self.ids_inputs:
                self.ids_inputs.append(id + '_segids')
            self.types_inputs[set_name].append(type)
            # if not required and (id + '_segids') not in self.optional_inputs:
            self.optional_inputs.append(id + '_segids')

        else:
            self.__setInput(data, set_name, type, id, overwrite_split, add_additional)


    def preprocessText(self, annotations_list, data_id, set_name, tokenization, build_vocabulary, max_text_len,
                       max_words, offset, fill, min_occ, pad_on_batch, words_so_far,
                       bert_hub_module_handle='https://tfhub.dev/google/bert_multi_cased_L-12_H-768_A-12/1',
                       bpe_codes=None, separator='@@', use_unk_class=False,
                       ):
        """
        Preprocess 'text' data type: Builds vocabulary (if necessary) and preprocesses the sentences.
        Also sets Dataset parameters.

        :param annotations_list: Path to the sentences to process.
        :param data_id: Dataset id of the data.
        :param set_name: Name of the current set ('train', 'val', 'test')
        :param tokenization: Tokenization to perform.
        :param build_vocabulary: Whether we should build a vocabulary for this text or not.
        :param max_text_len: Maximum length of the text. If max_text_len == 0, we treat the full sentence as a class.
        :param max_words: Maximum number of words to include in the dictionary.
        :param offset: Text shifting.
        :param fill: Whether we path with zeros at the beginning or at the end of the sentences.
        :param min_occ: Minimum occurrences of each word to be included in the dictionary.
        :param pad_on_batch: Whether we get sentences with length of the maximum length of the
                             minibatch or sentences with a fixed (max_text_length) length.
        :param words_so_far: Experimental feature. Should be ignored.
        :param bpe_codes: Codes used for applying BPE encoding.
        :param separator: BPE encoding separator.
        :param use_unk_class: Add a special class for the unknown word when maxt_text_len == 0.

        :return: Preprocessed sentences.
        """
        sentences = []
        if isinstance(annotations_list, str) and os.path.isfile(annotations_list):
            with codecs.open(annotations_list, 'r', encoding='utf-8') as list_:
                for line in list_:
                    sentences.append(line.rstrip('\n'))
        elif isinstance(annotations_list, list):
            sentences = annotations_list
        else:
            raise Exception(
                'Wrong type for "annotations_list". '
                'It must be a path to a text file with the sentences or a list of sentences. '
                'It currently is: %s' % (str(annotations_list)))

        # Tokenize sentences
        if max_text_len != 0:  # will only tokenize if we are not using the whole sentence as a class
            # Check if tokenization method exists
            if hasattr(self, tokenization):
                if 'bpe' in tokenization.lower():
                    if bpe_codes is None:
                        raise AssertionError('bpe_codes must be specified when applying a BPE tokenization.')
                    self.build_bpe(bpe_codes, separator=separator)
                if 'bert' in tokenization.lower():
                    self.build_bert_tokenizer(bert_hub_module_handle)
                tokfun = eval('self.' + tokenization)
                if not self.silence:
                    logger.info('\tApplying tokenization function: "' + tokenization + '".')
            else:
                raise Exception('Tokenization procedure "' + tokenization + '" is not implemented.')

            if 'bert' in tokenization.lower():
                # the next two are for BERT
                sentences_mask =  [None] * len(sentences)
                sentences_segids = [None] * len(sentences)
                for i, sentence in enumerate(sentences):
                    input_ids, input_mask, segment_ids = tokfun(sentence, max_text_len)
                    sentences[i] = input_ids
                    sentences_mask[i] = input_mask
                    sentences_segids[i] = segment_ids

                # free memory from the BERT tokenizer
                del self.bert_tokenizer
            else:
                for i, sentence in enumerate(sentences):
                    sentences[i] = tokfun(sentence)
        else:
            tokfun = None

        # Build vocabulary
        if isinstance(build_vocabulary, str):
            if build_vocabulary in self.vocabulary:
                self.vocabulary[data_id] = self.vocabulary[build_vocabulary]
                self.vocabulary_len[data_id] = self.vocabulary_len[build_vocabulary]
                if not self.silence:
                    logger.info('\tReusing vocabulary named "' + build_vocabulary + '" for data with data_id "' + data_id + '".')
            else:
                raise Exception('The parameter "build_vocabulary" must be a boolean '
                                'or a str containing an data_id of the vocabulary we want to copy.\n'
                                'It currently is: %s' % str(build_vocabulary))

        elif isinstance(build_vocabulary, dict):
            self.vocabulary[data_id] = build_vocabulary
            if not self.silence:
                logger.info('\tReusing vocabulary from dictionary for data with data_id "' + data_id + '".')

        elif build_vocabulary:
            if 'bert' in tokenization.lower():
                # initialise with an empty dict
                self.vocabulary[data_id] = {}

                vocab_dict = {'<unk>': 0}
                with codecs.open(self.bert_vocab_file, 'r', 'utf-8') as fh:
                  idx = 0
                  for line in fh:
                    tok = line.rstrip()
                    vocab_dict[tok] = idx
                    idx += 1
                inv_vocab_dict = {v: k for k, v in vocab_dict.items()}

                #tokids
                self.vocabulary[data_id]['words2idx'] = vocab_dict
                self.vocabulary[data_id]['idx2words'] = inv_vocab_dict
                self.vocabulary_len[data_id] = len(vocab_dict.keys())

                # mask
                self.vocabulary[data_id + "_mask"] = {}
                self.vocabulary[data_id + "_mask"]['words2idx'] = {u'<unk>': 0, u'0': 0, u'1': 1}
                self.vocabulary[data_id + "_mask"]['idx2words'] = {0: u'0', 1: u'1'}
                self.vocabulary_len[data_id + "_mask"] = 3

                # segids
                self.vocabulary[data_id + "_segids"] = {}
                self.vocabulary[data_id + "_segids"]['words2idx'] = {u'<unk>': 0, u'0': 0, u'1': 1}
                self.vocabulary[data_id + "_segids"]['idx2words'] = {0: u'0', 1: u'1'}
                self.vocabulary_len[data_id + "_segids"] = 3

            else:
                self.build_vocabulary(sentences, data_id,
                        max_text_len != 0,
                        min_occ=min_occ,
                        n_words=max_words,
                        use_extra_words=(max_text_len != 0),
                        use_unk_class=use_unk_class)

        if data_id not in self.vocabulary:
            raise Exception('The dataset must include a vocabulary with data_id "' + data_id +
                            '" in order to process the type "text" data. Set "build_vocabulary" to True if you want to use the current data for building the vocabulary.')

        # Store max text len
        self.max_text_len[data_id][set_name] = max_text_len
        self.text_offset[data_id] = offset
        self.fill_text[data_id] = fill
        self.pad_on_batch[data_id] = pad_on_batch
        self.words_so_far[data_id] = words_so_far

        if 'bert' in tokenization.lower():
            # if you use BERT, we create the corresponding attributes artificially
            if self.max_text_len.get(data_id + '_mask') is None:
                self.max_text_len[data_id + '_mask'] = dict()
            self.max_text_len[data_id + '_mask'][set_name] = max_text_len
            self.text_offset[data_id + '_mask'] = offset
            self.fill_text[data_id + '_mask'] = fill
            self.pad_on_batch[data_id + '_mask'] = pad_on_batch
            self.words_so_far[data_id + '_mask'] = words_so_far

            if self.max_text_len.get(data_id + '_segids') is None:
                self.max_text_len[data_id + '_segids'] = dict()
            self.max_text_len[data_id + '_segids'][set_name] = max_text_len
            self.text_offset[data_id + '_segids'] = offset
            self.fill_text[data_id + '_segids'] = fill
            self.pad_on_batch[data_id + '_segids'] = pad_on_batch
            self.words_so_far[data_id + '_segids'] = words_so_far

            return (sentences, sentences_mask, sentences_segids)
        else:
            return sentences

