import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from os import listdir
from os.path import isfile, join

import tensorflow as tf
import numpy as np
import random
import json
import copy

from transformers import BertTokenizer

class RACELoader:
    '''

    '''
    def __init__(self, filepath_middle, filepath_high, tokenizer, is_test=False, shuffle=False, max_seq_len=512, mode:str="sigmoid",
                 is_token_type_ids: bool=True, get_pred: bool=False, ids_to_exclude=None):

        #print("ids to exclude", ids_to_exclude)

        self.filepath_middle = filepath_middle
        self.filepath_high = filepath_high
        assert self.filepath_middle != "" or self.filepath_high != "", f"One of filepath_middle or filepath_high must " \
                                                                        f"not be an empty string!"
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.shuffle = shuffle
        self.max_seq_len = max_seq_len
        self.is_token_type_ids = is_token_type_ids
        self.mode = mode
        assert mode.lower() != "softmax", f"The softmax function has not been implemented."

        self.data = None

        if self.filepath_middle != "":
            self.filenames_middle = [name for name in listdir(self.filepath_middle) if isfile(join(self.filepath_middle, name))]
        else: self.filenames_middle = None
        if self.filepath_high != "":
            self.filenames_high = [name for name in listdir(self.filepath_high) if isfile(join(self.filepath_high, name))]
        else: self.filenames_high = None

        if not get_pred:
            self.processed_data = self._process_data()
        else:
            self.processed_data = self._process_data_for_pred()

        if ids_to_exclude is not None:
            exclusion_counter = 0
            dataset_counter = len(self.processed_data)
            for i in range(len(self.processed_data)-1,-1,-1):
                old_idx = self.processed_data[i]["idx"]
                if self.processed_data[i]["idx"] in ids_to_exclude:
                    exclusion_counter += 1
                    popped = self.processed_data.pop(i)
                    assert popped["idx"] == old_idx

            print(f"exclusion_counter: {exclusion_counter/4}\ntotal_number_of_examples: {dataset_counter/4}")

        print(f"len(processed_data): {len(self.processed_data)}")

    def _process_data_for_pred(self):

        example = []
        unique_counter = 1
        if self.filenames_middle is not None:
            for filename in self.filenames_middle:
                idx, race_passage, answer_char, answer_options, passage_questions = self.process_file(
                    filepath=self.filepath_middle + filename)
                # idx is a string.
                # race_passage is a string.
                # answer_char = ["A", "C", "B"]
                # answer_options is a list of lists, with each containing answer options for a given question.
                # passage_questions is a list of strings.
                tmp_dict = {}
                tmp_dict["idx"] = idx
                tmp_dict["passage"] = race_passage
                assert len(answer_char) == len(passage_questions)
                assert len(answer_char) == len(answer_options)
                for j in range(len(answer_char)):
                    tmp_dict2 = copy.deepcopy(tmp_dict)
                    tmp_dict2["idx"] = tmp_dict2["idx"] + "---" + str(unique_counter)
                    unique_counter += 1
                    tmp_dict2["correct_label"] = answer_char[j]
                    example.append(tmp_dict2)

        if self.filenames_high is not None:
            for filename in self.filenames_high:
                idx, race_passage, answer_char, answer_options, passage_questions = self.process_file(
                    filepath=self.filepath_high + filename)
                # idx is a string.
                # race_passage is a string.
                # answer_char = ["A", "C", "B"] is a list of stings
                # answer_options is a list of lists, with each containing answer options for a given question.
                # passage_questions is a list of strings.
                tmp_dict = {}
                tmp_dict["idx"] = idx
                tmp_dict["passage"] = race_passage
                assert len(answer_char) == len(passage_questions)
                assert len(answer_char) == len(answer_options)
                for j in range(len(answer_char)):
                    tmp_dict2 = copy.deepcopy(tmp_dict)
                    tmp_dict2["idx"] = tmp_dict2["idx"] + "---" + str(unique_counter)
                    unique_counter += 1
                    tmp_dict2["correct_label"] = answer_char[j]
                    example.append(tmp_dict2)
        return example


    def _process_data(self):

        example = []
        unique_counter = 1
        if self.filenames_middle is not None:
            for filename in self.filenames_middle:
                idx, race_passage, answer_char, answer_options, passage_questions = self.process_file(filepath=self.filepath_middle+filename)
                # idx is a string.
                # race_passage is a string.
                # answer_char = ["A", "C", "B"]
                # answer_options is a list of lists, with each containing answer options for a given question.
                # passage_questions is a list of strings.
                tmp_dict = {}
                tmp_dict["idx"] = idx
                tmp_dict["passage"] = race_passage
                assert len(answer_char) == len(passage_questions)
                assert len(answer_char) == len(answer_options)
                for j in range(len(answer_char)):
                    tmp_dict2 = copy.deepcopy(tmp_dict)
                    tmp_dict2["idx"] = tmp_dict2["idx"] + "---" + str(unique_counter)
                    unique_counter += 1
                    tmp_dict2["correct_label"] = answer_char[j]
                    tmp_dict2["question"] = passage_questions[j]
                    for k in range(len(answer_options[j])):
                        tmp_dict3 = copy.deepcopy(tmp_dict2)
                        tmp_dict3["answer_option_text"] = answer_options[j][k]
                        if k == 0: tmp_dict3["answer_char"] = "A"
                        elif k == 1: tmp_dict3["answer_char"] = "B"
                        elif k == 2: tmp_dict3["answer_char"] = "C"
                        elif k == 3: tmp_dict3["answer_char"] = "D"
                        else: raise Exception(f"There is only four answer options in RACE; there is "
                                              f"more answer options")
                        #print(tmp_dict3)
                        example.append(tmp_dict3)

        if self.filenames_high is not None:
            for filename in self.filenames_high:
                idx, race_passage, answer_char, answer_options, passage_questions = self.process_file(filepath=self.filepath_high+filename)
                # idx is a string.
                # race_passage is a string.
                # answer_char = ["A", "C", "B"] is a list of stings
                # answer_options is a list of lists, with each containing answer options for a given question.
                # passage_questions is a list of strings.
                tmp_dict = {}
                tmp_dict["idx"] = idx
                tmp_dict["passage"] = race_passage
                assert len(answer_char) == len(passage_questions)
                assert len(answer_char) == len(answer_options)
                for j in range(len(answer_char)):
                    tmp_dict2 = copy.deepcopy(tmp_dict)
                    tmp_dict2["idx"] = tmp_dict2["idx"] + "---" + str(unique_counter)
                    unique_counter += 1
                    tmp_dict2["correct_label"] = answer_char[j]
                    tmp_dict2["question"] = passage_questions[j]
                    for k in range(len(answer_options[j])):
                        tmp_dict3 = copy.deepcopy(tmp_dict2)
                        tmp_dict3["answer_option_text"] = answer_options[j][k]
                        if k == 0: tmp_dict3["answer_char"] = "A"
                        elif k == 1: tmp_dict3["answer_char"] = "B"
                        elif k == 2: tmp_dict3["answer_char"] = "C"
                        elif k == 3: tmp_dict3["answer_char"] = "D"
                        else: raise Exception(f"There is only four answer options in RACE; there is "
                                              f"more answer options")
                        #print(tmp_dict3)
                        example.append(tmp_dict3)

        return example

    def process_file(self, filepath: str):

        data = None
        with open(filepath, "r") as f:
            data = json.load(f)

        idx = data["id"]
        race_passage = data["article"]
        answer_char = [ao for ao in data['answers']]  # correct answer option stored in a list.
        answer_options = [aos for aos in data['options']]  # stores a list of lists, of which contain answer options for a given question.
        passage_questions = [ques for ques in data['questions']]  # will be a list of strings...

        return idx, race_passage, answer_char, answer_options, passage_questions

    def _shuffle(self):
        random.shuffle(self.processed_data)

    def _label_to_int(self, label: str, choice: str):
        if label == choice: return 1
        else: return 0

    def __call__(self, batch_size):
        if self.is_test:
            assert self.shuffle is False, f"Shuffling is strictly prohibited during test mode!"
            assert batch_size % 4 == 0, f"The batch size should be a multiple of 4!"
        if self.shuffle: self._shuffle()
        #[label, idx, stem, choice, text]

        batch_counter = 0
        batch_input = []  # [[senta, sentb], [senta, sentb]]
        idx_list = []
        labels = []
        for i, example in enumerate(self.processed_data):

            sentence_a, sentence_b = example["passage"], example["question"]  + " [SEP] "  + example["answer_option_text"]

            batch_input.append([sentence_a, sentence_b])
            idx_list.append(example["idx"]) # each item in the list will be a list of two numbers. [exidx, qidx]
            batch_counter += 1
            if not self.is_test:
                labels.append([self._label_to_int(example["correct_label"], example["answer_char"])])
            if batch_counter % batch_size == 0:
                tok_str_ = self.tokenizer.batch_encode_plus(batch_input,
                                                            add_special_tokens=True, padding="max_length",
                                                            max_length=self.max_seq_len, truncation="only_first")
                # truncation=True)
                batch_counter = 0
                batch_input = []

                tok_str_["input_ids"] = tf.cast(tf.convert_to_tensor(tok_str_["input_ids"]), dtype=tf.dtypes.int32)
                tok_str_["attention_mask"] = tf.cast(tf.convert_to_tensor(tok_str_["attention_mask"]),
                                                     dtype=tf.dtypes.int32)
                if self.is_token_type_ids:
                    tok_str_["token_type_ids"] = tf.cast(tf.convert_to_tensor(tok_str_["token_type_ids"]),
                                                         dtype=tf.dtypes.int32)

                idx_list_ = tf.convert_to_tensor(idx_list, dtype=tf.dtypes.string)
                idx_list = []

                if not self.is_test:
                    labels_ = labels  # store the values of the labels before resetting in the next line.
                    labels = []
                    labels_ = tf.cast(tf.convert_to_tensor(labels_), dtype=tf.dtypes.int8)

                if self.is_test:
                    yield idx_list_, tok_str_
                else:
                    yield idx_list_, tok_str_, labels_

        # if there are still samples in batch_input then output them as follows; i.e., in a smaller batch size.
        if len(batch_input) != 0:
            tok_str_ = self.tokenizer.batch_encode_plus(batch_input,
                                                        add_special_tokens=True, padding="max_length",
                                                        max_length=self.max_seq_len, truncation="only_first")
            # truncation=True)

            batch_input = []  # technically not needed here.

            tok_str_["input_ids"] = tf.cast(tf.convert_to_tensor(tok_str_["input_ids"]), dtype=tf.dtypes.int32)
            tok_str_["attention_mask"] = tf.cast(tf.convert_to_tensor(tok_str_["attention_mask"]),
                                                 dtype=tf.dtypes.int32)
            if self.is_token_type_ids:
                tok_str_["token_type_ids"] = tf.cast(tf.convert_to_tensor(tok_str_["token_type_ids"]),
                                                     dtype=tf.dtypes.int32)

            idx_list_ = tf.convert_to_tensor(idx_list, dtype=tf.dtypes.string)
            idx_list = []

            if not self.is_test:
                labels_ = labels  # store the values of the labels before resetting in the next line.
                labels = []
                labels_ = tf.cast(tf.convert_to_tensor(labels_), dtype=tf.dtypes.int8)

            if self.is_test:
                yield idx_list_, tok_str_
            else:
                yield idx_list_, tok_str_, labels_

        if self.is_test:
            yield None, None
        else:
            yield None, None, None

if __name__ == "__main__":

    is_test = False
    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    dataLoader = RACELoader(filepath_middle="/large_data/RACE/RACE/train/middle/",
                            #filepath_middle="",
                            #filepath_high="/large_data/RACE/RACE/train/high/",
                            filepath_high="",
                            tokenizer=tokenizer,
                            is_test=is_test)

    dataLoader.shuffle = False
    break_ = 100000000
    counter = 0
    if not is_test:
        for idx, example, label in dataLoader(batch_size=4):
            print(f"idx: {idx}\n"
                  f"example: {example}\n"
                  #f"example-input_ids: {example['input_ids']}\n"
                  f"example: {tokenizer.batch_decode(example['input_ids'])}\n"
                  f"label: {label}")
            if counter == break_: break
            counter += 1
    else:
        for idx, example in dataLoader(batch_size=4): 
            print(f"idx: {idx}\n"
                  f"example: {example}\n"
                  f"example: {tokenizer.batch_decode(example['input_ids'])}\n")
            if counter == break_: break
            counter += 1
