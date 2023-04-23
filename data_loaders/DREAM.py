import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import random
import json
import copy
import csv

from transformers import BertTokenizer

class DREAMLoader:
    '''

    '''
    def __init__(self, filepath, tokenizer, is_test=False, shuffle=False, max_seq_len=512, is_token_type_ids: bool=True,
                 get_pred: bool=False):

        self.filepath = filepath
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.shuffle = shuffle
        self.max_seq_len = max_seq_len
        self.is_token_type_ids = is_token_type_ids

        #self.data = None
        self.context = []
        self.question = []
        self.choices = []
        self.answer = []
        self.idx = []

        self.semi_processed_data = []

        self.data = []
        with open(filepath, "r") as f:
            self.data = json.load(f)

        #print(self.data[0:5])
        for i, example in enumerate(self.data):
            #print(f"example[]: {example[1]}")
            tmp_dict = {}
            tmp_dict["context"] = " ".join(example[0])
            #tmp_dict["idx"] = example[-1]
            for j in range(len(example[1])): # -3 instead of -2 b/c not inclusive upper parameter.
                tmp_dict2 = copy.deepcopy(tmp_dict)

                tmp_dict2["question"] = example[1][j]["question"]
                tmp_dict2["choices"] = example[1][j]["choice"]
                tmp_dict2["answer"] = example[1][j]["answer"]
                tmp_dict2["idx"] = example[-1] + "-" +str(j)
                self.semi_processed_data.append(tmp_dict2)

            #self.question.append(example[1][0]["question"])
            #self.choices.append(example[1][0]["choice"]) # list of strings
            #self.answer.append(example[1][0]["answer"])
            #self.idx.append(example[2])

        #assert len(self.context) == len(self.question)
        #assert len(self.context) == len(self.choices)
        #assert len(self.context) == len(self.answer)
        #assert len(self.context) == len(self.idx)

        if not get_pred:
            self.processed_data = self._process_data()
        else:
            self.processed_data = self._process_predictions()
        #print(self.processed_data[-1])


    def _process_data(self):

        #answer_types = ["entailment","contradiction","neutral"]

        example = []
        for i, dict_ in enumerate(self.semi_processed_data):

            tmp_dict = {}
            tmp_dict["context"] = dict_["context"]
            tmp_dict["question"] = dict_["question"]
            tmp_dict["idx"] = dict_["idx"]
            is_true = False
            for j, choice in enumerate(dict_["choices"]):
                tmp_dict2 = copy.deepcopy(tmp_dict)
                tmp_dict2["choice"] = choice
                if not self.is_test:
                    tmp_dict2["label"] = 1 if choice == dict_["answer"] else 0
                    if choice == dict_["answer"]: is_true = True
                else: is_true = True
                if j >= 3: raise Exception(f"There is more than three answer options! We only support 3.")
                example.append(tmp_dict2)
            assert is_true, f"No example was set to True."

        return example

    def _process_predictions(self):

        #answer_types = ["entailment","contradiction","neutral"]

        example = []
        for i, dict_ in enumerate(self.semi_processed_data):

            tmp_dict = {}
            tmp_dict["idx"] = dict_["idx"]
            for j, choice in enumerate(dict_["choices"]):
                if choice == dict_["answer"]:
                    ans = None
                    if j == 0: ans = "A"
                    elif j == 1: ans = "B"
                    elif j == 2: ans = "C"
                    else: raise Exception(f"Invalid j value: {j}!\nj should not be greater than two!")
                    tmp_dict["label"] = ans
                    break # breaks the inner loop
                if j >= 3: raise Exception(f"There is more than three answer options! We only support 3.")
            example.append(tmp_dict)
        return example

    def _shuffle(self):
        random.shuffle(self.processed_data)

    def __call__(self, batch_size):
        if self.is_test:
            assert self.shuffle is False, f"Shuffling is strictly prohibited during test mode!"
            assert batch_size % 3 == 0, f"The batch size should be a multiple of 3!"
        if self.shuffle: self._shuffle()

        # if is_test: self._call_is_test()
        # else: self._call_not_test()
        batch_counter = 0
        batch_input = []  # [[senta, sentb], [senta, sentb]]
        idx_list = []
        labels = []
        for i, example in enumerate(self.processed_data):

            # str_ = "[CLS] " + self.passage[i] + " [SEP] " + self.question[i] + " [SEP]"
            sentence_a, sentence_b = example["context"], example["question"] + " [SEP] " + example["choice"]
            batch_input.append([sentence_a, sentence_b])
            idx_list.append(example["idx"])
            batch_counter += 1
            if not self.is_test:
                labels.append([example["label"]])
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
    dataLoader = DREAMLoader(filepath="/large_data/DREAM/dream/data/dev.json",
                             tokenizer=tokenizer,
                             is_test=is_test, get_pred=True)
    print(dataLoader.processed_data)
    '''
    dataLoader.shuffle = False
    break_ = 100000000
    counter = 0
    if not is_test:
        for idx, example, label in dataLoader(batch_size=3):
            print(f"idx: {idx}\n"
                  f"example: {example}\n"
                  #f"example-input_ids: {example['input_ids']}\n"
                  f"example: {tokenizer.batch_decode(example['input_ids'])}\n"
                  f"label: {label}")
            if counter == break_: break
            counter += 1
    else:
        for idx, example in dataLoader(batch_size=3):
            print(f"idx: {idx}\n"
                  f"example: {example}\n"
                  f"example: {tokenizer.batch_decode(example['input_ids'])}\n")
            if counter == break_: break
            counter += 1
    '''