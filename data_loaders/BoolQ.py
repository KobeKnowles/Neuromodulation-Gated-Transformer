import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import random
import json

from transformers import BertTokenizer

class BoolQLoader:
    '''

    '''
    def __init__(self, filepath, tokenizer, is_test=False, shuffle=False, max_seq_len=512, is_token_type_ids: bool=True,
                 ids_to_exclude=None):

        self.filepath = filepath
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.shuffle = shuffle
        self.max_seq_len = max_seq_len
        self.is_token_type_ids = is_token_type_ids

        self.data = None
        self.passage = []
        self.question = []
        self.idx = []
        self.label = [] if not is_test else None


        with open(filepath, "r") as f:
            self.data = [json.loads(line) for line in f]


        #print("self.data: \n", self.data)

        exclusion_counter = 0
        dataset_original_size = len(self.data)
        for example in self.data:
            if ids_to_exclude is not None:
                if example["idx"] in ids_to_exclude:
                    exclusion_counter += 1
                    continue
            self.passage.append(example["passage"])
            self.question.append(example["question"])
            self.idx.append(example["idx"])
            if not is_test:
                self.label.append(example["label"])
                assert isinstance(example["label"], bool)

        if ids_to_exclude is not None:
            print(f"exclusion_counter: {exclusion_counter}\ntotal_number_of_examples: {dataset_original_size}")

        assert len(self.passage) == len(self.question)
        if not is_test: assert len(self.question) == len(self.label)

    def _shuffle(self):
        if not self.is_test:
            temp_list = list(zip(self.passage, self.question, self.label, self.idx))
            random.shuffle(temp_list)
            self.passage, self.question, self.label, self.idx = zip(*temp_list)
        else:
            temp_list = list(zip(self.passage, self.question, self.idx))
            random.shuffle(temp_list)
            self.passage, self.question, self.idx = zip(*temp_list)

    def __call__(self, batch_size):
        if self.shuffle: self._shuffle()

        #if is_test: self._call_is_test()
        #else: self._call_not_test()
        batch_counter = 0
        batch_input = []  #[[senta, sentb], [senta, sentb]]
        idx_list = []
        labels = []
        for i in range(len(self.passage)):

            #str_ = "[CLS] " + self.passage[i] + " [SEP] " + self.question[i] + " [SEP]"
            sentence_a, sentence_b = self.passage[i], self.question[i]
            batch_input.append([sentence_a, sentence_b])
            idx_list.append(int(self.idx[i]))
            batch_counter += 1
            if not self.is_test:
                #tmp_ = [0, 1] if int(self.label[i]) == 1  else [1, 0] # [False, True]; i.e., [0,1] is True while [1,0] is False.
                tmp_ = [int(self.label[i])]
                labels.append(tmp_)
            if batch_counter % batch_size == 0:
                tok_str_ = self.tokenizer.batch_encode_plus(batch_input,
                                                            add_special_tokens=True, padding="max_length",
                                                            max_length=self.max_seq_len, truncation="only_first")
                                                            #truncation=True)
                batch_counter = 0
                batch_input = []

                tok_str_["input_ids"] = tf.cast(tf.convert_to_tensor(tok_str_["input_ids"]), dtype=tf.dtypes.int32)
                tok_str_["attention_mask"] = tf.cast(tf.convert_to_tensor(tok_str_["attention_mask"]),
                                                     dtype=tf.dtypes.int32)
                if self.is_token_type_ids:
                    tok_str_["token_type_ids"] = tf.cast(tf.convert_to_tensor(tok_str_["token_type_ids"]),
                                                         dtype=tf.dtypes.int32)

                idx_list_ = tf.convert_to_tensor(idx_list, dtype=tf.dtypes.int32)
                idx_list = []

                if not self.is_test:
                    labels_ = labels # store the values of the labels before resetting in the next line.
                    labels = []
                    labels_ = tf.cast(tf.convert_to_tensor(labels_), dtype=tf.dtypes.int8)

                if self.is_test: yield idx_list_, tok_str_
                else: yield idx_list_, tok_str_, labels_

        # if there are still samples in batch_input then output them as follows; i.e., in a smaller batch size.
        if len(batch_input) != 0:
            tok_str_ = self.tokenizer.batch_encode_plus(batch_input,
                                                        add_special_tokens=True, padding="max_length",
                                                        max_length=self.max_seq_len, truncation="only_first")
                                                        #truncation=True)

            batch_input = [] # technically not needed here.

            tok_str_["input_ids"] = tf.cast(tf.convert_to_tensor(tok_str_["input_ids"]), dtype=tf.dtypes.int32)
            tok_str_["attention_mask"] = tf.cast(tf.convert_to_tensor(tok_str_["attention_mask"]),
                                                 dtype=tf.dtypes.int32)
            if self.is_token_type_ids:
                tok_str_["token_type_ids"] = tf.cast(tf.convert_to_tensor(tok_str_["token_type_ids"]),
                                                     dtype=tf.dtypes.int32)

            idx_list_ = tf.convert_to_tensor(idx_list, dtype=tf.dtypes.int32)
            idx_list = []

            if not self.is_test:
                labels_ = labels  # store the values of the labels before resetting in the next line.
                labels = []
                labels_ = tf.cast(tf.convert_to_tensor(labels_), dtype=tf.dtypes.int8)

            if self.is_test:
                yield idx_list_, tok_str_
            else:
                yield idx_list_, tok_str_, labels_

        if self.is_test: yield None, None
        else: yield None, None, None

if __name__ == "__main__":

    is_test = False
    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    dataLoader = BoolQLoader(filepath="/large_data/SuperGlue/BoolQ/train.jsonl",
                             tokenizer=tokenizer,
                             is_test=is_test)
    dataLoader.shuffle = True
    break_ = 100000000
    counter = 0
    if not is_test:
        for idx, example, label in dataLoader(batch_size=8):
            print(f"idx: {idx}\n"
                  f"example: {example}\n"
                  #f"example-input_ids: {example['input_ids']}\n"
                  f"example: {tokenizer.batch_decode(example['input_ids'])}\n"
                  f"label: {label}")
            if counter == break_: break
            counter += 1
    else:
        for idx, example in dataLoader(batch_size=8):
            print(f"idx: {idx}\n"
                  f"example: {example}\n"
                  f"example: {tokenizer.batch_decode(example['input_ids'])}\n")
            if counter == break_: break
            counter += 1
