import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import random
import json
import copy

from transformers import BertTokenizer

class ReClorLoader:
    '''

    '''
    def __init__(self, filepath, tokenizer, is_test=False, shuffle=False, max_seq_len=512, mode:str="sigmoid", is_token_type_ids: bool=True,
                 get_pred: bool=False):

        self.filepath = filepath
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.shuffle = shuffle
        self.max_seq_len = max_seq_len
        self.is_token_type_ids = is_token_type_ids
        self.mode = mode
        assert mode.lower() != "softmax", f"The softmax function has not been implemented."

        self.data = None
        self.context = [] # [{"label": "A", "text": "race track"}, {"label": "B", "text": "populated areas"},
        #{"label": "C", "text": "the desert"}, {"label": "D", "text": "apartment"}, {"label": "E", "text": "roadblock"}]
        self.question = []
        self.idx = [] # each id will be a string, e.g., 3d0f8824ea83ddcc9ab03055658b89d3
        self.label = [] # ["A", "B", "A"]
        self.answers = []

        with open(filepath, "r") as f:
            self.data = json.load(f)

        for example in self.data:
            self.context.append(example["context"])
            self.question.append(example["question"])
            self.answers.append(example["answers"])
            self.label.append(example["label"])
            self.idx.append(example["id_string"])


        assert len(self.context) == len(self.question)
        assert len(self.context) == len(self.answers)
        assert len(self.context) == len(self.label)
        assert len(self.context) == len(self.idx)

        if not get_pred:
            self.processed_data = self._process_data()
        else:
            self.processed_data = self._process_predictions()
        print(f"len(processed_data): {len(self.processed_data)}")

    def _process_data(self):

        example = []
        for i, idx in enumerate(self.idx):

            tmp_dict = {}
            tmp_dict["idx"] = idx
            tmp_dict["context"] = self.context[i]
            tmp_dict["question"] = self.question[i]

            for j, ans in enumerate(self.answers[i]):
                tmp_dict2 = copy.deepcopy(tmp_dict)
                tmp_dict2["answer"] = ans
                tmp_dict2["label"] = 1 if j == self.label[i] else 0
                example.append(tmp_dict2)

        return example

    def _process_predictions(self):

        example = []
        for i, idx in enumerate(self.idx):

            tmp_dict = {}
            tmp_dict["idx"] = idx
            tmp_dict["label"] = self.label[i]
            example.append(tmp_dict)

        return example

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

            sentence_a, sentence_b = example["context"], example["question"] + " [SEP] " + example["answer"]

            batch_input.append([sentence_a, sentence_b])
            idx_list.append(example["idx"]) # each item in the list will be a list of two numbers. [exidx, qidx]
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
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    dataLoader = ReClorLoader(filepath="/large_data/ReClor/val.json",
                              tokenizer=tokenizer,
                              is_test=is_test)


    dataLoader.shuffle = False
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
