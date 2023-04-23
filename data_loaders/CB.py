import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import random
import json
import copy

from transformers import BertTokenizer

class CBLoader:
    '''

    '''
    def __init__(self, filepath, tokenizer, is_test=False, shuffle=False, max_seq_len=512, mode:str="softmax",
                 is_token_type_ids: bool=True, ids_to_exclude=None):

        self.filepath = filepath
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.shuffle = shuffle
        self.max_seq_len = max_seq_len
        self.is_token_type_ids = is_token_type_ids

        self.data = None
        self.premise = []
        self.hypothesis = []
        self.idx = []
        self.label = [] if not is_test else None

        self.mode = mode.lower()

        with open(filepath, "r") as f:
            self.data = [json.loads(line) for line in f]

        exclusion_counter = 0
        dataset_original_size = len(self.data)
        for example in self.data:
            if ids_to_exclude is not None:
                if example["idx"] in ids_to_exclude:
                    exclusion_counter += 1
                    continue

            self.premise.append(example["premise"])
            self.hypothesis.append(example["hypothesis"])
            self.idx.append(example["idx"])
            if not is_test:
                self.label.append(example["label"]) # entailment|contradiction|neutral.
                # note: keep using a single value and pass each label as a separate example.
                # pre-process all inputs before hand so shuffling labels withing an example can take place.

        if ids_to_exclude is not None:
            print(f"exclusion_counter: {exclusion_counter}\ntotal_number_of_examples: {dataset_original_size}")

        assert len(self.premise) == len(self.hypothesis)
        if not is_test: assert len(self.hypothesis) == len(self.label)

        self.processed_data = self._process_data()
        #print(f"processed_data: {self.processed_data}")


    def _process_data(self):

        #answer_types = ["entailment","contradiction","neutral"]

        example = []
        for i, premise in enumerate(self.premise):

            tmp_dict = {}
            tmp_dict["premise"] = premise
            tmp_dict["hypothesis"] = self.hypothesis[i]
            tmp_dict["idx"] = self.idx[i]
            if self.mode == "softmax":
                if not self.is_test: tmp_dict["label"] = self.label[i]
                example.append(tmp_dict)
            elif self.mode == "sigmoid":
                for lbl in ["contradiction", "neutral", "entailment"]:
                    tmp_dict2 = copy.deepcopy(tmp_dict)
                    tmp_dict2["answer"] = lbl
                    if not self.is_test:
                        tmp_dict2["label"] = 1 if self.label[i] == lbl else 0
                    example.append(tmp_dict2)
            else: raise Exception(f"Invalid mode: {self.mode}")
        return example

    def _shuffle(self):
        random.shuffle(self.processed_data)

    def _label_to_int(self, label: str):
        if label == "entailment": return [0,0,1] #return 2
        elif label == "neutral": return [0,1,0] #return 1
        elif label == "contradiction": return [1,0,0] #return 0
        else: raise Exception(f"Invalid label: {label}!")

    def __call__(self, batch_size):

        if self.shuffle: self._shuffle()

        # if is_test: self._call_is_test()
        # else: self._call_not_test()
        batch_counter = 0
        batch_input = []  # [[senta, sentb], [senta, sentb]]
        idx_list = []
        labels = []
        for i, example in enumerate(self.processed_data):

            # str_ = "[CLS] " + self.passage[i] + " [SEP] " + self.question[i] + " [SEP]"
            if self.mode == "softmax":
                sentence_a, sentence_b = example["premise"], example["hypothesis"]
            elif self.mode == "sigmoid":
                if self.is_test: assert batch_size % 3 == 0, f"For CB there is a restriction on the batch size during " \
                                                             f"test mode; it must be a multiple of 3. This is because each " \
                                                             f"example is split into three (one for each answer option) and " \
                                                             f"to generate an answer (during test mode) we need all 3 examples" \
                                                             f" together in a batch, directly next to each other (also " \
                                                             f"without being shuffled)."
                if self.is_test: assert self.shuffle is False, f"Shuffling the data during test model is strictly prohibited!"
                sentence_a, sentence_b = example["premise"], example["hypothesis"] + " [SEP] " + example["answer"]
            else: raise Exception(f"Invalid mode: {self.mode}")

            batch_input.append([sentence_a, sentence_b])
            idx_list.append(example["idx"])
            batch_counter += 1
            if not self.is_test:
                if self.mode == "softmax":
                    tmp_ = self._label_to_int(example["label"])
                elif self.mode == "sigmoid":
                    tmp_ = [example["label"]]
                else: raise Exception(f"Invalid mode: {self.mode}")
                labels.append(tmp_)
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

        if self.is_test:
            yield None, None
        else:
            yield None, None, None

if __name__ == "__main__":

    is_test = False
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    dataLoader = CBLoader(filepath="/large_data/SuperGlue/CB/train.jsonl",
                             tokenizer=tokenizer,
                             is_test=is_test, mode="Sigmoid")
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
