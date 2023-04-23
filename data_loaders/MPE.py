import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import random
import json
import copy
import csv
import re

from transformers import BertTokenizer

class MPELoader:
    '''

    '''
    def __init__(self, filepath, tokenizer, is_test=False, shuffle=False, max_seq_len=512,
                 mode:str="sigmoid", is_token_type_ids: bool=True, ids_to_exclude=None):
        # predictor (json format)
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.shuffle = shuffle
        self.max_seq_len = max_seq_len
        self.is_token_type_ids = is_token_type_ids
        self.mode = mode

        self.ids_to_exclude = ids_to_exclude
        self.exclusion_counter = 0
        self.dataset_counter = 0

        self.data = None
        #{"label": "C", "text": "the desert"}, {"label": "D", "text": "apartment"}, {"label": "E", "text": "roadblock"}]
        self.idx = [] # each id will be a string, e.g., 3d0f8824ea83ddcc9ab03055658b89d3
        self.gold_label = []
        self.question = []
        self.premise = [] # sentence1
        self.hypothesis = [] # sentence2

        self.processed_data = []
        counter = 0
        with open(filepath, "r") as f:
            data_reader = csv.DictReader(f, delimiter="\t")
            for line in data_reader:
                if mode == "softmax":
                    line_ = self._process_line(line)
                    if line_ is None: continue
                    self.processed_data.append(line_)
                elif mode == "sigmoid":
                    line_entailment, line_neutral, line_contradiction = self._process_line(line)
                #print(line_)
                    if line_entailment is None: continue
                    self.processed_data.append(line_entailment)
                    self.processed_data.append(line_neutral)
                    self.processed_data.append(line_contradiction)

        if self.ids_to_exclude is not None:
            print(f"exclusion_counter: {self.exclusion_counter}\ntotal_number_of_examples: {self.dataset_counter}")

        # the input format is: Premise [SEP] Hypothesis
        print(f"len(processed_data): {len(self.processed_data)}")

    def _process_line(self, line):
        #{'ID': '1001', 'premise1': '4615316828.jpg#1/Very young girl holding a spoon in a restaurant setting.',
        #'premise2': '4615316828.jpg#2/A child holding a white, plastic spoon.',
        #'premise3': '4615316828.jpg#3/A child smiling as he holds his spoon.',
        #'premise4': '4615316828.jpg#4/A toddler plays with his food.',
        #'hypothesis': 'Toddler sitting.', 'entailment_judgments': '2', 'neutral_judgments': '0',
        #'contradiction_judgments': '3', 'gold_label': 'entailment'}
        self.dataset_counter += 1
        if self.mode == "softmax":
            if self.ids_to_exclude is not None:
                if int(line["ID"]) in self.ids_to_exclude:
                    self.exclusion_counter += 1
                    return None
            tmp_dict = {}
            tmp_dict["idx"] = line["ID"]
            tmp_dict["premise1"] = re.sub("\d*\.jpg#\d\/", "", line["premise1"])
            tmp_dict["premise2"] = re.sub("\d*\.jpg#\d\/", "", line["premise2"])
            tmp_dict["premise3"] = re.sub("\d*\.jpg#\d\/", "", line["premise3"])
            tmp_dict["premise4"] = re.sub("\d*\.jpg#\d\/", "", line["premise4"])
            tmp_dict["hypothesis"] = line["hypothesis"]
            tmp_dict["gold_label"] = line["gold_label"]
            return tmp_dict
        elif self.mode == "sigmoid":
            if self.ids_to_exclude is not None:
                if int(line["ID"]) in self.ids_to_exclude:
                    self.exclusion_counter += 1
                    return None, None, None
            tmp_dict = {}
            tmp_dict["idx"] = line["ID"]
            tmp_dict["premise1"] = re.sub("\d*\.jpg#\d\/", "", line["premise1"])
            tmp_dict["premise2"] = re.sub("\d*\.jpg#\d\/", "", line["premise2"])
            tmp_dict["premise3"] = re.sub("\d*\.jpg#\d\/", "", line["premise3"])
            tmp_dict["premise4"] = re.sub("\d*\.jpg#\d\/", "", line["premise4"])
            tmp_dict["hypothesis"] = line["hypothesis"]

            tmp_dict_entailment = copy.deepcopy(tmp_dict)
            tmp_dict_neutral = copy.deepcopy(tmp_dict)
            tmp_dict_contradiction = copy.deepcopy(tmp_dict)

            tmp_dict_entailment["answer"] = "entailment"
            tmp_dict_entailment["label"] = 1 if line["gold_label"] == "entailment" else 0

            tmp_dict_neutral["answer"] = "neutral"
            tmp_dict_neutral["label"] = 1 if line["gold_label"] == "neutral" else 0

            tmp_dict_contradiction["answer"] = "contradiction"
            tmp_dict_contradiction["label"] = 1 if line["gold_label"] == "contradiction" else 0

            return tmp_dict_entailment, tmp_dict_neutral, tmp_dict_contradiction


    def _label_to_int(self, label: str):
        if label == "entailment": return [0,0,1] #return 2
        elif label == "neutral": return [0,1,0] #return 1
        elif label == "contradiction": return [1,0,0] #return 0
        else: raise Exception(f"Invalid label: {label}!")

    def _shuffle(self):
        random.shuffle(self.processed_data)

    def __call__(self, batch_size):
        if self.is_test:
            assert self.shuffle is False, f"Shuffling is strictly prohibited during test mode!"
        if self.shuffle: self._shuffle()
        #[label, idx, stem, choice, text]

        batch_counter = 0
        batch_input = []  # [[senta, sentb], [senta, sentb]]
        idx_list = []
        labels = []
        for i, example in enumerate(self.processed_data):

            if self.mode == "softmax":
                sentence_a, sentence_b = example["premise1"]+" "+example["premise2"]+" "+example["premise3"]+" "+\
                                         example["premise4"], example["hypothesis"]
            elif self.mode == "sigmoid":
                if self.is_test: assert batch_size % 3 == 0, f"For MPE there is a restriction on the batch size during " \
                                                             f"test mode; it must be a multiple of 3. This is because each " \
                                                             f"example is split into three (one for each answer option) and " \
                                                             f"to generate an answer (during test mode) we need all 3 examples" \
                                                             f" together in a batch, directly next to each other (also " \
                                                             f"without being shuffled)."
                if self.is_test: assert self.shuffle is False, f"Shuffling the data during test model is strictly prohibited!"
                sentence_a, sentence_b = example["premise1"]+" "+example["premise2"]+" "+example["premise3"]+ " " + \
                                         example["premise4"], example["hypothesis"]+" [SEP] "+example["answer"]
            else: raise Exception(f"Invalid mode: {self.mode}")

            batch_input.append([sentence_a, sentence_b])
            idx_list.append(str(example["idx"])) # each item in the list will be a list of two numbers. [exidx, qidx]
            batch_counter += 1
            if not self.is_test:
                if self.mode == "softmax":
                    labels.append(self._label_to_int(example["gold_label"]))
                elif self.mode == "sigmoid":
                    labels.append([example["label"]])
                else: raise Exception(f"Invalid mode: {self.mode}")
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
    dataLoader = MPELoader(filepath="/large_data/MPE/MultiPremiseEntailment-master/data/MPE/mpe_train.txt",
                               tokenizer=tokenizer,
                               is_test=is_test, mode="sigmoid")

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
