import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import random
import json
import copy

from transformers import BertTokenizer

class SciTailLoader:
    '''

    '''
    def __init__(self, filepath, tokenizer, is_test=False, shuffle=False, max_seq_len=512, mode:str="sigmoid",
                 is_token_type_ids: bool=True, ids_to_exclude=None):
        # predictor (json format)
        self.filepath = filepath
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.shuffle = shuffle
        self.max_seq_len = max_seq_len
        self.is_token_type_ids = is_token_type_ids
        self.mode = mode
        assert mode.lower() != "softmax", f"The softmax function has not been implemented."

        self.data = None
        #{"label": "C", "text": "the desert"}, {"label": "D", "text": "apartment"}, {"label": "E", "text": "roadblock"}]
        self.idx = [] # each id will be a string, e.g., 3d0f8824ea83ddcc9ab03055658b89d3
        self.gold_label = []
        self.question = []
        self.premise = [] # sentence1
        self.hypothesis = [] # sentence2

        with open(filepath, "r") as f:
            self.data = [json.loads(line) for line in f]

        exclusion_counter = 0
        dataset_original_size = len(self.data)
        self.processed_data = []
        for i, example in enumerate(self.data):
            if ids_to_exclude is not None:
                if i in ids_to_exclude: # i is the idx for this dataset.
                    exclusion_counter += 1
                    continue

            tmp_dict = {}
            tmp_dict["idx"] = i
            tmp_dict["gold_label"] = example["gold_label"]
            assert example["gold_label"] in ["entails", "neutral"]
            tmp_dict["question"] = example["question"]
            tmp_dict["premise"] = example["sentence1"]
            tmp_dict["hypothesis"] = example["sentence2"]
            #print(tmp_dict)
            self.processed_data.append(tmp_dict)

        if ids_to_exclude is not None:
            print(f"exclusion_counter: {exclusion_counter}\ntotal_number_of_examples: {dataset_original_size}")

        # the input format is: Premise [SEP] Hypothesis

        #self.processed_data = self._process_data() # convert everything into a single dict.
        print(f"len(processed_data): {len(self.processed_data)}")


    #def _process_data(self):

        #example = []
        #for i, idx in enumerate(self.idx):
        #    tmp_dict = {}
        #    tmp_dict["idx"] = idx
        #    tmp_dict["gold_label"] = self.gold_label[i]
        #    tmp_dict["question"] = self.question[i]
        #    tmp_dict["premise"] = self.premise[i]
        #    tmp_dict["hypothesis"] = self.hypothesis[i]
        #    example.append(tmp_dict)
        #return example

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

            sentence_a, sentence_b = example["premise"], example["hypothesis"]#"(" + example["choice"] + ")" + example["text"]

            batch_input.append([sentence_a, sentence_b])
            idx_list.append(example["idx"]) # each item in the list will be a list of two numbers. [exidx, qidx]
            batch_counter += 1
            if not self.is_test:
                labels.append([1 if example["gold_label"] == "entails" else 0])
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

    is_test = True
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    dataLoader = SciTailLoader(filepath="/large_data/SciTail/SciTailV1.1/predictor_format/scitail_1.0_structure_dev.jsonl",
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