import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import random
import json
import copy

from transformers import BertTokenizer

class MultiRCLoader:
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

        '''
        {"idx": 0, 
        "version": 1.1, 
        "passage": {"text": ".... This requires only a very small force. ", 
                    "questions": [{"question": "... ?", 
                                       "answers": [{"text": "No", "idx": 0, "label": 0}, 
                                                   {"text": "Yes", "idx": 1, "label": 1}, 
                                                   {"text": "Less the mass, less the force applied", "idx": 2, "label": 1}, 
                                                   {"text": "It depends on the shape of the baseball", "idx": 3, "label": 0}], 
                                  "idx": 0}, 
                                  {"question": "What do you apply to an object to make it move or stop?", 
                                       "answers": [{"text": "Strength", "idx": 4, "label": 0}, 
                                                   {"text": "Nothing, it will stop on its own", "idx": 5, "label": 0}, 
                                                   {"text": "Apply force on the ball", "idx": 6, "label": 1}, 
                                                   {"text": "A force", "idx": 7, "label": 1}, 
                                                   {"text": "Pressure", "idx": 8, "label": 0}], 
                                  "idx": 1}, 
                                  {"question": "Does ... to it?", 
                                        "answers": [{"text": "How much an obj the objects mass", "idx": 9, "label": 0}, 
                                                    {"text": "No", "idx": 10, "label": 0}, 
                                                    {"text": "Motion changes on", "idx": 11, "label": 0}, 
                                                    {"text": "Yes", "idx": 12, "label": 1}], 
                                  "idx": 2}, 
                                  {"question": "What factors cause changes in motion of a moving object?",
                                        "answers": [{"text": "Shape of the object", "idx": 13, "label": 0}, 
                                                    {"text": "Mass of the object", "idx": 14, "label": 1}, 
                                                    {"text": "The object's mass", "idx": 15, "label": 0}, 
                                                    {"text": "The object's speed, direction", "idx": 16, "label": 1}, 
                                                    {"text": "Strength of the force applied", "idx": 17, "label": 1}, 
                                                    {"text": "The application of force", "idx": 18, "label": 1}, 
                                                    {"text": "Who is applying the force", "idx": 19, "label": 0}], 
                                  "idx": 3}]}}
        '''

        self.data = None
        self.passage = []
        self.idx = []

        self.ids_to_exclude = ids_to_exclude
        self.exclusion_counter = 0
        self.dataset_counter = 0

        with open(filepath, "r") as f: 
            self.data = [json.loads(line) for line in f]

        for example in self.data:
            self.passage.append(example["passage"])
            self.idx.append(example["idx"])



        assert len(self.passage) == len(self.idx)

        self.processed_data = self._process_data()
        #print(f"processed_data: {self.processed_data}")

        if self.ids_to_exclude is not None:
            print(f"exclusion_counter: {self.exclusion_counter}\ntotal_number_of_examples: {self.dataset_counter}")

        print(f"number of examples: {len(self.processed_data)}")


    def _process_data(self):

        # doing below is ok for this dataset as we predict for all answer, not choose the best out of the options.
        example = [] # {"passage":, "passage_idx":, "question":, "question_idx":, "answer":, "answer_idx":, "label":,}
        for i, passage in enumerate(self.passage):

            tmp_dict = {}
            tmp_dict["passage"] = passage["text"]
            tmp_dict["passage_idx"] = self.idx[i]

            for quest_dict in passage["questions"]:

                tmp_dict2 = copy.deepcopy(tmp_dict)

                tmp_dict2["question"] = quest_dict["question"]
                tmp_dict2["question_idx"] = quest_dict["idx"]

                for ans_dict in quest_dict["answers"]:
                    self.dataset_counter += 1
                    #text idx label

                    tmp_dict3 = copy.deepcopy(tmp_dict2)
                    if self.ids_to_exclude is not None:
                        if ans_dict["idx"] in self.ids_to_exclude:
                            self.exclusion_counter += 1
                            continue
                    tmp_dict3["answer"] = ans_dict["text"]
                    tmp_dict3["answer_idx"] = ans_dict["idx"]
                    if not self.is_test: tmp_dict3["label"] = ans_dict["label"]

                    example.append(tmp_dict3)

        return example

    def _shuffle(self):
        random.shuffle(self.processed_data)

    def _label_to_int(self, label: str):
        if label == 1: return [1] #return 1
        elif label == 0: return [0] #return 0
        else: raise Exception(f"Invalid label: {label}!")

    def __call__(self, batch_size):
        if self.is_test: assert self.shuffle is False, f"During test mode shuffle must be False."
        if self.shuffle: self._shuffle()

        batch_counter = 0
        batch_input = []  # [[senta, sentb], [senta, sentb]]
        idx_list = []
        labels = []
        for i, example in enumerate(self.processed_data):

            #{"passage":, "passage_idx":, "question":, "question_idx":, "answer":, "answer_idx":, "label":,}
            # str_ = "[CLS] " + self.passage[i] + " [SEP] " + self.question[i] + " [SEP]"
            sentence_a, sentence_b = example["passage"], example["question"] + " [SEP] " + example["answer"]
            batch_input.append([sentence_a, sentence_b])
            idx_list.append([example["passage_idx"], example["question_idx"], example["answer_idx"]])
            batch_counter += 1
            if not self.is_test:
                tmp_ = self._label_to_int(example["label"])
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

    is_test = True
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    dataLoader = MultiRCLoader(filepath="/large_data/SuperGlue/MultiRC/train.jsonl",
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
