import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import random
import json
import copy

from transformers import BertTokenizer

class ReCoRDLoader:
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
        self.passage = [] # {"text":"", "entities":[]}
        self.text = [] # each entry is a string; each entry is from a different examples; order is important.
        self.entities = [] # [{"start"0:, "end":1},{"start"2:, "end":5}]
        self.qas = [] # [{"query":"", "answers":[{"start":0, "end":1, "text":"Morgan"},{...}], "idx":0}, # There can be multiple queries.
        self.idx = []

        self.ids_to_exclude = ids_to_exclude
        self.exclusion_counter = 0
        self.dataset_counter = 0

        with open(filepath, "r") as f: 
            self.data = [json.loads(line) for line in f]

        for example in self.data:
            self.passage.append(example["passage"])
            self.text.append(example["passage"]["text"])
            self.entities.append(example["passage"]["entities"])
            self.qas.append(example["qas"])
            self.idx.append(example["idx"])

        assert len(self.passage) == len(self.text)
        assert len(self.entities) == len(self.text)
        assert len(self.qas) == len(self.text)
        assert len(self.idx) == len(self.text)

        self.processed_data = self._process_data()
        #print(f"processed_data: {self.processed_data[0:4]}")

        if self.ids_to_exclude is not None:
            print(f"exclusion_counter: {self.exclusion_counter}\ntotal_number_of_examples: {self.dataset_counter}")

        print(f"len(processed_data): {len(self.processed_data)}")


    def _process_data(self):

        example = []
        for i, text in enumerate(self.text):

            tmp_dict = {} # keys = [text, query, qidx, answer, label]
            tmp_dict["text"] = text

            ent = self.entities[i] # will be a list of dictionaries.
            unique_ent = self._return_unique_entities(text, ent)

            for j, query in enumerate(self.qas[i]): # iterate through each question.
                self.dataset_counter += 1
                tmp_dict_q = copy.deepcopy(tmp_dict)
                tmp_dict_q["query"] = self.qas[i][j]["query"]
                tmp_dict_q["idx"] = [self.idx[i], self.qas[i][j]["idx"]] # this is the question idx second and example idx first.
                if self.ids_to_exclude is not None:
                    if self.qas[i][j]["idx"] in self.ids_to_exclude:
                        self.exclusion_counter += 1
                        continue

                ans = self.qas[i][j]["answers"]
                unique_ans = self._return_unique_answers(ans)
                #qidx = self.qas[i][j]["idx"]

                for entity in unique_ent:
                    tmp_dict_ent = copy.deepcopy(tmp_dict_q)
                    tmp_dict_ent["answer"] = entity
                    if not self.is_test:
                        tmp_dict_ent["label"] = 1 if entity in unique_ans else 0

                    example.append(tmp_dict_ent)
                    #print(f"tmp_dict_ent: {tmp_dict_ent}\n"
                    #      f"entities: {ent}\n"
                    #      f"unique entities: {unique_ent}\n"
                    #      f"unique answer: {unique_ans}\n")

        return example

    def _return_unique_entities(self, text, entities):

        assert isinstance(text, str), f"the text variable should be a string, got {type(text)} instead!"
        unique_ent = []
        for dict_ in entities:
            assert isinstance(dict_, dict), f"dict_ should be a dictionary, got {type(dict_)} instead!"

            candidate_ent = text[dict_["start"]:dict_["end"]+1]
            if candidate_ent not in unique_ent: unique_ent.append(candidate_ent)
        #print(unique_ent)
        return unique_ent

    def _return_unique_answers(self, ans):
        # ans = [{"start":0, "end":1, "text":"Morgan"},{...}]
        unique_ans = []
        for dict_ in ans:
            if dict_["text"] not in unique_ans: unique_ans.append(dict_["text"])
        return unique_ans

    def _shuffle(self):
        random.shuffle(self.processed_data)

    def __call__(self, batch_size):
        if self.is_test: assert self.shuffle is False
        if self.shuffle: self._shuffle()
        #[text, query, qidx, answer, label]

        # if is_test: self._call_is_test()
        # else: self._call_not_test()
        batch_counter = 0
        batch_input = []  # [[senta, sentb], [senta, sentb]]
        idx_list = []
        labels = []
        for i, example in enumerate(self.processed_data):

            # str_ = "[CLS] " + self.passage[i] + " [SEP] " + self.question[i] + " [SEP]"
            # [text, query, idx, answer, label]
            sentence_a, sentence_b = example["text"], example["query"] + " [SEP] " + example["answer"]

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
    dataLoader = ReCoRDLoader(filepath="/large_data/SuperGlue/ReCoRD/val.jsonl",
                              tokenizer=tokenizer,
                              is_test=is_test)

    '''
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
    '''