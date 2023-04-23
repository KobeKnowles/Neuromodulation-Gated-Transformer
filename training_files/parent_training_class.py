'''
name: Kobe Knowles;
date: 9th June, 2022;
filename: parent_training_class.py;
purpose: to create a general training class to be used by many datasets.
'''

import tensorflow as tf
from tensorflow.python.keras.utils.layer_utils import count_params
import numpy as np
import time
import json

class GeneralTrainingClass(object):
    '''

    :param object:
    :return:
    '''

    def __init__(self, model, tokenizer, optimizer, loss_object, loss_function, strategy, checkpoint_path_folder,
                 error_function:str="sigmoid", loss_object2=None):

        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.loss_object = loss_object
        self.loss_object2 = loss_object2
        self.loss_function = loss_function
        self.strategy = strategy
        self.checkpoint_path_folder = checkpoint_path_folder
        self.error_function = error_function.lower()

    def _train_step2(self, example, label, head_num):

        loss, size = 0, 0
        with tf.GradientTape() as tape:
            _, pred_logits = self.model(example, training=True, head_num=head_num)

            assert len(pred_logits.shape) == 2, f"The number of dimensions of pred_logits should be 2, got " \
                                                f"{len(pred_logits.shape)}!"

            tmp_loss_object = None
            #error_funct = None
            if pred_logits.shape[1] == 1:
                tmp_loss_object = self.loss_object # this should be compatible with the sigmoid function.
                pred_prob = tf.math.sigmoid(pred_logits)
                #error_funct = "sigmoid"
            elif pred_logits.shape[1] > 1:
                tmp_loss_object = self.loss_object2 # this should be for the softmax function.
                pred_prob = tf.nn.softmax(pred_logits, axis=-1)
                #error_funct = "softmax"
            else: raise Exception(f"Invalid shape of the second dimension, got {pred_logits.shape[1]}")


            #pred_logits.shape (batch_size, n)
            #if error_funct == "sigmoid":
            #    pred_prob = tf.math.sigmoid(pred_logits)
            #elif error_funct == "softmax":
            #    assert pred_logits.shape[1] > 1 # if softmax is used then the shape of the second dimension should be more than one.
            #   pred_prob = tf.nn.softmax(pred_logits, axis=-1)

            loss, size = self.loss_function(label, pred_prob, tmp_loss_object)
            loss_ = loss / size

        gradients = tape.gradient(loss_, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        loss = tf.convert_to_tensor([loss], dtype=tf.dtypes.float32)
        size = tf.convert_to_tensor([size], dtype=tf.dtypes.float32)

        return loss, size

    @tf.function
    def _distributed_train_step2(self, example, label, head_num):
        loss, size = None, None
        if self.strategy is not None:
            loss, size = self.strategy.run(self._train_step2, args=(example, label, head_num,))

            if self.strategy.num_replicas_in_sync > 1:
                loss = tf.reduce_sum(loss.values)
                size = tf.reduce_sum(size.values)
            else:
                loss = tf.reduce_sum(loss)
                size = tf.reduce_sum(size)
        else:
            loss, size = self._train_step2(example, label, head_num)
            loss = tf.reduce_sum(loss)
            size = tf.reduce_sum(size)

        return loss, size


    def _train_step(self, example, label):

        loss, size = 0, 0
        with tf.GradientTape() as tape:
            _, pred_logits = self.model(example, training=True)

            #pred_logits.shape (batch_size, n)
            if self.error_function == "sigmoid":
                pred_prob = tf.math.sigmoid(pred_logits)
            elif self.error_function == "softmax":
                assert pred_logits.shape[1] > 1 # if softmax is used then the shape of the second dimension should be more than one.
                pred_prob = tf.nn.softmax(pred_logits, axis=-1)
            else:
                raise Exception(f"Invalid error function: {self.error_function}")

            loss, size = self.loss_function(label, pred_prob, self.loss_object)
            loss_ = loss / size

        gradients = tape.gradient(loss_, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        loss = tf.convert_to_tensor([loss], dtype=tf.dtypes.float32)
        size = tf.convert_to_tensor([size], dtype=tf.dtypes.float32)

        return loss, size

    @tf.function
    def _distributed_train_step(self, example, label):
        loss, size = None, None
        if self.strategy is not None:
            loss, size = self.strategy.run(self._train_step, args=(example, label,))

            if self.strategy.num_replicas_in_sync > 1:
                loss = tf.reduce_sum(loss.values)
                size = tf.reduce_sum(size.values)
            else:
                loss = tf.reduce_sum(loss)
                size = tf.reduce_sum(size)
        else:
            loss, size = self._train_step(example, label)
            loss = tf.reduce_sum(loss)
            size = tf.reduce_sum(size)

        return loss, size

    def _create_dict_with_exp_setup(self, config_dict, num_epochs, avg_epoch_time_sec, dataset,
                                    tf_seed, random_seed, batch_size, num_GPUs, train_save_filepath,
                                    learning_rate_start_value, learning_rate_end_value, decay_steps, decay_function):
        exp_setup_dict = {}
        exp_setup_dict["trainable_params"] = count_params(self.model.trainable_weights)
        exp_setup_dict["non_trainable_params"] = count_params(self.model.non_trainable_weights)
        exp_setup_dict["config_params"] = config_dict
        exp_setup_dict["epochs"] = num_epochs
        exp_setup_dict["avg_epoch_time_sec"] = avg_epoch_time_sec
        exp_setup_dict["avg_epoch_time_minutes"] = avg_epoch_time_sec/60
        exp_setup_dict["avg_epoch_time_hours"] = avg_epoch_time_sec/(60*60)
        exp_setup_dict["dataset"] = dataset
        exp_setup_dict["tf_seed"] = tf_seed
        exp_setup_dict["random_seed"] = random_seed
        exp_setup_dict["max_seq_length"] = config_dict["max_seq_len"]
        exp_setup_dict["batch_size"] = batch_size
        exp_setup_dict["num_GPUs"] = num_GPUs
        exp_setup_dict["learning_rate_start_value"] = learning_rate_start_value
        exp_setup_dict["learning_rate_end_value"] = learning_rate_end_value
        exp_setup_dict["decay_steps"] = decay_steps
        exp_setup_dict["decay_function"] = decay_function

        with open(train_save_filepath+"training_setup.json", "w") as f:
            json.dump(exp_setup_dict, f)

    def _create_dict_with_exp_setup_iteration(self, config_dict, total_time, dataset, tf_seed, random_seed, batch_size,
                                              num_GPUs, train_save_filepath, learning_rate_start_value,
                                              learning_rate_end_value, learning_rate_linear_warmup_value,
                                              decay_steps, warmup_steps, decay_function, num_aux_toks,
                                              shuffle_datasets):
        exp_setup_dict = {}
        exp_setup_dict["trainable_params"] = count_params(self.model.trainable_weights)
        exp_setup_dict["non_trainable_params"] = count_params(self.model.non_trainable_weights)
        exp_setup_dict["config_params"] = config_dict

        exp_setup_dict["total_time_secs"] = total_time
        exp_setup_dict["total_time_minutes"] = total_time/60
        exp_setup_dict["total_time_hours"] = total_time/(60*60)
        exp_setup_dict["dataset"] = dataset
        exp_setup_dict["tf_seed"] = tf_seed
        exp_setup_dict["random_seed"] = random_seed
        exp_setup_dict["max_seq_length"] = config_dict["max_seq_len"]
        exp_setup_dict["batch_size"] = batch_size
        exp_setup_dict["num_GPUs"] = num_GPUs
        exp_setup_dict["learning_rate_start_value"] = learning_rate_start_value
        exp_setup_dict["learning_rate_end_value"] = learning_rate_end_value
        exp_setup_dict["learning_rate_linear_warmup_value"] = learning_rate_linear_warmup_value
        exp_setup_dict["decay_steps"] = decay_steps
        exp_setup_dict["warmup_steps"] = warmup_steps
        exp_setup_dict["decay_function"] = decay_function
        exp_setup_dict["num_aux_toks"] = num_aux_toks
        exp_setup_dict["shuffle_datasets"] = shuffle_datasets

        with open(train_save_filepath+"training_setup.json", "w") as f:
            json.dump(exp_setup_dict, f)


    def training_iteration(self, iterations: int, dataLoader, train_save_filepath: str, config_dict: dict,
                       dataset: str, tf_seed: int, random_seed: int, batch_size: int,
                       save_iterations: list=[100000,400000], print_every_iterations: int=100, num_GPUs: int=1,
                       learning_rate_start_value: int=0.0001, learning_rate_end_value: int=0,
                       learning_rate_linear_warmup_value=None, decay_steps=None, warmup_steps=None,
                       decay_function: str="cosine", num_aux_toks: int=0, shuffle_datasets: bool=True):

        if len(save_iterations) == 0: print(f"No models will be saved")

        start = time.time()

        iteration_loss_total = 0
        iteration_size_total = 0

        iteration_counter = 0
        for (input_ids, attention_mask, token_type_ids, label, head_num) in dataLoader: # the dataLoader should only run for a specified number of iterations, not one run through a, or multiple, datasets.
            example = {"input_ids":input_ids, "attention_mask":attention_mask, "token_type_ids":token_type_ids}
            iteration_counter += 1
            head_num = head_num.numpy().tolist()
            assert isinstance(head_num, int), f"Expected type int, got {type(head_num)}"
            if iteration_counter == 2: print(f"model.summary: {self.model.summary()}")
            loss_tmp, size_tmp = self._distributed_train_step2(example, label, head_num)

            if size_tmp == 0:
                print(f"The size is zero for the current batch; skipping to the next batch!")
                continue

            iteration_loss_total += loss_tmp
            iteration_size_total += size_tmp

            loss_ = loss_tmp / size_tmp
            loss_ = tf.cast(loss_, dtype=tf.dtypes.float32)

            if iteration_counter % print_every_iterations == 0:
                print(f"Iteration: {iteration_counter}\tLoss: {loss_:4f}")

            header = True if iteration_counter == 1 else False
            self._save_batch_loss_only("train", 1, iteration_counter, train_save_filepath, header, loss_)

            if iteration_counter in save_iterations:
                self.model.save_pretrained(self.checkpoint_path_folder+"iteration"+str(iteration_counter))

        total_time = time.time()-start
        print(f'Time taken for {iteration_counter} iterations: {total_time:.2f} secs\n')

        self._create_dict_with_exp_setup_iteration(config_dict=config_dict, total_time=total_time, dataset=dataset,
                                         tf_seed=tf_seed, random_seed=random_seed, batch_size=batch_size,
                                         num_GPUs=num_GPUs, train_save_filepath=train_save_filepath,
                                         learning_rate_start_value=learning_rate_start_value,
                                         learning_rate_end_value=learning_rate_end_value,
                                         learning_rate_linear_warmup_value=learning_rate_linear_warmup_value,
                                         decay_steps=decay_steps, warmup_steps=warmup_steps,
                                         decay_function=decay_function, num_aux_toks=num_aux_toks,
                                         shuffle_datasets=shuffle_datasets)


    def training_epoch(self, epoch_start: int, epoch_end: int, data_dict: {}, train_save_filepath: str, val_save_filepath: str,
                       config_dict: dict, dataset: str, tf_seed: int, random_seed: int, batch_size: int,
                       save_end_epoch: bool=True, print_every_iterations: int=100, num_GPUs: int=1,
                       learning_rate_start_value: int=0.0001, learning_rate_end_value: int=0,
                       decay_steps=None, decay_function="cosine"):

        if not save_end_epoch: print(f"Warning: save_end_epoch is False, the model's parameters will never be saved!")

        store_epoch_time = []

        for e in range(epoch_start, epoch_end):

            start = time.time()

            batch = 0
            epoch_loss_total = 0
            epoch_size_total = 0

            for (idx, input_ids, attention_mask, token_type_ids, label) in data_dict["train"]:
                example = {"input_ids":input_ids, "attention_mask":attention_mask, "token_type_ids":token_type_ids}
                batch += 1
                if e == epoch_start and batch == 2: print(f"model.summary: {self.model.summary()}")
                loss_tmp, size_tmp = self._distributed_train_step(example, label)

                if size_tmp == 0:
                    print(f"The size is zero for the current batch; skipping to the next batch!")
                    continue

                epoch_loss_total += loss_tmp
                epoch_size_total += size_tmp

                loss_ = loss_tmp / size_tmp
                loss_ = tf.cast(loss_, dtype=tf.dtypes.float32)

                if batch % print_every_iterations == 0:
                    print(f"Batch: {batch}\tEpoch: {e+1}\tLoss: {loss_:4f}")

                header = True if (batch == 1 and e == 0) else False
                self._save_batch_loss_only("train", e+1, batch, train_save_filepath, header, loss_)

            total_loss = epoch_loss_total / epoch_size_total

            header_ = True if e == 0 else False
            self._save_epoch_loss_only("train", e+1, train_save_filepath, header_, total_loss)

            epoch_time = time.time()-start
            store_epoch_time.append(epoch_time)
            print(f'Epoch: {e+1} Loss: {total_loss:.4f}')
            print(f'Time taken for epoch {e+1}: {epoch_time:.2f} secs\n')

            if save_end_epoch:
                self.model.save_pretrained(self.checkpoint_path_folder+"epoch"+str(e+1))

            if "val" in data_dict.keys():
                print(f"Running through the validation set now!")
                self.run_validation(e, val_save_filepath, data_dict["val"])

        # epoch_end - epoch_start (10-0; 0 to 9; 10 in total is correct)
        num_epochs = epoch_end - epoch_start
        avg_epoch_time_sec = sum(store_epoch_time)/num_epochs
        self._create_dict_with_exp_setup(config_dict=config_dict, num_epochs=num_epochs,
                                         avg_epoch_time_sec=avg_epoch_time_sec, dataset=dataset,
                                         tf_seed=tf_seed, random_seed=random_seed, batch_size=batch_size,
                                         num_GPUs=num_GPUs, train_save_filepath=train_save_filepath,
                                         learning_rate_start_value=learning_rate_start_value,
                                         learning_rate_end_value=learning_rate_end_value,
                                         decay_steps=decay_steps, decay_function=decay_function)

    def run_validation(self, e, save_filepath_val, data):
        start = time.time()
        epoch_loss_total = 0
        epoch_size_total = 0
        for (idx, input_ids, attention_mask, token_type_ids, label) in data:
            example = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

            loss_tmp, size_tmp = self._distributed_val_step(example, label)

            if size_tmp == 0:
                print(f"The size is zero for the current batch; skipping to the next batch!")
                continue

            epoch_loss_total += loss_tmp
            epoch_size_total += size_tmp

        total_loss = epoch_loss_total / epoch_size_total

        print(f'Validation Epoch: {e+1} Loss: {total_loss:.4f}')
        print(f'Time taken for validation epoch {e+1}: {time.time()-start:.2f} secs\n')

        header_ = True if e == 0 else False
        self._save_epoch_loss_only("val", e+1, save_filepath_val, header_, total_loss)

    def _val_step(self, example, label):

        loss, size = 0, 0
        with tf.GradientTape() as tape:
            _, pred_logits = self.model(example, training=False)

            #pred_logits.shape (batch_size, n)
            if self.error_function == "sigmoid":
                pred_prob = tf.math.sigmoid(pred_logits)
            elif self.error_function == "softmax":
                assert pred_logits.shape[1] > 1 # if softmax is used then the shape of the second dimension should be more than one.
                pred_prob = tf.nn.softmax(pred_logits, axis=-1)
            else: raise Exception(f"Invalid error function: {self.error_function}")

            loss, size = self.loss_function(label, pred_prob, self.loss_object)

        loss = tf.convert_to_tensor([loss], dtype=tf.dtypes.float32)
        size = tf.convert_to_tensor([size], dtype=tf.dtypes.float32)

        return loss, size

    @tf.function
    def _distributed_val_step(self, example, label):
        loss, size = None, None
        if self.strategy is not None:
            loss, size = self.strategy.run(self._val_step, args=(example, label,))

            if self.strategy.num_replicas_in_sync > 1:
                loss = tf.reduce_sum(loss.values)
                size = tf.reduce_sum(size.values)
            else:
                loss = tf.reduce_sum(loss)
                size = tf.reduce_sum(size)
        else:
            loss, size = self._val_step(example, label)
            loss = tf.reduce_sum(loss)
            size = tf.reduce_sum(size)

        return loss, size


    def _save_batch_loss_only(self, type_, epoch, batch, save_filepath, header, loss):

        assert isinstance(type_, str) and isinstance(save_filepath, str)
        file = save_filepath + type_ + "batch" + ".txt"
        with open(file, "a") as f:
            if header: f.write("Epoch Batch Loss \n")
            f.write(f"{epoch} {batch} {loss} \n")

    def _save_epoch_loss_only(self, type_, epoch, save_filepath, header, loss):

        assert isinstance(type_, str) and isinstance(save_filepath, str)
        file = save_filepath + type_ + "epoch" + ".txt"
        with open(file, "a") as f:
            if header: f.write("Epoch Loss \n")
            f.write(f"{epoch} {loss} \n")

    def _idx_example_test_helper(self, data, test_filepath, pred_mode="default"): # used for BoolQ
        # Each line of the prediction files should be a JSON entry, with an 'idx' field to
        # identify the example and a 'label' field with the prediction.
        if pred_mode == "default":
            for (idx, input_ids, attention_mask, token_type_ids) in data:
                example = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

                pred_bool = self.get_pred_batch_support_cls1(example)
                # [[True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True],
                # [True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True],
                # [True], [True], [True], [True], [True], [True], [True], [True]]

                idx_list = idx.numpy().tolist()

                for i, pred in enumerate(pred_bool): # arg will be an item in the list, an integer.
                    assert len(pred) == 1
                    with open(test_filepath, "a") as f:
                        line = {"idx":int(idx_list[i]), "label": "true" if pred[0] else "false"}
                        json.dump(line, f)
                        f.write('\n')
        elif pred_mode == "multiple_heads":
            for (idx, input_ids, attention_mask, token_type_ids, head_num) in data:
                example = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
                head_num = head_num.numpy().tolist()
                assert isinstance(head_num, int), f"Expected type int, got {type(head_num)}"

                pred_bool = self.get_pred_batch_support_cls1(example, head_num=head_num)
                # [[True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True],
                # [True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True],
                # [True], [True], [True], [True], [True], [True], [True], [True]]

                idx_list = idx.numpy().tolist()

                for i, pred in enumerate(pred_bool): # arg will be an item in the list, an integer.
                    assert len(pred) == 1
                    with open(test_filepath, "a") as f:
                        line = {"idx":int(idx_list[i]), "label": "true" if pred[0] else "false"}
                        json.dump(line, f)
                        f.write('\n')

    def _idx_example_test_helper_StrategyQA(self, data, test_filepath): # used for BoolQ
        # Each line of the prediction files should be a JSON entry, with an 'idx' field to
        # identify the example and a 'label' field with the prediction.
        for (idx, input_ids, attention_mask, token_type_ids) in data:
            example = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

            pred_bool = self.get_pred_batch_support_cls1(example)
            # [[True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True],
            # [True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True],
            # [True], [True], [True], [True], [True], [True], [True], [True]]

            idx_list = idx.numpy().tolist()

            for i, pred in enumerate(pred_bool): # arg will be an item in the list, an integer.
                assert len(pred) == 1
                with open(test_filepath, "a") as f:
                    line = {"idx":str(idx_list[i]), "label": "true" if pred[0] else "false"}
                    json.dump(line, f)
                    f.write('\n')

    def _idx_predictions_WSC(self, data, test_filepath):
        # Each line of the prediction files should be a JSON entry, with an 'idx' field to
        # identify the example and a 'label' field with the prediction.

        for (idx, input_ids, attention_mask, token_type_ids) in data:
            example = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

            pred_bool = self.get_pred_batch_support_cls1(example)
            # [[True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True],
            # [True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True],
            # [True], [True], [True], [True], [True], [True], [True], [True]]

            idx_list = idx.numpy().tolist()

            for i, pred in enumerate(pred_bool): # arg will be an item in the list, an integer.
                assert len(pred) == 1
                with open(test_filepath, "a") as f:
                    line = {"idx":int(idx_list[i]), "label": "True" if pred[0] else "False"}
                    json.dump(line, f)
                    f.write('\n')

    def _create_tmp_dict(self, idx_passage):
        return {"idx":idx_passage, "passage":{}}

    def _create_tmp_dict_question(self, question_idx):
        return {"idx":question_idx, "answers": []}

    def _create_tmp_dict_answer(self, label_idx, label):
        return {"idx":label_idx, "label":label}

    def _idx_predictions_MultiRC(self, data, test_filepath):

        '''
        {"idx": 0, "passage": {"questions": [{"idx": 0, "answers": [{"idx": 0, "label": 0}, {"idx": 1, "label": 0}, {"idx": 2, "label": 0}]}, {"idx": 1, "answers": [{"idx": 3, "label": 0}, {"idx": 4, "label": 0}, {"idx": 5, "label": 0}, {"idx": 6, "label": 0}, {"idx": 7, "label": 0}]}]}}
        '''

        #tmp_dict = {"idx":0, "passage": {}}
        #tmp_dict_passage = {"questions": [{"idx":0, "answers": []}]}
        #tmp_dict_answer = {"idx":0, "label":0}
        store_results = []
        start = True
        old_ex_idx = 0
        tmp_dict = None
        for (idx, input_ids, attention_mask, token_type_ids) in data:
            example = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

            pred_bool = self.get_pred_batch_support_cls1(example)
            # [[True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True],
            # [True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True],
            # [True], [True], [True], [True], [True], [True], [True], [True]]

            idx_list = idx.numpy().tolist()

            assert len(pred_bool) == len(idx_list)
            for i, pred in enumerate(pred_bool): # arg will be an item in the list, an integer.
                assert len(pred) == 1

                if start:
                    tmp_dict = self._create_tmp_dict(idx_list[i][0])
                    start = False
                    old_ex_idx = idx_list[i][0]

                if old_ex_idx != idx_list[i][0]:
                    store_results.append(tmp_dict)
                    tmp_dict = self._create_tmp_dict(idx_list[i][0])
                    old_ex_idx = idx_list[i][0]

                if len(tmp_dict["passage"].keys()) == 0: # passage will be an empty dict.
                    tmp_dict["passage"]["questions"] = [] # will be a list of dicts.
                    tmp_dict["passage"]["questions"].append(self._create_tmp_dict_question(idx_list[i][1]))
                    tmp_dict["passage"]["questions"][-1]["answers"].append(self._create_tmp_dict_answer(
                        label_idx=idx_list[i][2], label=1 if pred[0] else 0))
                else:
                    if tmp_dict["passage"]["questions"][-1]["idx"] == idx_list[i][1]:
                        #_create_tmp_dict_answer
                        tmp_dict["passage"]["questions"][-1]["answers"].append(self._create_tmp_dict_answer(
                            label_idx=idx_list[i][2], label=1 if pred[0] else 0))
                    else: # create a new question dict.
                        tmp_dict["passage"]["questions"].append(self._create_tmp_dict_question(idx_list[i][1]))
                        tmp_dict["passage"]["questions"][-1]["answers"].append(self._create_tmp_dict_answer(
                            label_idx=idx_list[i][2], label=1 if pred[0] else 0))

        store_results.append(tmp_dict)

        # use to save later...
        for dict_ in store_results:
            with open(test_filepath, "a") as f:
                json.dump(dict_, f)
                f.write('\n')


    def _idx_predictions_ReCoRD(self, data, test_filepath):

        max_question_idx = 0

        store_results = []
        for (idx, input_ids, attention_mask, token_type_ids) in data:
            example = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

            pred_prob = self.get_pred_batch_support_cls1_scores(example)
            # [num1, num2, num3, ...]

            idx_list = idx.numpy().tolist()
            for list_ in idx_list:
                if list_[1] > max_question_idx:
                    max_question_idx = list_[1]

            answers = self._extract_answer_ReCoRD(input_ids)

            assert len(pred_prob) == len(idx_list)
            assert len(pred_prob) == len(answers)
            for i, pred_value in enumerate(pred_prob):  # pred will be a number between 0 and 1.

                # get prediction and extract an answer.
                tmp_dict = {} # {"example_idx": ,"question_idx": , "cls_value": , "answer":}
                tmp_dict["question_idx"], tmp_dict["cls_sig_value"], tmp_dict["answer"] = idx[i][1], pred_value, answers[i]
                store_results.append(tmp_dict)

        # need to get the max answer for each question_idx and store it in the following format:
        # {"idx":0, "label": "Ward"}
        print(f"The max question idx is: {max_question_idx}")
        tmp_list = [[] for _ in range(max_question_idx+1)] # 0 to max_question_idx inclusive.
        for i, dict_ in enumerate(store_results):
            tmp_list[dict_["question_idx"]].append(dict_)

        # use to save later...
        for i, list_of_dicts in enumerate(tmp_list):
            qidx_check = i
            with open(test_filepath, "a") as f:

                tmp_dict = {}
                max_value_and_answer = [0, ""]
                for dict_ in list_of_dicts:
                    assert dict_["question_idx"] == qidx_check, f"Invalid question idx! It should be {qidx_check}, " \
                                                                f"got {dict_['question_idx']}!"
                    if dict_["cls_sig_value"] > max_value_and_answer[0]:
                        max_value_and_answer[0], max_value_and_answer[1] = dict_["cls_sig_value"], dict_["answer"]

                tmp_dict["idx"], tmp_dict["label"] = qidx_check, max_value_and_answer[1]

                json.dump(tmp_dict, f)
                f.write('\n')

    def _extract_answer_ReCoRD(self, input_ids):
        # input_ids.shape == (batch_size, max_seq_len)
        #self.tokenizer.batch_decode(example['input_ids'])

        # get [SEP] token id.
        sep_token_list = self.tokenizer.encode("[SEP]", add_special_tokens=False)
        assert len(sep_token_list) == 1, f"Error: the tokenizer should return a list of length 1, got {sep_token_list}!"
        sep_token_id = sep_token_list[0]

        batch_answers = []
        for b in range(input_ids.shape[0]):
            # get [SEP] token positions
            sep_token_positions = [i for i in range(input_ids.shape[1]) if input_ids[b,i]==sep_token_id]
            assert len(sep_token_positions) == 3, f"there should be three sep tokens in each example, got " \
                                                  f"{sep_token_positions}: {len(sep_token_positions)}!"
            batch_answers.append(self.tokenizer.decode(input_ids[b,sep_token_positions[-2]+1:sep_token_positions[-1]]))
        #print(batch_answers)
        return batch_answers



    def get_pred_batch_support_cls1_scores(self, example):
        _, pred_logits = self.model(example, training=False)

        # pred_logits.shape (batch_size, 1)
        pred_prob = tf.math.sigmoid(pred_logits) # batch_size, 1
        return tf.squeeze(pred_prob).numpy().tolist() # [0.02,0.45,0.97] for a batch size of 3.

    def get_pred_batch_support_cls1(self, example, head_num=None):
        pred_logits=None
        if head_num is None:
            out, pred_logits = self.model(example, training=False)
        else:
            out, pred_logits = self.model(example, training=False, head_num=head_num)

        #print(out.last_hidden_state.shape)
        #print(out.last_hidden_state[:,24,:])
        # pred_logits.shape (batch_size, 1)
        #print(pred_logits.shape)
        pred_prob = tf.math.sigmoid(pred_logits)
        #print(pred_prob)
        #print(f"pred_prob: {pred_prob}")
        #max_args = tf.argmax(pred_prob, axis=-1) # the softmax isn't needed here; the logits can be used.
        pred_bool = tf.math.greater(pred_prob, 0.5).numpy().tolist()
        #print(f"preb_bool: {pred_bool}")
        #max_args = max_args.numpy().tolist() # [1,2,3,4, ..., etc.]
        return pred_bool

    def get_pred_batch_support_clsn(self, example):
        _, pred_logits = self.model(example, training=False)

        max_args = tf.argmax(pred_logits, axis=-1) # the softmax isn't needed here; the logits can be used.
        max_args = max_args.numpy().tolist() # [1,2,3,4, ..., etc.]
        return max_args

    def _idx_predictions_CB_sigmoid(self, data, test_filepath, n):
        # Each line of the prediction files should be a JSON entry, with an 'idx' field to
        # identify the example and a 'label' field with the prediction.
        for (idx, input_ids, attention_mask, token_type_ids) in data:
            example = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

            pred_bool = self.get_pred_batch_support_mcqan(example, n=n)

            idx_list = idx.numpy().tolist()
            assert float(len(pred_bool)) == float(len(idx_list)/n)
            for i, pred in enumerate(pred_bool):  # arg will be an item in the list, an integer.
                with open(test_filepath, "a") as f:
                    max_str = None
                    if pred == 0: max_str = "contradiction"
                    elif pred == 1: max_str = "neutral"
                    if pred == 2: max_str = "entailment"
                    assert max_str is not None
                    line = {"idx": int(idx_list[i*n]), "label": max_str}
                    json.dump(line, f)
                    f.write('\n')

    def _idx_predictions_MPE_sigmoid(self, data, test_filepath, n):
        # Each line of the prediction files should be a JSON entry, with an 'idx' field to
        # identify the example and a 'label' field with the prediction.
        for (idx, input_ids, attention_mask, token_type_ids) in data:
            example = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

            pred_bool = self.get_pred_batch_support_mcqan(example, n=n)

            idx_list = idx.numpy().tolist()
            assert float(len(pred_bool)) == float(len(idx_list)/n)
            for i, pred in enumerate(pred_bool):  # arg will be an item in the list, an integer.
                with open(test_filepath, "a") as f:
                    max_str = None
                    if pred == 0: max_str = "entailment"
                    elif pred == 1: max_str = "neutral"
                    if pred == 2: max_str = "contradiction"
                    assert max_str is not None
                    line = {"idx": int(idx_list[i*n]), "label": max_str}
                    json.dump(line, f)
                    f.write('\n')

    def _idx_predictions_RACE_sigmoid(self, data, test_filepath, n):
        # Each line of the prediction files should be a JSON entry, with an 'idx' field to
        # identify the example and a 'label' field with the prediction.
        for (idx, input_ids, attention_mask, token_type_ids) in data:
            example = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

            pred_bool = self.get_pred_batch_support_mcqan(example, n=n)

            idx_list = idx.numpy().tolist()
            assert float(len(pred_bool)) == float(len(idx_list)/n)
            for i, pred in enumerate(pred_bool):  # arg will be an item in the list, an integer.
                with open(test_filepath, "a") as f:
                    max_str = None
                    if pred == 0: max_str = "A"
                    elif pred == 1: max_str = "B"
                    elif pred == 2: max_str = "C"
                    elif pred == 3: max_str = "D"
                    assert max_str is not None
                    line = {"idx": str(idx_list[i*n]), "label": max_str}
                    json.dump(line, f)
                    f.write('\n')

    def _idx_predictions_DREAM_sigmoid(self, data, test_filepath, n):
        # Each line of the prediction files should be a JSON entry, with an 'idx' field to
        # identify the example and a 'label' field with the prediction.
        for (idx, input_ids, attention_mask, token_type_ids) in data:
            example = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

            pred_bool = self.get_pred_batch_support_mcqan(example, n=n)

            idx_list = idx.numpy().tolist()
            assert float(len(pred_bool)) == float(len(idx_list)/n)
            for i, pred in enumerate(pred_bool):  # arg will be an item in the list, an integer.
                with open(test_filepath, "a") as f:
                    max_str = None
                    if pred == 0: max_str = "A"
                    elif pred == 1: max_str = "B"
                    elif pred == 2: max_str = "C"
                    assert max_str is not None
                    line = {"idx": str(idx_list[i*n]), "label": max_str}
                    json.dump(line, f)
                    f.write('\n')

    def _idx_predictions_ReClor_sigmoid(self, data, test_filepath, n):
        # Each line of the prediction files should be a JSON entry, with an 'idx' field to
        # identify the example and a 'label' field with the prediction.
        for (idx, input_ids, attention_mask, token_type_ids) in data:
            example = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

            pred_bool = self.get_pred_batch_support_mcqan(example, n=n)

            idx_list = idx.numpy().tolist()
            assert float(len(pred_bool)) == float(len(idx_list)/n)
            for i, pred in enumerate(pred_bool):  # arg will be an item in the list, an integer.
                with open(test_filepath, "a") as f:
                    max_str = None
                    if pred == 0: max_str = 0
                    elif pred == 1: max_str = 1
                    elif pred == 2: max_str = 2
                    elif pred == 3: max_str = 3
                    assert max_str is not None
                    line = {"idx": str(idx_list[i*n]), "label": max_str}
                    json.dump(line, f)
                    f.write('\n')

    def _idx_predictions_CQA_sigmoid(self, data, test_filepath, n):
        # Each line of the prediction files should be a JSON entry, with an 'idx' field to
        # identify the example and a 'label' field with the prediction.
        for (idx, input_ids, attention_mask, token_type_ids) in data:
            example = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

            pred_bool = self.get_pred_batch_support_mcqan(example, n=n)

            idx_list = idx.numpy().tolist()
            assert float(len(pred_bool)) == float(len(idx_list)/n)
            for i, pred in enumerate(pred_bool):  # arg will be an item in the list, an integer.
                with open(test_filepath, "a") as f:
                    max_str = None
                    if pred == 0: max_str = "A"
                    elif pred == 1: max_str = "B"
                    elif pred == 2: max_str = "C"
                    elif pred == 3: max_str = "D"
                    elif pred == 4: max_str = "E"
                    assert max_str is not None
                    line = {"idx": str(idx_list[i*n]), "label": max_str}
                    json.dump(line, f)
                    f.write('\n')

    def _idx_predictions_PIQA_sigmoid(self, data, test_filepath, n):
        # Each line of the prediction files should be a JSON entry, with an 'idx' field to
        # identify the example and a 'label' field with the prediction.
        for (idx, input_ids, attention_mask, token_type_ids) in data:
            example = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

            pred_bool = self.get_pred_batch_support_mcqan(example, n=n)

            idx_list = idx.numpy().tolist()
            assert float(len(pred_bool)) == float(len(idx_list)/n)
            for i, pred in enumerate(pred_bool):  # arg will be an item in the list, an integer.
                with open(test_filepath, "a") as f:
                    max_int = None
                    if pred == 0: max_int = 0
                    elif pred == 1: max_int = 1
                    assert max_int is not None
                    line = {"idx": str(idx_list[i*n]), "label": max_int}
                    json.dump(line, f)
                    f.write('\n')

    def _idx_predictions_SIQA_sigmoid(self, data, test_filepath, n):
        # Each line of the prediction files should be a JSON entry, with an 'idx' field to
        # identify the example and a 'label' field with the prediction.
        for (idx, input_ids, attention_mask, token_type_ids) in data:
            example = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

            pred_bool = self.get_pred_batch_support_mcqan(example, n=n)

            idx_list = idx.numpy().tolist()
            assert float(len(pred_bool)) == float(len(idx_list)/n)
            for i, pred in enumerate(pred_bool):  # arg will be an item in the list, an integer.
                with open(test_filepath, "a") as f:
                    max_int = None
                    if pred == 0: max_int = 1
                    elif pred == 1: max_int = 2
                    elif pred == 2: max_int = 3
                    assert max_int is not None
                    line = {"idx": str(idx_list[i*n]), "label": max_int}
                    json.dump(line, f)
                    f.write('\n')

    def _idx_predictions_CB_softmax(self, data, test_filepath):
        # Each line of the prediction files should be a JSON entry, with an 'idx' field to
        # identify the example and a 'label' field with the prediction.
        for (idx, input_ids, attention_mask, token_type_ids) in data:
            example = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
            pred_bool = self.get_pred_batch_support_clsn(example)
            # [1,2,3,4,5,6,7,8,9]
            idx_list = idx.numpy().tolist()
            assert len(pred_bool) == len(idx_list)
            for i, pred in enumerate(pred_bool):  # arg will be an item in the list, an integer.
                with open(test_filepath, "a") as f:
                    label_ = ''
                    if pred == 0:
                        label_ = "contradiction"
                    elif pred == 1:
                        label_ = "neutral"
                    elif pred == 2:
                        label_ = "entailment"
                    else:
                        raise Exception(f"Invalid prediction index!")
                    line = {"idx": int(idx_list[i]), "label": label_}
                    json.dump(line, f)
                    f.write('\n')

    def _idx_predictions_RTE(self, data, test_filepath):
        # Each line of the prediction files should be a JSON entry, with an 'idx' field to
        # identify the example and a 'label' field with the prediction.

        for (idx, input_ids, attention_mask, token_type_ids) in data:
            example = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

            pred_bool = self.get_pred_batch_support_cls1(example)
            # [[True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True],
            # [True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True], [True],
            # [True], [True], [True], [True], [True], [True], [True], [True]]

            idx_list = idx.numpy().tolist()

            for i, pred in enumerate(pred_bool):  # arg will be an item in the list, an integer.
                assert len(pred) == 1
                with open(test_filepath, "a") as f:
                    line = {"idx": int(idx_list[i]), "label": "entailment" if pred[0] else "not_entailment"}
                    json.dump(line, f)
                    f.write('\n')

    def _idx_predictions_COPA(self, data, test_filepath):

        for (idx, input_ids, attention_mask, token_type_ids) in data:
            example = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}

            pred_bool = self.get_pred_batch_support_mcqa2(example)

            idx_list = idx.numpy().tolist()
            assert float(len(pred_bool)) == float(len(idx_list)/2)
            for i, pred in enumerate(pred_bool):  # arg will be an item in the list, an integer.
                with open(test_filepath, "a") as f:
                    line = {"idx": int(idx_list[i*2]), "label": pred}
                    json.dump(line, f)
                    f.write('\n')

    def get_pred_batch_support_mcqa2(self, example):
        # this function is for COPA (the step of two is the main addition).
        _, pred_logits = self.model(example, training=False)
        # pred_logits.shape = (batch_size, 1)
        max_args = []
        for i in range(0,pred_logits.shape[0],2):
            if pred_logits[i,0] > pred_logits[i+1,0]: max_args.append(0)
            else: max_args.append(1)
        return max_args

    def get_pred_batch_support_mcqan(self, example, n):
        _, pred_logits = self.model(example, training=False)
        # pred_logits.shape = (batch_size, 1)
        max_args = []
        for i in range(0,pred_logits.shape[0],n):
            max_j = 0
            max_val = 0
            for j in range(0,n):
                #print(f"pred_logits[{i}+{j},0]: {pred_logits[i+j,0]}")
                if pred_logits[i+j,0] > max_val:
                    max_j = j
                    max_val = pred_logits[i+j,0]
            #print(f"BREAK")
            max_args.append(max_j)

            # for CB 0 is contradiction, 1 is neutral, and 2 is entailment.
        return max_args

    def get_test_results(self, data, test_filepath, type_, pred_mode="default"):
        '''
        Function: get_test_results;
        Description: This function stores the model's predictions in a file with its assocaited idx.
        :return:
        '''

        if type_ == "idx_example_test_helper" or type_ == "_idx_example_test_helper" \
            or type_ == "idx_predictions_BoolQ" or type_ == "idx_predictions_WiC":
            self._idx_example_test_helper(data, test_filepath)
        elif type_ == "idx_predictions_StrategyQA":
            self._idx_example_test_helper_StrategyQA(data, test_filepath)
        elif type_ == "idx_predictions_CB_softmax":
            self._idx_predictions_CB_softmax(data, test_filepath)
        elif type_ == "idx_predictions_CB_sigmoid" or type_ == "idx_predictions_SNLI_sigmoid":
            # contradiction, neutral, entailment triplet ordering.
            self._idx_predictions_CB_sigmoid(data, test_filepath, n=3)
        elif type_ == "idx_predictions_MPE_sigmoid":
            # entailment, neutral, contradiction triplet ordering.
            self._idx_predictions_MPE_sigmoid(data, test_filepath, n=3)
        elif type_ == "idx_predictions_RACE_sigmoid":
            self._idx_predictions_RACE_sigmoid(data, test_filepath, n=4)
        elif type_ == "idx_predictions_DREAM_sigmoid":
            self._idx_predictions_DREAM_sigmoid(data, test_filepath, n=3)
        elif type_ == "idx_predictions_ReClor_sigmoid":
            self._idx_predictions_ReClor_sigmoid(data, test_filepath, n=4)
        elif type_ == "idx_predictions_CQA_sigmoid":
            self._idx_predictions_CQA_sigmoid(data, test_filepath, n=5)
        elif type_ == "idx_predictions_PIQA_sigmoid":
            self._idx_predictions_PIQA_sigmoid(data, test_filepath, n=2)
        elif type_ == "idx_predictions_SIQA_sigmoid":
            self._idx_predictions_SIQA_sigmoid(data, test_filepath, n=3)
        elif type_ == "idx_predictions_COPA":
            self._idx_predictions_COPA(data, test_filepath)
        elif type_ == "idx_predictions_RTE" or type_ == "idx_predictions_SciTail" or type_ == "idx_predictions_MED":
            self._idx_predictions_RTE(data, test_filepath)
        elif type_ == "idx_predictions_WSC":
            self._idx_predictions_WSC(data, test_filepath)
        elif type_ == "idx_predictions_MultiRC":
            self._idx_predictions_MultiRC(data, test_filepath)
        elif type_ == "idx_predictions_ReCoRD":
            self._idx_predictions_ReCoRD(data, test_filepath)


def loss_function_cls(target, prediction, loss_object):
    # target.shape = (batch_size, n_cls_opt) |  [[0,1],[1,0],[0,1]] # true predictions.
    # prediction.shape = (batch_size, n_cls_opt) | [[.4,.6],[.1,.9],[.5,.5]]
    assert target.shape == prediction.shape, f"The target and prediction shapes do not match {target.shape}{prediction.shape}!"
    loss_ = loss_object(target, prediction)  # get loss from the loss_object -> here reduction must be 'none'.
    return tf.reduce_sum(loss_), target.shape[0] # the size will be the batch size; each batch has one loss value assocaited with it.