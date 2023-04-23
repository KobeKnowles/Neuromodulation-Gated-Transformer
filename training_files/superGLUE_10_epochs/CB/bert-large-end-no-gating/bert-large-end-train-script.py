import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPUS_AVAILABLE = 1

import sys
sys.path.append("../../../..")

import tensorflow as tf

#import logging
#logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

print(f"\nTensorflow version: {tf.__version__}\n")

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np

from training_files.parent_training_class import *

import random

from training_files.training_functions import *
from misc.BertConfigClasses import BertModelConfigs
from training_files.parent_training_class import loss_function_cls

from data_loaders.general_data_loader import *
from data_loaders.BoolQ import *

from misc.get_dataset_filepaths import get_dataset_filepaths

if __name__ == "__main__":

    type_ = "bert_large_cased_no_gating_end_only_3_layers_512sl"
    pretrained = "bert-large-cased"

    config = BertModelConfigs()
    config_dict = config.return_config_dictionary(type_=type_)
    config_dict["cls_dense_layer_number_of_options"] = 3  # manually change the layer to have three answer options.

    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    batch_size = 8*GPUS_AVAILABLE
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='none')

    filepaths_BoolQ, filepaths_CB, filepaths_COPA, filepaths_MultiRC, filepaths_ReCoRD, \
                filepaths_RTE, filepaths_WiC, filepaths_WSC = get_dataset_filepaths()

    dataset = "CB"
    data_loader_dict = {"filepath_BoolQ":filepaths_BoolQ,
                        "filepath_CB":filepaths_CB,
                        "filepath_COPA":filepaths_COPA,
                        "filepath_MultiRC":filepaths_MultiRC,
                        "filepath_ReCoRD":filepaths_ReCoRD,
                        "filepath_RTE":filepaths_RTE,
                        "filepath_WiC":filepaths_WiC,
                        "filepath_WSC":filepaths_WSC,
                        "is_test_BoolQ":False,
                        "is_test_CB":False,
                        "is_test_COPA":False,
                        "is_test_MultiRC":False,
                        "is_test_ReCoRD":False,
                        "is_test_RTE":False,
                        "is_test_WiC":False,
                        "is_test_WSC":False,
                        "is_BoolQ":False,
                        "is_CB":True,
                        "is_COPA":False,
                        "is_MultiRC":False,
                        "is_ReCoRD":False,
                        "is_RTE":False,
                        "is_WiC":False,
                        "is_WSC":False,
                        "max_seq_len":config_dict["max_seq_len"],
                        "is_train":True,
                        "is_val":True,
                        "is_test":False} # is_val is an easy way to tell hte training function that we are utilising the val dataset.

    decay_steps = 32*10 #(250 examples/8batchsize) across 10 epochs | 93.75
    decay_function="cosine_decay"
    learning_rate_starting_value = 0.00001
    learning_rate_end_value = 0
    learning_rate=tf.keras.experimental.CosineDecay(learning_rate_starting_value, decay_steps,
                                                    alpha=learning_rate_end_value)

    tf_seed = random.randint(0, sys.maxsize)
    random_seed = random.randint(0, sys.maxsize)

    strategy = None
    save_end_epoch = True
    print_every_iterations = 10
    epoch_start = 0
    epoch_end = 10
    train_shuffle = True
    val_shuffle = False

    train_save_filepath = "/data/kkno604/NGT_experiments_updated/superGLUE-experiments/CB/bert-large/no-gating-end/exp3/Results/"
    val_save_filepath = "/data/kkno604/NGT_experiments_updated/superGLUE-experiments/CB/bert-large/no-gating-end/exp3/Results/"
    checkpoint_save_filepath = "/data/kkno604/NGT_experiments_updated/superGLUE-experiments/CB/bert-large/no-gating-end/exp3/Checkpoints/"

    general_training_function(config_dict=config_dict, data_loader_dict=data_loader_dict, strategy=strategy,
                              learning_rate=learning_rate, learning_rate_starting_value=learning_rate_starting_value,
                              learning_rate_end_value=learning_rate_end_value, tokenizer=tokenizer,
                              batch_size=batch_size, loss_object=loss_object, loss_function=loss_function_cls,
                              pretrained=pretrained, tf_seed=tf_seed, random_seed=random_seed,
                              epoch_start=epoch_start, epoch_end=epoch_end, train_save_filepath=train_save_filepath,
                              val_save_filepath=val_save_filepath, save_end_epoch=save_end_epoch,
                              checkpoint_save_filepath=checkpoint_save_filepath, print_every_iterations=print_every_iterations,
                              dataset=dataset, num_GPUs=GPUS_AVAILABLE, train_shuffle=train_shuffle,
                              val_shuffle=val_shuffle, decay_steps=decay_steps, decay_function=decay_function,
                              error_function="softmax")