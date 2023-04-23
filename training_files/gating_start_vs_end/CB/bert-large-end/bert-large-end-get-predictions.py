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

from training_files.get_metric_score_functions import *
from misc.BertConfigClasses import BertModelConfigs
from misc.get_dataset_filepaths import get_dataset_filepaths
from training_files.parent_training_class import loss_function_cls

from data_loaders.general_data_loader import *
from data_loaders.BoolQ import *


if __name__ == "__main__":

    type_ = "bert_large_cased_gating_end_only_3_layers_512sl"
    config = BertModelConfigs()
    config_dict = config.return_config_dictionary(type_=type_)
    config_dict["cls_dense_layer_number_of_options"] = 3  # manually change the layer to have three answer options.

    filepaths_BoolQ, filepaths_CB, filepaths_COPA, filepaths_MultiRC, filepaths_ReCoRD, \
    filepaths_RTE, filepaths_WiC, filepaths_WSC = get_dataset_filepaths()

    dataset = "CB"
    data_loader_dict = {"filepath_BoolQ": filepaths_BoolQ,
                        "filepath_CB": filepaths_CB,
                        "filepath_COPA": filepaths_COPA,
                        "filepath_MultiRC": filepaths_MultiRC,
                        "filepath_ReCoRD": filepaths_ReCoRD,
                        "filepath_RTE": filepaths_RTE,
                        "filepath_WiC": filepaths_WiC,
                        "filepath_WSC": filepaths_WSC,
                        "is_test_BoolQ": False,
                        "is_test_CB": True,
                        "is_test_COPA": False,
                        "is_test_MultiRC": False,
                        "is_test_ReCoRD": False,
                        "is_test_RTE": False,
                        "is_test_WiC": False,
                        "is_test_WSC": False,
                        "is_BoolQ": False,
                        "is_CB": True,
                        "is_COPA": False,
                        "is_MultiRC": False,
                        "is_ReCoRD": False,
                        "is_RTE": False,
                        "is_WiC": False,
                        "is_WSC": False,
                        "max_seq_len": config_dict["max_seq_len"],
                        "is_train": False,
                        "is_val": True,
                        "is_test": False}  # is_val is an easy way to tell the training function that we are utilising the val dataset.

    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    batch_size = 6*GPUS_AVAILABLE
    strategy = None

    #epoch=3
    exp=3
    for epoch in range(1,4):
        type_ = "idx_predictions_CB_softmax"
        pretrained = "/data/kkno604/NGT_experiments_updated/hyperparameter-search/CB/BERT-large/" \
                     "gating-end/exp"+str(exp)+"/Checkpoints/epoch"+str(epoch)+"/"
        test_save_filepath = "/data/kkno604/NGT_experiments_updated/hyperparameter-search/CB/BERT-large/" \
                             "gating-end/exp"+str(exp)+"/Results/val-predictions/epoch"+str(epoch)+".jsonl"

        general_get_predictions(config_dict=config_dict, data_loader_dict=data_loader_dict, strategy=strategy,
                                tokenizer=tokenizer, batch_size=batch_size, pretrained=pretrained,
                                shuffle_=False, type_=type_, dataset=dataset, test_filepath=test_save_filepath,
                                error_function="softmax")