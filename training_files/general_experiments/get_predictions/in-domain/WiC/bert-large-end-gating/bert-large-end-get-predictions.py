import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPUS_AVAILABLE = 1

import sys
sys.path.append("../../../../../..")

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

    type__ = "bert_large_cased_gating_end_only_3_layers_512sl"
    #type__ = "bert_large_cased_original_512sl"
    config = BertModelConfigs()
    config_dict = config.return_config_dictionary(type_=type__)

    #filepaths_BoolQ, filepaths_CB, filepaths_COPA, filepaths_MultiRC, filepaths_ReCoRD, \
    #filepaths_RTE, filepaths_WiC, filepaths_WSC = get_dataset_filepaths()

    dataset = "WiC_test"
    data_loader_dict = {"filepath_BoolQ": "/large_data/SuperGlue/BoolQ/val.jsonl",
                        "filepath_CB": "/large_data/SuperGlue/CB/val.jsonl",
                        "filepath_COPA": "/large_data/SuperGlue/COPA/val.jsonl",
                        "filepath_MultiRC": "/large_data/SuperGlue/MultiRC/val.jsonl",
                        "filepath_ReCoRD": "/large_data/SuperGlue/ReCoRD/val.jsonl",
                        "filepath_RTE": "/large_data/SuperGlue/RTE/val.jsonl",
                        "filepath_WiC": "/large_data/SuperGlue/WiC/val.jsonl",
                        "filepath_WSC": "/large_data/SuperGlue/WSC/val.jsonl",
                        "filepath_CQA": "/large_data/CommonsenseQA/dev_rand_split.jsonl",
                        "filepath_RACE_middle": "/large_data/RACE/RACE/dev/middle/",
                        "filepath_RACE_high": "/large_data/RACE/RACE/dev/high/",
                        "filepath_SciTail": "/large_data/SciTail/SciTailV1.1/predictor_format/scitail_1.0_structure_dev.jsonl",
                        "filepath_MPE": "/large_data/MPE/MultiPremiseEntailment-master/data/MPE/mpe_dev.txt",
                        "is_test_BoolQ": False,
                        "is_test_CB": False,
                        "is_test_COPA": False,
                        "is_test_MultiRC": False,
                        "is_test_ReCoRD": False,
                        "is_test_RTE": False,
                        "is_test_WiC": True,
                        "is_test_WSC": False,
                        "is_test_CQA": False,
                        "is_test_RACE": False,
                        "is_test_SciTail": False,
                        "is_test_MPE": False,
                        "is_BoolQ": False,
                        "is_CB": False,
                        "is_COPA": False,
                        "is_MultiRC": False,
                        "is_ReCoRD": False,
                        "is_RTE": False,
                        "is_WiC": True,
                        "is_WSC": False,
                        "is_CQA": False,
                        "is_RACE": False,
                        "is_SciTail": False,
                        "is_MPE": False,
                        #"max_seq_len": config_dict["max_seq_len"],
                        "is_train": False,
                        "is_val": True,
                        "is_test": False}

    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    batch_size = 32*GPUS_AVAILABLE
    strategy = None

    config_dict["is_diagnostics"] = False
    config_dict["max_seq_len"] = 510

    exp=1
    num_aux_toks=2
    start_max_seq_len = 5-num_aux_toks
    updated_max_seq_len = 510#config_dict["max_seq_len"]
    type_ = "idx_predictions_WiC"
    for iteration in [50000, 100000, 150000, 200000]:

        pretrained = "/data/kkno604/NGT_experiments_updated/general_experiments/gating-end/" \
                     "exp"+str(exp)+"/Checkpoints/iteration"+str(iteration)+"/"
        test_save_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/gating-end/exp"+str(exp)+"/" \
                             "Results/get_results_in_domain/WiC/prediction_files/iteration"+str(iteration)+".jsonl"

        GQA_get_predictions(config_dict=config_dict, data_loader_dict=data_loader_dict, strategy=strategy,
                            tokenizer=tokenizer, batch_size=batch_size, pretrained=pretrained,
                            shuffle_=False, type_=type_, dataset=dataset, test_filepath=test_save_filepath,
                            error_function_cb="sigmoid", error_function_mpe="sigmoid", error_function="sigmoid",
                            head_strategy="uniform", dataset_strategy="uniform", num_iterations=200000,
                            is_aux_toks=True, num_aux_toks=num_aux_toks, start_max_seq_len=start_max_seq_len,
                            updated_max_seq_len=updated_max_seq_len)