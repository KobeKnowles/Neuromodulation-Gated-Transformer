import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
GPUS_AVAILABLE = 1

import sys
sys.path.append("../../..")

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

import transformers
from transformers import BertConfig, TFBertModel, BertTokenizer

from training_files.get_metric_score_functions import *
from misc.BertConfigClasses import BertModelConfigs
from misc.get_dataset_filepaths import get_dataset_filepaths
from training_files.parent_training_class import loss_function_cls
from training_files.parent_training_class import *

from data_loaders.general_data_loader import *

from get_test_results_final.get_cor_incor_idx.get_BoolQ_results import *#BoolQ_return_correct_incorrect_ids
from get_test_results_final.get_cor_incor_idx.get_CB_results import *#CB_return_correct_incorrect_ids
from get_test_results_final.get_cor_incor_idx.get_COPA_results import *#COPA_return_correct_incorrect_ids
from get_test_results_final.get_cor_incor_idx.get_CQA_results import *#CQA_return_correct_incorrect_ids
from get_test_results_final.get_cor_incor_idx.get_MPE_results import *#MPE_return_correct_incorrect_ids
from get_test_results_final.get_cor_incor_idx.get_MultiRC_results import *#MultiRC_return_correct_incorrect_ids
from get_test_results_final.get_cor_incor_idx.get_RACE_results import *#RACE_return_correct_incorrect_ids
from get_test_results_final.get_cor_incor_idx.get_ReCoRD_results import *#ReCoRD_return_correct_incorrect_ids
from get_test_results_final.get_cor_incor_idx.get_RTE_results import *#RTE_return_correct_incorrect_ids
from get_test_results_final.get_cor_incor_idx.get_SciTail_results import *#SciTail_return_correct_incorrect_ids
from get_test_results_final.get_cor_incor_idx.get_WiC_results import *#WiC_return_correct_incorrect_ids
from get_test_results_final.get_cor_incor_idx.get_WSC_results import *#WSC_return_correct_incorrect_ids

def get_nm_probe(config_dict, data_loader_dict, strategy, tokenizer, batch_size, pretrained,
                 shuffle_=False, type_="_idx_example_test_helper", dataset="BoolQ", test_filepath="",
                 error_function_cb="softmax", error_function_mpe="sigmoid", error_function="none",
                 head_strategy=None, dataset_strategy=None, num_iterations=None, is_aux_toks=True,
                 num_aux_toks: int=2, start_max_seq_len: int=3, updated_max_seq_len: int=510,
                 gqa:bool = False, is_global_probe_dataset=False,
                 global_filepath="", is_qualitative_probe=False,
                 qualitative_filepath="", is_global_before_and_after=False,
                 global_before_after_filepath="", ids_to_exclude=None):

    assert  is_global_probe_dataset or is_qualitative_probe or is_global_before_and_after, f"One of " \
        f"is_global_probe_dataset, is_qualitative_probe, or is_global_before_and_after must be True!"

    if not gqa:
        assert data_loader_dict["is_test_BoolQ"] or data_loader_dict["is_test_CB"] or data_loader_dict["is_test_COPA"] \
               or data_loader_dict["is_test_MultiRC"] or data_loader_dict["is_test_ReCoRD"] \
               or data_loader_dict["is_test_RTE"] or data_loader_dict["is_test_WiC"] \
               or data_loader_dict["is_test_WSC"] or data_loader_dict["is_test_MPE"] \
               or data_loader_dict["is_test_RACE"] or data_loader_dict["is_test_SciTail"] \
               or data_loader_dict["is_test_CQA"], f"One dataset's is_test should be set to True!"
    else:
        assert data_loader_dict["is_test_BoolQ"] or data_loader_dict["is_test_CB"] or data_loader_dict["is_test_COPA"] \
               or data_loader_dict["is_test_MultiRC"] or data_loader_dict["is_test_ReCoRD"] \
               or data_loader_dict["is_test_RTE"] or data_loader_dict["is_test_WiC"] \
               or data_loader_dict["is_test_WSC"] or data_loader_dict["is_test_MPE"] \
               or data_loader_dict["is_test_RACE"] or data_loader_dict["is_test_SciTail"] \
               or data_loader_dict["is_test_CQA"] or data_loader_dict["is_test_SNLI"] \
               or data_loader_dict["is_test_MED"] or data_loader_dict["is_test_PIQA"] \
               or data_loader_dict["is_test_SIQA"] or data_loader_dict["is_test_ReClor"] \
               or data_loader_dict["is_test_DREAM"] or data_loader_dict["is_test_StrategyQA"], \
            f"One dataset's is_test should be set to True!"

    model = None
    if strategy is not None:
        with strategy.scope():
            config = BertConfig(vocab_size=config_dict["vocab_size"], hidden_size=config_dict["hidden_size"],
                                num_hidden_layers=config_dict["num_hidden_layers"],
                                num_attention_heads=config_dict["num_attention_heads"],
                                intermediate_size=config_dict["intermediate_size"],
                                hidden_act=config_dict["hidden_act"],
                                hidden_dropout_prob=config_dict["hidden_dropout_prob"],
                                attention_probs_dropout_prob=config_dict["attention_probs_dropout_prob"],
                                max_position_embeddings=config_dict["max_position_embeddings"],
                                type_vocab_size=config_dict["type_vocab_size"],
                                initializer_range=config_dict["initializer_range"],
                                layer_norm_eps=config_dict["layer_norm_eps"],
                                pad_token_id=config_dict["pad_token_id"],
                                position_embedding_type=config_dict["position_embedding_type"],
                                use_cache=config_dict["use_cache"],
                                classifier_dropout=config_dict["classifier_dropout"],
                                gating_block_num_layers=config_dict["gating_block_num_layers"],
                                gating_block_end=config_dict["gating_block_end"],
                                gating_block_end_position=config_dict["gating_block_end_position"],
                                gating_block_middle=config_dict["gating_block_middle"],
                                gating_block_middle_position=config_dict["gating_block_middle_position"],
                                gating_block_start=config_dict["gating_block_start"],
                                gating_block_start_position=config_dict["gating_block_start_position"],
                                nm_gating=config_dict["nm_gating"],
                                cls_dense_layer_number_of_options=config_dict["cls_dense_layer_number_of_options"],
                                is_diagnostics=config_dict["is_diagnostics"],
                                num_aux_toks=num_aux_toks, max_seq_len=start_max_seq_len,
                                is_global_probe_dataset=is_global_probe_dataset,
                                global_filepath=global_filepath,
                                is_qualitative_probe=is_qualitative_probe,
                                qualitative_filepath=qualitative_filepath,
                                is_global_before_and_after=is_global_before_and_after,
                                global_before_after_filepath=global_before_after_filepath)
            model = TFBertModel.from_pretrained(pretrained, config=config)
            model.config.max_seq_len = config_dict["max_seq_len"]
            assert config_dict["max_seq_len"] == updated_max_seq_len
    else:
        config = BertConfig(vocab_size=config_dict["vocab_size"], hidden_size=config_dict["hidden_size"],
                            num_hidden_layers=config_dict["num_hidden_layers"],
                            num_attention_heads=config_dict["num_attention_heads"],
                            intermediate_size=config_dict["intermediate_size"],
                            hidden_act=config_dict["hidden_act"],
                            hidden_dropout_prob=config_dict["hidden_dropout_prob"],
                            attention_probs_dropout_prob=config_dict["attention_probs_dropout_prob"],
                            max_position_embeddings=config_dict["max_position_embeddings"],
                            type_vocab_size=config_dict["type_vocab_size"],
                            initializer_range=config_dict["initializer_range"],
                            layer_norm_eps=config_dict["layer_norm_eps"],
                            pad_token_id=config_dict["pad_token_id"],
                            position_embedding_type=config_dict["position_embedding_type"],
                            use_cache=config_dict["use_cache"],
                            classifier_dropout=config_dict["classifier_dropout"],
                            gating_block_num_layers=config_dict["gating_block_num_layers"],
                            gating_block_end=config_dict["gating_block_end"],
                            gating_block_end_position=config_dict["gating_block_end_position"],
                            gating_block_middle=config_dict["gating_block_middle"],
                            gating_block_middle_position=config_dict["gating_block_middle_position"],
                            gating_block_start=config_dict["gating_block_start"],
                            gating_block_start_position=config_dict["gating_block_start_position"],
                            nm_gating=config_dict["nm_gating"],
                            cls_dense_layer_number_of_options=config_dict["cls_dense_layer_number_of_options"],
                            is_diagnostics=config_dict["is_diagnostics"],
                            num_aux_toks=num_aux_toks, max_seq_len=start_max_seq_len,
                            is_global_probe_dataset=is_global_probe_dataset,
                            global_filepath=global_filepath,
                            is_qualitative_probe=is_qualitative_probe,
                            qualitative_filepath=qualitative_filepath,
                            is_global_before_and_after=is_global_before_and_after,
                            global_before_after_filepath=global_before_after_filepath)

        model = TFBertModel.from_pretrained(pretrained, config=config)

        model.config.max_seq_len = config_dict["max_seq_len"]
        assert config_dict["max_seq_len"] == updated_max_seq_len

    data = None
    dloader = None
    if not gqa:  # this if else is for compatibility with old code before the GQA experiments.
        dloader = general_data_loader(filepath_BoolQ=data_loader_dict["filepath_BoolQ"],
                                      filepath_CB=data_loader_dict["filepath_CB"],
                                      filepath_COPA=data_loader_dict["filepath_COPA"],
                                      filepath_MultiRC=data_loader_dict["filepath_MultiRC"],
                                      filepath_ReCoRD=data_loader_dict["filepath_ReCoRD"],
                                      filepath_RTE=data_loader_dict["filepath_RTE"],
                                      filepath_WiC=data_loader_dict["filepath_WiC"],
                                      filepath_WSC=data_loader_dict["filepath_WSC"],
                                      filepath_CQA=data_loader_dict["filepath_CQA"],
                                      filepath_RACE_middle=data_loader_dict["filepath_RACE_middle"],
                                      filepath_RACE_high=data_loader_dict["filepath_RACE_high"],
                                      filepath_SciTail=data_loader_dict["filepath_SciTail"],
                                      filepath_MPE=data_loader_dict["filepath_MPE"],
                                      tokenizer=tokenizer,
                                      is_test_BoolQ=data_loader_dict["is_test_BoolQ"],
                                      is_test_CB=data_loader_dict["is_test_CB"],
                                      is_test_COPA=data_loader_dict["is_test_COPA"],
                                      is_test_MultiRC=data_loader_dict["is_test_MultiRC"],
                                      is_test_ReCoRD=data_loader_dict["is_test_ReCoRD"],
                                      is_test_RTE=data_loader_dict["is_test_RTE"],
                                      is_test_WiC=data_loader_dict["is_test_WiC"],
                                      is_test_WSC=data_loader_dict["is_test_WSC"],
                                      is_test_CQA=data_loader_dict["is_test_CQA"],
                                      is_test_RACE=data_loader_dict["is_test_RACE"],
                                      is_test_SciTail=data_loader_dict["is_test_SciTail"],
                                      is_test_MPE=data_loader_dict["is_test_MPE"],
                                      is_BoolQ=data_loader_dict["is_BoolQ"], is_CB=data_loader_dict["is_CB"],
                                      is_COPA=data_loader_dict["is_COPA"], is_MultiRC=data_loader_dict["is_MultiRC"],
                                      is_ReCoRD=data_loader_dict["is_ReCoRD"], is_MPE=data_loader_dict["is_MPE"],
                                      is_RTE=data_loader_dict["is_RTE"], is_WiC=data_loader_dict["is_WiC"],
                                      is_WSC=data_loader_dict["is_WSC"], is_CQA=data_loader_dict["is_CQA"],
                                      is_RACE=data_loader_dict["is_RACE"], is_SciTail=data_loader_dict["is_SciTail"],
                                      error_function_cb=error_function_cb, error_function_mpe=error_function_mpe,
                                      max_seq_len=model.config.max_seq_len, ids_to_exclude=ids_to_exclude)
    else:
        dloader = general_data_loader(filepath_BoolQ=data_loader_dict["filepath_BoolQ"],
                                      filepath_CB=data_loader_dict["filepath_CB"],
                                      filepath_COPA=data_loader_dict["filepath_COPA"],
                                      filepath_MultiRC=data_loader_dict["filepath_MultiRC"],
                                      filepath_ReCoRD=data_loader_dict["filepath_ReCoRD"],
                                      filepath_RTE=data_loader_dict["filepath_RTE"],
                                      filepath_WiC=data_loader_dict["filepath_WiC"],
                                      filepath_WSC=data_loader_dict["filepath_WSC"],
                                      filepath_CQA=data_loader_dict["filepath_CQA"],
                                      filepath_RACE_middle=data_loader_dict["filepath_RACE_middle"],
                                      filepath_RACE_high=data_loader_dict["filepath_RACE_high"],
                                      filepath_SciTail=data_loader_dict["filepath_SciTail"],
                                      filepath_MPE=data_loader_dict["filepath_MPE"],
                                      filepath_SNLI=data_loader_dict["filepath_SNLI"],
                                      filepath_MED=data_loader_dict["filepath_MED"],
                                      filepath_PIQA=data_loader_dict["filepath_PIQA"],
                                      filepath_labels_PIQA=data_loader_dict["filepath_labels_PIQA"],
                                      filepath_SIQA=data_loader_dict["filepath_SIQA"],
                                      filepath_labels_SIQA=data_loader_dict["filepath_labels_SIQA"],
                                      filepath_ReClor=data_loader_dict["filepath_ReClor"],
                                      filepath_DREAM=data_loader_dict["filepath_DREAM"],
                                      filepath_StrategyQA=data_loader_dict["filepath_StrategyQA"],
                                      tokenizer=tokenizer,
                                      is_test_BoolQ=data_loader_dict["is_test_BoolQ"],
                                      is_test_CB=data_loader_dict["is_test_CB"],
                                      is_test_COPA=data_loader_dict["is_test_COPA"],
                                      is_test_MultiRC=data_loader_dict["is_test_MultiRC"],
                                      is_test_ReCoRD=data_loader_dict["is_test_ReCoRD"],
                                      is_test_RTE=data_loader_dict["is_test_RTE"],
                                      is_test_WiC=data_loader_dict["is_test_WiC"],
                                      is_test_WSC=data_loader_dict["is_test_WSC"],
                                      is_test_CQA=data_loader_dict["is_test_CQA"],
                                      is_test_RACE=data_loader_dict["is_test_RACE"],
                                      is_test_SciTail=data_loader_dict["is_test_SciTail"],
                                      is_test_MPE=data_loader_dict["is_test_MPE"],
                                      is_test_SNLI=data_loader_dict["is_test_SNLI"],
                                      is_test_MED=data_loader_dict["is_test_MED"],
                                      is_test_PIQA=data_loader_dict["is_test_PIQA"],
                                      is_test_SIQA=data_loader_dict["is_test_SIQA"],
                                      is_test_ReClor=data_loader_dict["is_test_ReClor"],
                                      is_test_DREAM=data_loader_dict["is_test_DREAM"],
                                      is_test_StrategyQA=data_loader_dict["is_test_StrategyQA"],
                                      is_BoolQ=data_loader_dict["is_BoolQ"], is_CB=data_loader_dict["is_CB"],
                                      is_COPA=data_loader_dict["is_COPA"], is_MultiRC=data_loader_dict["is_MultiRC"],
                                      is_ReCoRD=data_loader_dict["is_ReCoRD"], is_MPE=data_loader_dict["is_MPE"],
                                      is_RTE=data_loader_dict["is_RTE"], is_WiC=data_loader_dict["is_WiC"],
                                      is_WSC=data_loader_dict["is_WSC"], is_CQA=data_loader_dict["is_CQA"],
                                      is_RACE=data_loader_dict["is_RACE"], is_SciTail=data_loader_dict["is_SciTail"],
                                      is_SNLI=data_loader_dict["is_SNLI"], is_MED=data_loader_dict["is_MED"],
                                      is_PIQA=data_loader_dict["is_PIQA"], is_SIQA=data_loader_dict["is_SIQA"],
                                      is_ReClor=data_loader_dict["is_ReClor"], is_DREAM=data_loader_dict["is_DREAM"],
                                      is_StrategyQA=data_loader_dict["is_StrategyQA"],
                                      error_function_cb=error_function_cb, error_function_mpe=error_function_mpe,
                                      max_seq_len=model.config.max_seq_len, ids_to_exclude=ids_to_exclude)

    data = dloader.get_generalQA_generator(batch_size=batch_size, head_strategy=head_strategy,
                                           dataset_strategy=dataset_strategy, shuffle_datasets=shuffle_, type=dataset,
                                           num_iterations=num_iterations, is_aux_toks=is_aux_toks)
    if strategy is not None:
        data = strategy.experimental_distribute_dataset(data)

    #pad_tok_id = tokenizer.decode("[PAD]")
    #assert isinstance(pad_tok_id, list) and len(pad_tok_id) == 1
    #pad_tok_id = pad_tok_id[0]
    pad_tok_id = 0 # this is hardcoded.

    model.bert.encoder.reset_interval_dict()

    for (idx, input_ids, attention_mask, token_type_ids) in data:
        example = {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids,
                   "pad_tok_id":pad_tok_id}
        model(example, training=False)

    if model.config.is_global_probe_dataset: model.bert.encoder.save_global()

    #train_class = GeneralTrainingClass(model=model, tokenizer=tokenizer, optimizer=None, loss_object=None,
    #                                   loss_object2=None, loss_function=None, strategy=strategy,
    #                                   checkpoint_path_folder="", error_function=error_function)
    #train_class.get_test_results(data, type_=type_, test_filepath=test_filepath)


if __name__ == "__main__":

    type__ = "bert_large_cased_gating_end_only_3_layers_512sl"
    #type__ = "bert_large_cased_original_512sl"
    config = BertModelConfigs()
    config_dict = config.return_config_dictionary(type_=type__)

    #filepaths_BoolQ, filepaths_CB, filepaths_COPA, filepaths_MultiRC, filepaths_ReCoRD, \
    #filepaths_RTE, filepaths_WiC, filepaths_WSC = get_dataset_filepaths()

    dataset_prefix = "MPE"
    dataset = dataset_prefix+"_test"
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
                        "filepath_SNLI": "/large_data/SNLI_1.0/snli_1.0/snli_1.0_dev.jsonl",
                        "filepath_MED": "/large_data/MED/MED/MED.tsv",
                        "filepath_PIQA": "/large_data/PIQA/valid.jsonl",
                        "filepath_labels_PIQA": "/large_data/PIQA/valid-labels.lst",
                        "filepath_SIQA": "/large_data/SIQA/socialiqa-train-dev/dev.jsonl",
                        "filepath_labels_SIQA": "/large_data/SIQA/socialiqa-train-dev/dev-labels.lst",
                        "filepath_ReClor": "/large_data/ReClor/val.json",
                        "filepath_DREAM": "/large_data/DREAM/dream/data/dev.json",
                        "filepath_StrategyQA": "/large_data/StrategyQA/strategyqa/data/strategyqa/dev.json",
                        "is_test_BoolQ": False,
                        "is_test_CB": False,
                        "is_test_COPA": False,
                        "is_test_MultiRC": False,
                        "is_test_ReCoRD": False,
                        "is_test_RTE": False,
                        "is_test_WiC": False,
                        "is_test_WSC": False,
                        "is_test_CQA": False,
                        "is_test_RACE": False,
                        "is_test_SciTail": False,
                        "is_test_MPE": True,
                        "is_test_SNLI": False,
                        "is_test_MED": False,
                        "is_test_PIQA": False,
                        "is_test_SIQA": False,
                        "is_test_ReClor": False,
                        "is_test_DREAM": False,
                        "is_test_StrategyQA": False,
                        "is_BoolQ": False,
                        "is_CB": False,
                        "is_COPA": False,
                        "is_MultiRC": False,
                        "is_ReCoRD": False,
                        "is_RTE": False,
                        "is_WiC": False,
                        "is_WSC": False,
                        "is_CQA": False,
                        "is_RACE": False,
                        "is_SciTail": False,
                        "is_MPE": True,
                        "is_SNLI": False,
                        "is_MED": False,
                        "is_PIQA": False,
                        "is_SIQA": False,
                        "is_ReClor": False,
                        "is_DREAM": False,
                        "is_StrategyQA": False,
                        # "max_seq_len": config_dict["max_seq_len"],
                        "is_train": False,
                        "is_val": True,
                        "is_test": False}

    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    batch_size = 30*GPUS_AVAILABLE
    strategy = None

    config_dict["is_diagnostics"] = False
    config_dict["max_seq_len"] = 510

    is_global_probe_dataset = True
    global_filepath = ""
    is_qualitative_probe = False
    qualitative_filepath = ""
    is_global_before_and_after = False
    global_before_after_filepath = ""

    #exp=3
    num_aux_toks=2
    start_max_seq_len = 5-num_aux_toks
    updated_max_seq_len = 510
    type_ = "gating-end"
    for exp in range(1,4):
        for iteration in [200000]:
            correct_ids, incorrect_ids = None, None
            for correct_false_bool in [True, False]: # True means correct predictions; False means incorrect predictions.

                pretrained = "/data/kkno604/NGT_experiments_updated/general_experiments/gating-end/" \
                             "exp"+str(exp)+"/Checkpoints/iteration"+str(iteration)+"/"
                # test_save_filepath isn't utilised.
                test_save_filepath = "Doesn't matter"

                prediction_filepath = ""
                #ids_to_exclude = None
                if correct_ids is None or incorrect_ids is None:
                    if dataset_prefix == "BoolQ":
                        prediction_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/" \
                                "exp"+str(exp)+"/Results/get_results_in_domain/BoolQ/prediction_files/iteration"+str(iteration)+".jsonl"
                        correct_ids, incorrect_ids = BoolQ_return_correct_incorrect_ids(pred_file=prediction_filepath) # answer_file is correct here already for our purposes.
                    elif dataset_prefix == "CB":
                        prediction_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/" \
                                "exp"+str(exp)+"/Results/get_results_in_domain/CB/prediction_files/iteration"+str(iteration)+".jsonl"
                        correct_ids, incorrect_ids = CB_return_correct_incorrect_ids(pred_file=prediction_filepath)
                    elif dataset_prefix == "COPA":
                        prediction_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/" \
                                "exp"+str(exp)+"/Results/get_results_in_domain/COPA/prediction_files/iteration"+str(iteration)+".jsonl"
                        correct_ids, incorrect_ids = COPA_return_correct_incorrect_ids(pred_file=prediction_filepath)
                    elif dataset_prefix == "CQA":
                        prediction_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/" \
                                "exp"+str(exp)+"/Results/get_results_in_domain/CQA/prediction_files/iteration"+str(iteration)+".jsonl"
                        correct_ids, incorrect_ids = CQA_return_correct_incorrect_ids(pred_file=prediction_filepath)
                    elif dataset_prefix == "MPE":
                        prediction_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/" \
                                "exp"+str(exp)+"/Results/get_results_in_domain/MPE/prediction_files/iteration"+str(iteration)+".jsonl"
                        correct_ids, incorrect_ids = MPE_return_correct_incorrect_ids(pred_file=prediction_filepath)
                    elif dataset_prefix == "MultiRC":
                        prediction_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/" \
                                "exp"+str(exp)+"/Results/get_results_in_domain/MultiRC/prediction_files/iteration"+str(iteration)+".jsonl"
                        correct_ids, incorrect_ids = MultiRC_return_correct_incorrect_ids(pred_file=prediction_filepath)
                    elif dataset_prefix == "RACE":
                        prediction_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/" \
                                "exp"+str(exp)+"/Results/get_results_in_domain/RACE/prediction_files/iteration"+str(iteration)+".jsonl"
                        correct_ids, incorrect_ids = RACE_return_correct_incorrect_ids(pred_file=prediction_filepath)
                        #print(f"correct_ids: {correct_ids}\nincorrect_ids: {incorrect_ids}")
                    elif dataset_prefix == "ReCoRD":
                        prediction_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/" \
                                "exp"+str(exp)+"/Results/get_results_in_domain/ReCoRD/prediction_files/iteration"+str(iteration)+".jsonl"
                        correct_ids, incorrect_ids = ReCoRD_return_correct_incorrect_ids(pred_file=prediction_filepath)
                    elif dataset_prefix == "RTE":
                        prediction_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/" \
                                "exp"+str(exp)+"/Results/get_results_in_domain/RTE/prediction_files/iteration"+str(iteration)+".jsonl"
                        correct_ids, incorrect_ids = RTE_return_correct_incorrect_ids(pred_file=prediction_filepath)
                    elif dataset_prefix == "SciTail":
                        prediction_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/" \
                                "exp"+str(exp)+"/Results/get_results_in_domain/SciTail/prediction_files/iteration"+str(iteration)+".jsonl"
                        correct_ids, incorrect_ids = SciTail_return_correct_incorrect_ids(pred_file=prediction_filepath)
                    elif dataset_prefix == "WiC":
                        prediction_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/" \
                                "exp"+str(exp)+"/Results/get_results_in_domain/WiC/prediction_files/iteration"+str(iteration)+".jsonl"
                        correct_ids, incorrect_ids = WiC_return_correct_incorrect_ids(pred_file=prediction_filepath)
                    elif dataset_prefix == "WSC":
                        prediction_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/"+type_+"/" \
                                "exp"+str(exp)+"/Results/get_results_in_domain/WSC/prediction_files/iteration"+str(iteration)+".jsonl"
                        correct_ids, incorrect_ids = WSC_return_correct_incorrect_ids(pred_file=prediction_filepath)
                    else: raise Exception(f"Invalid dataset_prefix: {dataset_prefix}!")

                if correct_false_bool: # True meaning correct; we exclude correct ids and only process incorrect ids
                    global_filepath = "/data/kkno604/NGT_experiments_updated/neuromodulation_probe/global_incorrect/"+dataset_prefix+"/" \
                                      ""+dataset_prefix+"_global_iteration"+str(iteration)+"test_pad_works_exp"+str(exp)+".jsonl"
                else: # False meaning incorrect; we exclude incorrect ids and only process correct ids
                    global_filepath = "/data/kkno604/NGT_experiments_updated/neuromodulation_probe/global_correct/" + dataset_prefix + "/" \
                                       "" + dataset_prefix + "_global_iteration" + str(iteration) + "test_pad_works_exp" + str(exp) + ".jsonl"

                get_nm_probe(config_dict=config_dict, data_loader_dict=data_loader_dict, strategy=strategy,
                             tokenizer=tokenizer, batch_size=batch_size, pretrained=pretrained,
                             shuffle_=False, type_=type_, dataset=dataset, test_filepath=test_save_filepath,
                             error_function_cb="sigmoid", error_function_mpe="sigmoid", error_function="sigmoid",
                             head_strategy="uniform", dataset_strategy="uniform", num_iterations=200000,
                             is_aux_toks=True, num_aux_toks=num_aux_toks, start_max_seq_len=start_max_seq_len,
                             updated_max_seq_len=updated_max_seq_len, is_global_probe_dataset=is_global_probe_dataset,
                             global_filepath=global_filepath, is_qualitative_probe=is_qualitative_probe,
                             qualitative_filepath=qualitative_filepath, is_global_before_and_after=is_global_before_and_after,
                             global_before_after_filepath=global_before_after_filepath,
                             ids_to_exclude=correct_ids if correct_false_bool else incorrect_ids)
