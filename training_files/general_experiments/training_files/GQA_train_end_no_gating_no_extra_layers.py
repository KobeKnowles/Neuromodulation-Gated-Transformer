import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
GPUS_AVAILABLE = 1

import sys
sys.path.append("../../..")

import tensorflow as tf

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

print(f"\nTensorflow version: {tf.__version__}\n")

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np

from training_files.parent_training_class import *

import random

from training_files.training_functions import *
from misc.BertConfigClasses import BertModelConfigs
from misc.learning_rate_schedules import *
from training_files.parent_training_class import loss_function_cls

from data_loaders.general_data_loader import *

from misc.get_dataset_filepaths import get_dataset_filepaths


if __name__ == "__main__":

    batch_size = 8*GPUS_AVAILABLE
    shuffle_datasets = True
    is_test = False
    num_iterations = 200000
    #strategy = tf.distribute.MirroredStrategy()
    strategy = None
    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")

    type_ = "bert_large_cased_original_512sl"
    pretrained = "bert-large-cased"
    config = BertModelConfigs()
    config_dict = config.return_config_dictionary(type_=type_)

    tf_seed = random.randint(0, sys.maxsize)
    random_seed = random.randint(0, sys.maxsize)

    tf.random.set_seed(tf_seed)
    random.seed(random_seed)

    decay_steps = 190000
    warmup_steps = 9999 # 10000-1
    decay_function = "cosine_decay"
    learning_rate_starting_value = 0.000005
    learning_rate_end_value = 0
    learning_rate_linear_warmup_value = 0.00001
    learning_rate = CosineDecayLW(start_lr=learning_rate_starting_value, lower_bound_lr=learning_rate_end_value,
                                  upper_bound_lr=learning_rate_linear_warmup_value,
                                  warmup_steps=warmup_steps, decay_steps=decay_steps)
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction='none')
    loss_object2 = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction='none')
    # 9999, not 10000 warmup steps because the first step is zero and is the starting learning rate. We include the first step (step 0) when counting the warmup steps.
    # decay_steps+warmup_steps = 199999 in this instance not including the start step, which is correct when adding in step 0.

    num_aux_toks = 0
    config_dict["max_seq_len"] = 510 # note 512 b/c of the two auxiliary tokens; we can't change the max_position_embeddings,
    # it raises an error.

    model, optimizer = None, None
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
                                num_aux_toks=num_aux_toks, max_seq_len=5-num_aux_toks)
            model = TFBertModel.from_pretrained(pretrained, config=config)
            optimizer = AdamWeightDecay(learning_rate, beta_1=0.9, beta_2=0.98, weight_decay_rate=0.00001)
            # this is required because the model is built with a sequence length of 5; after it is built we set the max
            # sequence length back to what was originally intended (512).
            model.config.max_seq_len = config_dict["max_seq_len"]
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
                            num_aux_toks=num_aux_toks, max_seq_len=5-num_aux_toks)
        model = TFBertModel.from_pretrained(pretrained, config=config)
        optimizer = AdamWeightDecay(learning_rate, beta_1=0.9, beta_2=0.98, weight_decay_rate=0.00001)
        # this is required because the model is built with a sequence length of 5; after it is built we set the max
        # sequence length back to what was originally intended (512).
        model.config.max_seq_len = config_dict["max_seq_len"]

    dloader = general_data_loader(filepath_BoolQ="/large_data/SuperGlue/BoolQ/train.jsonl",
                                  filepath_CB="/large_data/SuperGlue/CB/train.jsonl",
                                  filepath_COPA="/large_data/SuperGlue/COPA/train.jsonl",
                                  filepath_MultiRC="/large_data/SuperGlue/MultiRC/train.jsonl",
                                  filepath_ReCoRD="/large_data/SuperGlue/ReCoRD/train.jsonl",
                                  filepath_RTE="/large_data/SuperGlue/RTE/train.jsonl",
                                  filepath_WiC="/large_data/SuperGlue/WiC/train.jsonl",
                                  filepath_WSC="/large_data/SuperGlue/WSC/train.jsonl",
                                  filepath_CQA="/large_data/CommonsenseQA/train_rand_split.jsonl",
                                  filepath_RACE_middle="/large_data/RACE/RACE/train/middle/",
                                  filepath_RACE_high="/large_data/RACE/RACE/train/high/",
                                  filepath_SciTail="/large_data/SciTail/SciTailV1.1/predictor_format/scitail_1.0_structure_train.jsonl",
                                  filepath_MPE="/large_data/MPE/MultiPremiseEntailment-master/data/MPE/mpe_train.txt",
                                  tokenizer=tokenizer,
                                  is_test_BoolQ=is_test, is_test_CB=is_test, is_test_COPA=is_test,
                                  is_test_MultiRC=is_test,
                                  is_test_ReCoRD=is_test, is_test_RTE=is_test, is_test_WiC=is_test,
                                  is_test_WSC=is_test,
                                  is_test_CQA=is_test, is_test_RACE=is_test, is_test_SciTail=is_test,
                                  is_test_MPE=is_test,
                                  is_BoolQ=True, is_CB=True, is_COPA=True, is_MultiRC=True, is_ReCoRD=True, is_MPE=True,
                                  is_RTE=True, is_WiC=True, is_WSC=True, is_CQA=True, is_RACE=True, is_SciTail=True,
                                  error_function_cb="sigmoid", error_function_mpe="sigmoid",
                                  max_seq_len=config_dict["max_seq_len"])
    print(f"training sets loaded!")
    data = dloader.get_generalQA_generator(batch_size, head_strategy="uniform", dataset_strategy="uniform",
                                           shuffle_datasets=shuffle_datasets, type="default",
                                           num_iterations=num_iterations, is_aux_toks=False)
    if strategy is not None:
        data = strategy.experimental_distribute_dataset(data)

    train_save_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/no-gating-no-extra-layers/exp3/Results/"
    checkpoint_save_filepath = "/data/kkno604/NGT_experiments_updated/general_experiments/no-gating-no-extra-layers/exp3/Checkpoints/"

    error_function="none" # this doesn't matter.
    train_class = GeneralTrainingClass(model=model, tokenizer=tokenizer, optimizer=optimizer,
                                       loss_object=loss_object, loss_object2=loss_object2,
                                       loss_function=loss_function_cls, strategy=strategy,
                                       checkpoint_path_folder=checkpoint_save_filepath,
                                       error_function=error_function)

    train_class.training_iteration(iterations=num_iterations, dataLoader=data,
                                   train_save_filepath=train_save_filepath, config_dict=config_dict,
                                   dataset="GQA_uniform", tf_seed=tf_seed, random_seed=random_seed,
                                   batch_size=batch_size, save_iterations=[50000, 100000, 150000, 200000],
                                   print_every_iterations=100, num_GPUs=GPUS_AVAILABLE,
                                   learning_rate_start_value=learning_rate_starting_value,
                                   learning_rate_end_value=learning_rate_end_value,
                                   learning_rate_linear_warmup_value=learning_rate_linear_warmup_value,
                                   decay_steps=decay_steps, warmup_steps=warmup_steps,
                                   decay_function="cosine", num_aux_toks=num_aux_toks,
                                   shuffle_datasets=shuffle_datasets)