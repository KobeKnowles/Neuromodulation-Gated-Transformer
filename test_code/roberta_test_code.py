import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
GPUS_AVAILABLE = 1

#import tensorflow.python.framework.ops

import sys
sys.path.append("..")

import tensorflow as tf
import numpy as np
import random

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

print(f"\nTensorflow version: {tf.__version__}\n")

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import transformers
from transformers import RobertaConfig, TFRobertaModel, RobertaTokenizer
from misc.RoBERTaConfigClasses import *
# import dataloaders below
from data_loaders.BoolQ import *


if __name__ == "__main__":

    type_ = "roberta_large_gating_end_only_3_layers_512sl"
    #type_ = "roberta_large_no_gating_end_only_3_layers_512sl"
    #type_ = "roberta_large_original_512sl"
    pretrained = "roberta-large"

    config_ = RoBERTaModelConfigs()
    config_dict = config_.return_config_dictionary(type_=type_)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    config = RobertaConfig(vocab_size=config_dict["vocab_size"], hidden_size=config_dict["hidden_size"],
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
                           gating_block_num_layers=config_dict["gating_block_num_layers"],
                           gating_block_end=config_dict["gating_block_end"],
                           gating_block_end_position=config_dict["gating_block_end_position"],
                           gating_block_middle=config_dict["gating_block_middle"],
                           gating_block_middle_position=config_dict["gating_block_middle_position"],
                           gating_block_start=config_dict["gating_block_start"],
                           gating_block_start_position=config_dict["gating_block_start_position"],
                           nm_gating=config_dict["nm_gating"],
                           cls_dense_layer_number_of_options=config_dict["cls_dense_layer_number_of_options"],
                           is_diagnostics=False,
                           max_seq_len=5,
                           num_aux_toks=0, multiple_heads=False)#config_dict["is_diagnostics"])
    model = TFRobertaModel.from_pretrained(pretrained, config=config)
    print(f"max_seq_len_before: {model.config.max_seq_len}")
    model.config.max_seq_len=512
    print(f"max_seq_len_after: {model.config.max_seq_len}")

    #model.from_pretrained("bert-base-cased")
    #tf.random.set_seed(8)
    #random.seed(4534)
    #model = TFBertModel.from_pretrained("bert-base-cased", config=config)
    #model.from_pretrained("/data/kkno604/NGT_experiments/Ablation/BoolQ/tmp_save_test/epoch1/")

    #ckpt = tf.train.Checkpoint(model=model)

    #ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, )
 
    #ckpt.restore("/large_data/BERT_checkpoints/cased_L-12_H-768_A-12/bert_model_ckpt")

    #model.save_pretrained("/data/kkno604/NGT_experiments/Ablation/BoolQ/tmp_save_test/epoch1")

    is_test = False
    dataLoader = BoolQLoader(filepath="/large_data/SuperGlue/BoolQ/train.jsonl",
                             tokenizer=tokenizer,
                             is_test=is_test, max_seq_len=512)
    dataLoader.shuffle=False

    if not is_test:
        _, example, _ = next(dataLoader(batch_size=2, is_roberta=True))
    else:
        _, example = next(dataLoader(batch_size=2, is_roberta=True))

    #print(f"example: {example}")

    #model(input_ids=example['input_ids'], attention_mask=example['attention_mask'],
          #token_type_ids=example['token_type_ids'])
    #print(f"\n\n\nexample: {example}\n\n\n")

    _, pred = model(example, head_num=4)
    #print(pred)

    #tf.random.set_seed(8)
    #random.seed(456468343)
    #model = TFBertModel.from_pretrained("bert-base-cased", config=config)
    # model.from_pretrained("/data/kkno604/NGT_experiments/Ablation/BoolQ/tmp_save_test/epoch1/")

    # ckpt = tf.train.Checkpoint(model=model)

    # ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, )

    # ckpt.restore("/large_data/BERT_checkpoints/cased_L-12_H-768_A-12/bert_model_ckpt")

    # model.save_pretrained("/data/kkno604/NGT_experiments/Ablation/BoolQ/tmp_save_test/epoch1")

    #is_test = False
    #dataLoader = BoolQLoader(filepath="/large_data/SuperGlue/BoolQ/train.jsonl",
    #                         tokenizer=tokenizer,
    #                         is_test=is_test)

    #if not is_test:
    #    _, example, _ = next(dataLoader(batch_size=2))
    #else:
    #    _, example = next(dataLoader(batch_size=2))

    # print(f"example: {example}")

    # model(input_ids=example['input_ids'], attention_mask=example['attention_mask'],
    # token_type_ids=example['token_type_ids'])
    #_, pred = model(example)
    #print(pred)

    #tf.random.set_seed(24)
    #random.seed(1516854)
    #model = TFBertModel.from_pretrained("bert-base-cased", config=config)
    # model.from_pretrained("/data/kkno604/NGT_experiments/Ablation/BoolQ/tmp_save_test/epoch1/")

    # ckpt = tf.train.Checkpoint(model=model)

    # ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, )

    # ckpt.restore("/large_data/BERT_checkpoints/cased_L-12_H-768_A-12/bert_model_ckpt")

    # model.save_pretrained("/data/kkno604/NGT_experiments/Ablation/BoolQ/tmp_save_test/epoch1")

    #is_test = False
    #dataLoader = BoolQLoader(filepath="/large_data/SuperGlue/BoolQ/train.jsonl",
    #                         tokenizer=tokenizer,
    #                         is_test=is_test)

    #if not is_test:
    #    _, example, _ = next(dataLoader(batch_size=2))
    #else:
    #    _, example = next(dataLoader(batch_size=2))

    # print(f"example: {example}")

    # model(input_ids=example['input_ids'], attention_mask=example['attention_mask'],
    # token_type_ids=example['token_type_ids'])
    #_, pred = model(example)
    #print(pred)

    #tf.random.set_seed(8)
    #random.seed(248)
    #model = TFBertModel.from_pretrained("bert-base-cased", config=config)
    # model.from_pretrained("/data/kkno604/NGT_experiments/Ablation/BoolQ/tmp_save_test/epoch1/")

    # ckpt = tf.train.Checkpoint(model=model)

    # ckpt_manager = tf.train.CheckpointManager(ckpt, ckpt_path, )

    # ckpt.restore("/large_data/BERT_checkpoints/cased_L-12_H-768_A-12/bert_model_ckpt")

    # model.save_pretrained("/data/kkno604/NGT_experiments/Ablation/BoolQ/tmp_save_test/epoch1")

    #is_test = False
    #dataLoader = BoolQLoader(filepath="/large_data/SuperGlue/BoolQ/train.jsonl",
    #                         tokenizer=tokenizer,
    #                         is_test=is_test)

    #if not is_test:
    #    _, example, _ = next(dataLoader(batch_size=2))
    #else:
    #    _, example = next(dataLoader(batch_size=2))

    # print(f"example: {example}")

    # model(input_ids=example['input_ids'], attention_mask=example['attention_mask'],
    # token_type_ids=example['token_type_ids'])
    #_, pred = model(example)
    #print(pred)

    #print(f"model.summary random: {model.summary()}")

    #model = model.from_pretrained("/data/kkno604/NGT_experiments/Ablation/BoolQ/tmp_save_test/epoch1/")
    #model.bert.config = config # need to re-apply the config as the gating params are overriden.
    #model.bert.encoder.config = config # i.e., during training the gating parts are not run and is_diagnostics doesnt' work.
    #print(model(example))
    #print(f"model.summary random: {model.summary()}")

    #model.save_pretrained("/data/kkno604/NGT_experiments/Ablation/BoolQ/tmp_save_test/epoch1")


    #while True:
    #    for _, example, _ in dataLoader(shuffle=False):
    #        print(model(example))

