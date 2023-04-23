import copy
import os

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
GPUS_AVAILABLE = 1

# import tensorflow.python.framework.ops

import sys

sys.path.append("..")

import tensorflow as tf
import numpy as np

import logging

# logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

print(f"\nTensorflow version: {tf.__version__}\n")

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import transformers
from transformers import BertConfig, TFBertModel, BertTokenizer

# import dataloaders below
from data_loaders.BoolQ import *

if __name__ == "__main__":

    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    config = BertConfig(vocab_size=28996, hidden_size=768, num_hidden_layers=12,
                        num_attention_heads=12, intermediate_size=3072, hidden_act="gelu",
                        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1, max_position_embeddings=512,
                        type_vocab_size=2, initializer_range=0.02, layer_norm_eps=1e-12,
                        pad_token_id=0, position_embedding_type="absolute", use_cache=True,
                        classifier_dropout=None, gating_block_num_layers=3, gating_block_end=True,
                        gating_block_end_position=9, gating_block_middle=True, gating_block_middle_position=6,
                        gating_block_start=True, gating_block_start_position=3, nm_gating=True,
                        cls_dense_layer_number_of_options=2, is_diagnostics=False)
    model_no_changes = TFBertModel(config=config)

    model_load_weights = copy.deepcopy(model_no_changes)
    ckpt = tf.train.Checkpoint(model=model_load_weights)
    ckpt.restore("/large_data/BERT_checkpoints/cased_L-12_H-768_A-12/bert_model_ckpt")

    is_test = False
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    dataLoader = BoolQLoader(filepath="/large_data/SuperGlue/BoolQ/train.jsonl",
                             tokenizer=tokenizer,
                             is_test=is_test)

    if not is_test:
        _, example, _ = next(dataLoader(batch_size=2))
    else:
        _, example = next(dataLoader(batch_size=2))

    # print(f"example: {example}")

    # model(input_ids=example['input_ids'], attention_mask=example['attention_mask'],
    # token_type_ids=example['token_type_ids'])
    out1, pred1 = model_no_changes(example) # the same sample for both.
    out2, pred2 = model_load_weights(example)

    print(f"pred_no_changes: {pred1}\n"
          f"pred_changes: {pred2}")

    if pred1.numpy().tolist() == pred2.numpy().tolist():
        print(f"No changes between the the two models!")
    else:
        print(f"The two models produce a different output!")

    #for layer in model_no_changes.layers:
    #    print(layer)

    #print(model_no_changes.trainable_variables())


