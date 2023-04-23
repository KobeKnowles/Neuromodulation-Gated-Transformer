import tensorflow as tf
import random

import transformers
from transformers import BertConfig, TFBertModel, BertTokenizer, AdamWeightDecay

import sys
sys.path.append("../..")

from data_loaders.general_data_loader import *
from data_loaders.BoolQ import *
from training_files.parent_training_class import *

def general_training_function(config_dict, data_loader_dict, strategy, learning_rate, learning_rate_starting_value,
                              learning_rate_end_value, tokenizer, batch_size, loss_object, loss_function, pretrained,
                              tf_seed, random_seed, epoch_start, epoch_end, train_save_filepath, val_save_filepath,
                              checkpoint_save_filepath, save_end_epoch, print_every_iterations, dataset, num_GPUs,
                              train_shuffle, val_shuffle, decay_steps, decay_function, error_function:str="sigmoid"):

    tf.random.set_seed(tf_seed)
    random.seed(random_seed) # for reporducibility if the data is shuffled.

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
                                is_diagnostics=config_dict["is_diagnostics"])
            model = TFBertModel.from_pretrained(pretrained, config=config)
            optimizer = AdamWeightDecay(learning_rate, beta_1=0.9, beta_2=0.999, weight_decay_rate=0.01)
    else:
        config = BertConfig(vocab_size=config_dict["vocab_size"], hidden_size=config_dict["hidden_size"],
                            num_hidden_layers=config_dict["num_hidden_layers"],
                            num_attention_heads=config_dict["num_attention_heads"],
                            intermediate_size=config_dict["intermediate_size"], hidden_act=config_dict["hidden_act"],
                            hidden_dropout_prob=config_dict["hidden_dropout_prob"],
                            attention_probs_dropout_prob=config_dict["attention_probs_dropout_prob"],
                            max_position_embeddings=config_dict["max_position_embeddings"],
                            type_vocab_size=config_dict["type_vocab_size"],
                            initializer_range=config_dict["initializer_range"],
                            layer_norm_eps=config_dict["layer_norm_eps"],
                            pad_token_id=config_dict["pad_token_id"],
                            position_embedding_type=config_dict["position_embedding_type"],
                            use_cache=config_dict["use_cache"], classifier_dropout=config_dict["classifier_dropout"],
                            gating_block_num_layers=config_dict["gating_block_num_layers"],
                            gating_block_end=config_dict["gating_block_end"],
                            gating_block_end_position=config_dict["gating_block_end_position"],
                            gating_block_middle=config_dict["gating_block_middle"],
                            gating_block_middle_position=config_dict["gating_block_middle_position"],
                            gating_block_start=config_dict["gating_block_start"],
                            gating_block_start_position=config_dict["gating_block_start_position"],
                            nm_gating=config_dict["nm_gating"],
                            cls_dense_layer_number_of_options=config_dict["cls_dense_layer_number_of_options"],
                            is_diagnostics=config_dict["is_diagnostics"])
        model = TFBertModel.from_pretrained(pretrained, config=config)
        optimizer = AdamWeightDecay(learning_rate, beta_1=0.9, beta_2=0.999, weight_decay_rate=0.01)

    data_dict = {}
    if data_loader_dict["is_train"]:
        dloader_train = general_data_loader(filepath_BoolQ=data_loader_dict["filepath_BoolQ"]["BoolQ_train"],
                                            filepath_CB=data_loader_dict["filepath_CB"]["CB_train"],
                                            filepath_COPA=data_loader_dict["filepath_COPA"]["COPA_train"],
                                            filepath_MultiRC=data_loader_dict["filepath_MultiRC"]["MultiRC_train"],
                                            filepath_ReCoRD=data_loader_dict["filepath_ReCoRD"]["ReCoRD_train"],
                                            filepath_RTE=data_loader_dict["filepath_RTE"]["RTE_train"],
                                            filepath_WiC=data_loader_dict["filepath_WiC"]["WiC_train"],
                                            filepath_WSC=data_loader_dict["filepath_WSC"]["WSC_train"],
                                            tokenizer=tokenizer,
                                            is_test_BoolQ=data_loader_dict["is_test_BoolQ"],
                                            is_test_CB=data_loader_dict["is_test_CB"],
                                            is_test_COPA=data_loader_dict["is_test_COPA"],
                                            is_test_MultiRC=data_loader_dict["is_test_MultiRC"],
                                            is_test_ReCoRD=data_loader_dict["is_test_ReCoRD"],
                                            is_test_RTE=data_loader_dict["is_test_RTE"],
                                            is_test_WiC=data_loader_dict["is_test_WiC"],
                                            is_test_WSC=data_loader_dict["is_test_WSC"],
                                            is_BoolQ=data_loader_dict["is_BoolQ"],
                                            is_CB=data_loader_dict["is_CB"],
                                            is_COPA=data_loader_dict["is_COPA"],
                                            is_MultiRC=data_loader_dict["is_MultiRC"],
                                            is_ReCoRD=data_loader_dict["is_ReCoRD"],
                                            is_RTE=data_loader_dict["is_RTE"],
                                            is_WiC=data_loader_dict["is_WiC"],
                                            is_WSC=data_loader_dict["is_WSC"],
                                            max_seq_len=config_dict["max_seq_len"],
                                            error_function_cb=error_function)
        print(f"{dataset} training set loaded!")
        data_dict["train"] = dloader_train.get_generators(batch_size=batch_size, shuffle=train_shuffle, type=dataset)
        if strategy is not None:
            data_dict["train"] = strategy.experimental_distribute_dataset(data_dict["train"])

    if data_loader_dict["is_val"]:
        dloader_val = general_data_loader(filepath_BoolQ=data_loader_dict["filepath_BoolQ"]["BoolQ_val"],
                                            filepath_CB=data_loader_dict["filepath_CB"]["CB_val"],
                                            filepath_COPA=data_loader_dict["filepath_COPA"]["COPA_val"],
                                            filepath_MultiRC=data_loader_dict["filepath_MultiRC"]["MultiRC_val"],
                                            filepath_ReCoRD=data_loader_dict["filepath_ReCoRD"]["ReCoRD_val"],
                                            filepath_RTE=data_loader_dict["filepath_RTE"]["RTE_val"],
                                            filepath_WiC=data_loader_dict["filepath_WiC"]["WiC_val"],
                                            filepath_WSC=data_loader_dict["filepath_WSC"]["WSC_val"],
                                            tokenizer=tokenizer,
                                            is_test_BoolQ=data_loader_dict["is_test_BoolQ"],
                                            is_test_CB=data_loader_dict["is_test_CB"],
                                            is_test_COPA=data_loader_dict["is_test_COPA"],
                                            is_test_MultiRC=data_loader_dict["is_test_MultiRC"],
                                            is_test_ReCoRD=data_loader_dict["is_test_ReCoRD"],
                                            is_test_RTE=data_loader_dict["is_test_RTE"],
                                            is_test_WiC=data_loader_dict["is_test_WiC"],
                                            is_test_WSC=data_loader_dict["is_test_WSC"],
                                            is_BoolQ=data_loader_dict["is_BoolQ"],
                                            is_CB=data_loader_dict["is_CB"],
                                            is_COPA=data_loader_dict["is_COPA"],
                                            is_MultiRC=data_loader_dict["is_MultiRC"],
                                            is_ReCoRD=data_loader_dict["is_ReCoRD"],
                                            is_RTE=data_loader_dict["is_RTE"],
                                            is_WiC=data_loader_dict["is_WiC"],
                                            is_WSC=data_loader_dict["is_WSC"],
                                            max_seq_len=config_dict["max_seq_len"],
                                            error_function_cb=error_function)
        print(f"{dataset} validation set loaded!")
        data_dict["val"] = dloader_val.get_generators(batch_size=batch_size, shuffle=val_shuffle, type=dataset)
        if strategy is not None:
            data_dict["val"] = strategy.experimental_distribute_dataset(data_dict["val"])

    assert len(data_dict.keys()) > 0, f"No dataset is loaded; make sure is_train and is_val is set to True!"

    train_class = GeneralTrainingClass(model=model, tokenizer=tokenizer, optimizer=optimizer, loss_object=loss_object,
                                       loss_function=loss_function, strategy=strategy,
                                       checkpoint_path_folder=checkpoint_save_filepath, error_function=error_function)

    train_class.training_epoch(epoch_start=epoch_start, epoch_end=epoch_end, data_dict=data_dict,
                               train_save_filepath=train_save_filepath, val_save_filepath=val_save_filepath,
                               config_dict=config_dict, dataset=dataset, tf_seed=tf_seed, random_seed=random_seed,
                               batch_size=batch_size, num_GPUs=num_GPUs, learning_rate_start_value=learning_rate_starting_value,
                               learning_rate_end_value=learning_rate_end_value, save_end_epoch=save_end_epoch,
                               print_every_iterations=print_every_iterations, decay_steps=decay_steps,
                               decay_function=decay_function)