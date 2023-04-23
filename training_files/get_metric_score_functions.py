import tensorflow as tf
import random

import transformers
from transformers import BertConfig, TFBertModel, BertTokenizer

import sys
sys.path.append("../..")

from data_loaders.general_data_loader import *
from data_loaders.BoolQ import *
from training_files.parent_training_class import *

def general_get_predictions(config_dict, data_loader_dict, strategy, tokenizer, batch_size, pretrained,
                            shuffle_=False, type_="_idx_example_test_helper", dataset="BoolQ", test_filepath="",
                            error_function="sigmoid"):

    assert data_loader_dict["is_test_BoolQ"] or data_loader_dict["is_test_CB"] or data_loader_dict["is_test_COPA"] \
                or data_loader_dict["is_test_MultiRC"] or data_loader_dict["is_test_ReCoRD"] \
                or data_loader_dict["is_test_RTE"] or data_loader_dict["is_test_WiC"] \
                or data_loader_dict["is_test_WSC"], f"One dataset's is_test should be set to True!"

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
                                is_diagnostics=config_dict["is_diagnostics"])
            model = TFBertModel.from_pretrained(pretrained, config=config)
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

    data=None
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
        data = dloader_train.get_generators(batch_size=batch_size, shuffle=train_shuffle, type=dataset)
        if strategy is not None:
            data = strategy.experimental_distribute_dataset(data)
    elif data_loader_dict["is_val"]:
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
        data = dloader_val.get_generators(batch_size=batch_size, shuffle=shuffle_, type=dataset)
        if strategy is not None:
            data = strategy.experimental_distribute_dataset(data)
    elif data_loader_dict["is_test"]:
        dloader_test = general_data_loader(filepath_BoolQ=data_loader_dict["filepath_BoolQ"]["BoolQ_test"],
                                          filepath_CB=data_loader_dict["filepath_CB"]["CB_test"],
                                          filepath_COPA=data_loader_dict["filepath_COPA"]["COPA_test"],
                                          filepath_MultiRC=data_loader_dict["filepath_MultiRC"]["MultiRC_test"],
                                          filepath_ReCoRD=data_loader_dict["filepath_ReCoRD"]["ReCoRD_test"],
                                          filepath_RTE=data_loader_dict["filepath_RTE"]["RTE_test"],
                                          filepath_WiC=data_loader_dict["filepath_WiC"]["WiC_test"],
                                          filepath_WSC=data_loader_dict["filepath_WSC"]["WSC_test"],
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
        data = dloader_test.get_generators(batch_size=batch_size, shuffle=shuffle_, type=dataset)
        if strategy is not None:
            data = strategy.experimental_distribute_dataset(data)


    train_class = GeneralTrainingClass(model=model, tokenizer=tokenizer, optimizer=None, loss_object=None,
                                       loss_function=None, strategy=strategy, checkpoint_path_folder="",
                                       error_function=error_function)
    train_class.get_test_results(data, type_=type_, test_filepath=test_filepath)

def GQA_get_predictions(config_dict, data_loader_dict, strategy, tokenizer, batch_size, pretrained,
                        shuffle_=False, type_="_idx_example_test_helper", dataset="BoolQ", test_filepath="",
                        error_function_cb="softmax", error_function_mpe="sigmoid", error_function="none",
                        head_strategy=None, dataset_strategy=None, num_iterations=None, is_aux_toks=True,
                        num_aux_toks: int=2, start_max_seq_len: int=3, updated_max_seq_len: int=510,
                        gqa:bool = False):

    if not gqa:
        assert data_loader_dict["is_test_BoolQ"] or data_loader_dict["is_test_CB"] or data_loader_dict["is_test_COPA"] \
                    or data_loader_dict["is_test_MultiRC"] or data_loader_dict["is_test_ReCoRD"] \
                    or data_loader_dict["is_test_RTE"] or data_loader_dict["is_test_WiC"] \
                    or data_loader_dict["is_test_WSC"] or data_loader_dict["is_test_MPE"]\
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
               or data_loader_dict["is_test_SIQA"] or data_loader_dict["is_test_ReClor"]\
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
                                num_aux_toks=num_aux_toks, max_seq_len=start_max_seq_len)
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
                            num_aux_toks=num_aux_toks, max_seq_len=start_max_seq_len)

        #model = TFBertModel(config=config)
        #model = model.from_pretrained(pretrained)
        model = TFBertModel.from_pretrained(pretrained, config=config, ignore_mismatched_sizes=True)
        #print(model.layers)

        model.config.max_seq_len = config_dict["max_seq_len"]
        assert config_dict["max_seq_len"] == updated_max_seq_len

    data=None
    dloader = None
    if not gqa: # this if else is for compatibility with old code before the GQA experiments.
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
                                      is_test_BoolQ=data_loader_dict["is_test_BoolQ"], is_test_CB=data_loader_dict["is_test_CB"],
                                      is_test_COPA=data_loader_dict["is_test_COPA"], is_test_MultiRC=data_loader_dict["is_test_MultiRC"],
                                      is_test_ReCoRD=data_loader_dict["is_test_ReCoRD"], is_test_RTE=data_loader_dict["is_test_RTE"],
                                      is_test_WiC=data_loader_dict["is_test_WiC"], is_test_WSC=data_loader_dict["is_test_WSC"],
                                      is_test_CQA=data_loader_dict["is_test_CQA"], is_test_RACE=data_loader_dict["is_test_RACE"],
                                      is_test_SciTail=data_loader_dict["is_test_SciTail"], is_test_MPE=data_loader_dict["is_test_MPE"],
                                      is_BoolQ=data_loader_dict["is_BoolQ"], is_CB=data_loader_dict["is_CB"],
                                      is_COPA=data_loader_dict["is_COPA"], is_MultiRC=data_loader_dict["is_MultiRC"],
                                      is_ReCoRD=data_loader_dict["is_ReCoRD"], is_MPE=data_loader_dict["is_MPE"],
                                      is_RTE=data_loader_dict["is_RTE"], is_WiC=data_loader_dict["is_WiC"],
                                      is_WSC=data_loader_dict["is_WSC"], is_CQA=data_loader_dict["is_CQA"],
                                      is_RACE=data_loader_dict["is_RACE"], is_SciTail=data_loader_dict["is_SciTail"],
                                      error_function_cb=error_function_cb, error_function_mpe=error_function_mpe,
                                      max_seq_len=model.config.max_seq_len)
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
                                      max_seq_len=model.config.max_seq_len)

    data = dloader.get_generalQA_generator(batch_size=batch_size, head_strategy=head_strategy,
                                           dataset_strategy=dataset_strategy, shuffle_datasets=shuffle_, type=dataset,
                                           num_iterations=num_iterations, is_aux_toks=is_aux_toks)
    if strategy is not None:
        data = strategy.experimental_distribute_dataset(data)


    #train_class = GeneralTrainingClass(model=model, tokenizer=tokenizer, optimizer=None, loss_object=None,
    #                                   loss_function=None, strategy=strategy, checkpoint_path_folder="",
    #                                   error_function=error_function)
    train_class = GeneralTrainingClass(model=model, tokenizer=tokenizer, optimizer=None, loss_object=None,
                                       loss_object2=None, loss_function=None, strategy=strategy,
                                       checkpoint_path_folder="", error_function=error_function)
    train_class.get_test_results(data, type_=type_, test_filepath=test_filepath)


