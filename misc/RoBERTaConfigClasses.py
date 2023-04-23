class RoBERTaModelConfigs(object):

    def __init__(self):

        #https://huggingface.co/bert-base-cased/blob/main/config.json for bert-base-cased configuration.

        self.roberta_large_original_512sl = {
            "attention_probs_dropout_prob": 0.1, "bos_token_id": 0, "eos_token_id": 2, "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1, "hidden_size": 1024, "initializer_range": 0.02, "intermediate_size": 4096,
            "layer_norm_eps": 1e-05, "max_position_embeddings": 514, "model_type": "roberta", "num_attention_heads": 16,
            "num_hidden_layers": 24, "pad_token_id": 1, "type_vocab_size": 1, "vocab_size": 50265, "max_seq_len": 512,
            "position_embedding_type":"absolute", "cls_dense_layer_number_of_options": 1, "gating_block_num_layers": 3,
            "gating_block_end": False, "gating_block_end_position": 21, "gating_block_middle": False,
            "gating_block_middle_position": 12, "gating_block_start": False, "gating_block_start_position": 3,
            "nm_gating": False, "is_diagnostics": False,
        }

        self.roberta_large_no_gating_end_only_3_layers_512sl = {
            "attention_probs_dropout_prob": 0.1, "bos_token_id": 0, "eos_token_id": 2, "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1, "hidden_size": 1024, "initializer_range": 0.02, "intermediate_size": 4096,
            "layer_norm_eps": 1e-05, "max_position_embeddings": 514, "model_type": "roberta", "num_attention_heads": 16,
            "num_hidden_layers": 24, "pad_token_id": 1, "type_vocab_size": 1, "vocab_size": 50265, "max_seq_len": 512,
            "position_embedding_type": "absolute", "cls_dense_layer_number_of_options": 1, "gating_block_num_layers": 3,
            "gating_block_end": True, "gating_block_end_position": 21, "gating_block_middle": False,
            "gating_block_middle_position": 12, "gating_block_start": False, "gating_block_start_position": 3,
            "nm_gating": False, "is_diagnostics": False,
        }

        self.roberta_large_gating_end_only_3_layers_512sl = {
            "attention_probs_dropout_prob": 0.1, "bos_token_id": 0, "eos_token_id": 2, "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1, "hidden_size": 1024, "initializer_range": 0.02, "intermediate_size": 4096,
            "layer_norm_eps": 1e-05, "max_position_embeddings": 514, "model_type": "roberta", "num_attention_heads": 16,
            "num_hidden_layers": 24, "pad_token_id": 1, "type_vocab_size": 1, "vocab_size": 50265, "max_seq_len": 512,
            "position_embedding_type": "absolute", "cls_dense_layer_number_of_options": 1, "gating_block_num_layers": 3,
            "gating_block_end": True, "gating_block_end_position": 21, "gating_block_middle": False,
            "gating_block_middle_position": 12, "gating_block_start": False, "gating_block_start_position": 3,
            "nm_gating": True, "is_diagnostics": False,
        }

    def return_config_dictionary(self, type_):

        # base models here.
        if type_ == "roberta_large_original_512sl": return self.roberta_large_original_512sl
        elif type_ == "roberta_large_no_gating_end_only_3_layers_512sl": return self.roberta_large_no_gating_end_only_3_layers_512sl
        elif type_ == "roberta_large_gating_end_only_3_layers_512sl": return self.roberta_large_gating_end_only_3_layers_512sl






