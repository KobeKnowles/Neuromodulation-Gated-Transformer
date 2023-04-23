import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np

import sys
sys.path.append("..")

from data_loaders.BoolQ import *
from data_loaders.CB import *
from data_loaders.SNLI import *
from data_loaders.COPA import *
from data_loaders.MultiRC import *
from data_loaders.ReCoRD import *
from data_loaders.RTE import *
from data_loaders.WiC import *
from data_loaders.WSC import *
from data_loaders.CQA import *
from data_loaders.RACE import *
from data_loaders.SciTail import *
from data_loaders.MPE import *
from data_loaders.MED import *
from data_loaders.PIQA import *
from data_loaders.SIQA import *
from data_loaders.ReClor import *
from data_loaders.DREAM import *
from data_loaders.StrategyQA import *

from transformers import BertTokenizer

import random


class general_data_loader(object):
    '''

    '''
    def __init__(self, filepath_BoolQ: str, filepath_CB: str, filepath_COPA: str, filepath_MultiRC: str,
                 filepath_ReCoRD: str, filepath_RTE: str, filepath_WiC: str, filepath_WSC: str, tokenizer,
                 is_test_BoolQ: bool, is_test_CB: bool, is_test_COPA: bool, is_test_MultiRC: bool,
                 is_test_ReCoRD: bool, is_test_RTE: bool, is_test_WiC: bool, is_test_WSC: bool,
                 is_BoolQ: bool, is_CB: bool, is_COPA: bool, is_MultiRC: bool,
                 is_ReCoRD: bool, is_RTE: bool, is_WiC: bool, is_WSC: bool, max_seq_len:int=512,
                 filepath_CQA: str="", is_test_CQA: bool=False, is_CQA: bool=False,
                 filepath_RACE_middle: str = "", filepath_RACE_high: str = "",
                 is_test_RACE: bool = False, is_RACE: bool = False,
                 filepath_SciTail: str = "", is_test_SciTail: bool = False, is_SciTail: bool = False,
                 filepath_MPE: str = "", is_test_MPE: bool = False, is_MPE: bool = False,
                 filepath_SNLI: str = "", is_test_SNLI: bool = False, is_SNLI: bool = False,
                 filepath_MED: str = "", is_test_MED: bool = False, is_MED: bool = False,
                 filepath_PIQA: str = "", filepath_labels_PIQA: str = "", is_test_PIQA: bool = False, is_PIQA: bool = False,
                 filepath_SIQA: str = "", filepath_labels_SIQA: str = "", is_test_SIQA: bool = False, is_SIQA: bool = False,
                 filepath_ReClor: str = "", is_test_ReClor: bool = False, is_ReClor: bool = False,
                 filepath_DREAM: str = "", is_test_DREAM: bool = False, is_DREAM: bool = False,
                 filepath_StrategyQA: str = "", is_test_StrategyQA: bool = False, is_StrategyQA: bool = False,
                 error_function_cb:str="sigmoid", error_function_mpe:str="sigmoid",
                 is_token_type_ids: bool=True, is_diagnostics: bool=False, ids_to_exclude=None):
        '''

        '''
        # filepaths for each dataset.
        self.filepath_BoolQ = filepath_BoolQ
        self.filepath_CB = filepath_CB
        self.filepath_COPA = filepath_COPA
        self.filepath_MultiRC = filepath_MultiRC
        self.filepath_ReCoRD = filepath_ReCoRD
        self.filepath_RTE = filepath_RTE
        self.filepath_WiC = filepath_WiC
        self.filepath_WSC = filepath_WSC
        self.filepath_CQA = filepath_CQA
        self.filepath_RACE_middle = filepath_RACE_middle
        self.filepath_RACE_high = filepath_RACE_high
        self.filepath_SciTail = filepath_SciTail
        self.filepath_MPE = filepath_MPE
        self.filepath_SNLI = filepath_SNLI
        self.filepath_MED = filepath_MED
        self.filepath_PIQA = filepath_PIQA
        self.filepath_labels_PIQA = filepath_labels_PIQA
        self.filepath_SIQA = filepath_SIQA
        self.filepath_labels_SIQA = filepath_labels_SIQA
        self.filepath_ReClor = filepath_ReClor
        self.filepath_DREAM = filepath_DREAM
        self.filepath_StrategyQA = filepath_StrategyQA

        # whether or not the dataloader is to return the labels.
        self.is_test_BoolQ = is_test_BoolQ
        self.is_test_CB = is_test_CB
        self.is_test_COPA = is_test_COPA
        self.is_test_MultiRC = is_test_MultiRC
        self.is_test_ReCoRD = is_test_ReCoRD
        self.is_test_RTE = is_test_RTE
        self.is_test_WiC = is_test_WiC
        self.is_test_WSC = is_test_WSC
        self.is_test_CQA = is_test_CQA
        self.is_test_RACE = is_test_RACE
        self.is_test_SciTail = is_test_SciTail
        self.is_test_MPE = is_test_MPE
        self.is_test_SNLI = is_test_SNLI
        self.is_test_MED = is_test_MED
        self.is_test_PIQA = is_test_PIQA
        self.is_test_SIQA = is_test_SIQA
        self.is_test_ReClor = is_test_ReClor
        self.is_test_DREAM = is_test_DREAM
        self.is_test_StrategyQA = is_test_StrategyQA

        # if we are utilsing a certain dataset.
        self.is_BoolQ = is_BoolQ
        self.is_CB = is_CB
        self.is_COPA = is_COPA
        self.is_MultiRC = is_MultiRC
        self.is_ReCoRD = is_ReCoRD
        self.is_RTE = is_RTE
        self.is_WiC = is_WiC
        self.is_WSC = is_WSC
        self.is_CQA = is_CQA
        self.is_RACE = is_RACE
        self.is_SciTail = is_SciTail
        self.is_MPE = is_MPE
        self.is_SNLI = is_SNLI
        self.is_MED = is_MED
        self.is_PIQA = is_PIQA
        self.is_SIQA = is_SIQA
        self.is_ReClor = is_ReClor
        self.is_DREAM = is_DREAM
        self.is_StrategyQA = is_StrategyQA

        self.max_seq_len = max_seq_len

        self.tokenizer = tokenizer

        self.is_diagnostics = is_diagnostics

        self.is_token_type_ids = is_token_type_ids
        assert is_token_type_ids, f"is_token_type_ids must be set to True; False is not supported (TODO: need to modify" \
                                  f" the code to support it by eliminating token_type_ids from being yielded by the " \
                                  f"generators)."

        self.BoolQ_counter = 0
        self.CB_counter = 0
        self.COPA_counter = 0
        self.MultiRC_counter = 0
        self.ReCoRD_counter = 0
        self.RTE_counter = 0
        self.WiC_counter = 0
        self.WSC_counter = 0
        self.CQA_counter = 0
        self.RACE_counter = 0
        self.SciTail_counter = 0
        self.MPE_counter = 0

        self.head1_counter = 0
        self.head2_counter = 0
        self.head3_counter = 0
        self.head4_counter = 0
        self.head5_counter = 0
        self.head6_counter = 0


        assert self.is_BoolQ or self.is_CB or self.is_COPA or self.is_MultiRC or self.is_ReCoRD \
            or self.is_RTE or self.is_WiC or self.is_WSC or self.is_CQA or self.is_RACE or self.is_SciTail\
            or self.is_MPE or self.is_SNLI or self.is_MED or self.is_PIQA or self.is_SIQA or self.is_ReClor\
            or self.is_DREAM or self.is_StrategyQA, f"One data loader needs to be set to True; none were set to True."

        if self.is_BoolQ: self.BoolQ_loader = BoolQLoader(filepath=self.filepath_BoolQ, tokenizer=self.tokenizer,
                                                          is_test=self.is_test_BoolQ, max_seq_len=self.max_seq_len,
                                                          is_token_type_ids=is_token_type_ids, ids_to_exclude=ids_to_exclude)
        if self.is_CB: self.CB_loader = CBLoader(filepath=self.filepath_CB, tokenizer=self.tokenizer,
                                                 is_test=self.is_test_CB, max_seq_len=self.max_seq_len,
                                                 mode=error_function_cb, is_token_type_ids=is_token_type_ids, ids_to_exclude=ids_to_exclude)
        if self.is_MultiRC: self.MultiRC_loader = MultiRCLoader(filepath=self.filepath_MultiRC, tokenizer=self.tokenizer,
                                                 is_test=self.is_test_MultiRC, max_seq_len=self.max_seq_len,
                                                                is_token_type_ids=is_token_type_ids, ids_to_exclude=ids_to_exclude)
        if self.is_ReCoRD: self.ReCoRD_loader = ReCoRDLoader(filepath=self.filepath_ReCoRD, tokenizer=self.tokenizer,
                                                             is_test=self.is_test_ReCoRD, max_seq_len=self.max_seq_len,
                                                             is_token_type_ids=is_token_type_ids, ids_to_exclude=ids_to_exclude)
        if self.is_COPA: self.COPA_loader = COPALoader(filepath=self.filepath_COPA, tokenizer=self.tokenizer,
                                                       is_test=self.is_test_COPA, max_seq_len=self.max_seq_len,
                                                       is_token_type_ids=is_token_type_ids, ids_to_exclude=ids_to_exclude)
        if self.is_RTE: self.RTE_loader = RTELoader(filepath=self.filepath_RTE, tokenizer=self.tokenizer,
                                                    is_test=self.is_test_RTE, max_seq_len=self.max_seq_len,
                                                    is_token_type_ids=is_token_type_ids, ids_to_exclude=ids_to_exclude)
        if self.is_MED: self.MED_loader = MEDLoader(filepath=self.filepath_MED, tokenizer=self.tokenizer,
                                                    is_test=self.is_test_MED, max_seq_len=self.max_seq_len,
                                                    is_token_type_ids=is_token_type_ids)
        if self.is_ReClor: self.ReClor_loader = ReClorLoader(filepath=self.filepath_ReClor, tokenizer=self.tokenizer,
                                                    is_test=self.is_test_ReClor, max_seq_len=self.max_seq_len,
                                                    is_token_type_ids=is_token_type_ids)
        if self.is_DREAM: self.DREAM_loader = DREAMLoader(filepath=self.filepath_DREAM, tokenizer=self.tokenizer,
                                                    is_test=self.is_test_DREAM, max_seq_len=self.max_seq_len,
                                                    is_token_type_ids=is_token_type_ids)
        if self.is_StrategyQA: self.StrategyQA_loader = StrategyQALoader(filepath=self.filepath_StrategyQA, tokenizer=self.tokenizer,
                                                    is_test=self.is_test_StrategyQA, max_seq_len=self.max_seq_len,
                                                    is_token_type_ids=is_token_type_ids)
        if self.is_WiC: self.WiC_loader = WiCLoader(filepath=self.filepath_WiC, tokenizer=self.tokenizer,
                                                    is_test=self.is_test_WiC, max_seq_len=self.max_seq_len,
                                                    is_token_type_ids=is_token_type_ids, ids_to_exclude=ids_to_exclude)
        if self.is_WSC: self.WSC_loader = WSCLoader(filepath=self.filepath_WSC, tokenizer=self.tokenizer,
                                                    is_test=self.is_test_WSC, max_seq_len=self.max_seq_len,
                                                    is_token_type_ids=is_token_type_ids, ids_to_exclude=ids_to_exclude)
        if self.is_CQA: self.CQA_loader = CQALoader(filepath=self.filepath_CQA, tokenizer=self.tokenizer,
                                                    is_test=self.is_test_CQA, max_seq_len=self.max_seq_len,
                                                    is_token_type_ids=is_token_type_ids, ids_to_exclude=ids_to_exclude) # no support for the softmax function mode here.
        if self.is_PIQA: self.PIQA_loader = PIQALoader(filepath=self.filepath_PIQA, filepath_labels=self.filepath_labels_PIQA,
                                                       tokenizer=self.tokenizer, is_test=self.is_test_PIQA,
                                                       max_seq_len=self.max_seq_len, is_token_type_ids=is_token_type_ids) # no support for the softmax function mode here.
        if self.is_SIQA: self.SIQA_loader = SIQALoader(filepath=self.filepath_SIQA, filepath_labels=self.filepath_labels_SIQA,
                                                       tokenizer=self.tokenizer, is_test=self.is_test_SIQA,
                                                       max_seq_len=self.max_seq_len, is_token_type_ids=is_token_type_ids) # no support for the softmax function mode here.
        if self.is_RACE: self.RACE_loader = RACELoader(filepath_middle=self.filepath_RACE_middle,
                                                       filepath_high=self.filepath_RACE_high, tokenizer=self.tokenizer,
                                                       is_test=self.is_test_RACE, max_seq_len=self.max_seq_len,
                                                       is_token_type_ids=is_token_type_ids, ids_to_exclude=ids_to_exclude) # no support for the softmax function mode here.
        if self.is_SciTail: self.SciTail_loader = SciTailLoader(filepath=self.filepath_SciTail, tokenizer=self.tokenizer,
                                                    is_test=self.is_test_SciTail, max_seq_len=self.max_seq_len,
                                                                is_token_type_ids=is_token_type_ids, ids_to_exclude=ids_to_exclude)
        if self.is_MPE: self.MPE_loader = MPELoader(filepath=self.filepath_MPE, tokenizer=self.tokenizer,
                                                    is_test=self.is_test_MPE, max_seq_len=self.max_seq_len,
                                                    is_token_type_ids=is_token_type_ids, mode=error_function_mpe, ids_to_exclude=ids_to_exclude)
        if self.is_SNLI: self.SNLI_loader = SNLILoader(filepath=self.filepath_SNLI, tokenizer=self.tokenizer,
                                                       is_test=self.is_test_SNLI, max_seq_len=self.max_seq_len,
                                                       mode=error_function_cb, is_token_type_ids=is_token_type_ids)

    def get_BoolQ_loader_generator(self):

        self.BoolQ_loader.shuffle = self.shuffle

        if not self.is_test_BoolQ:
            for idx, example, label in self.BoolQ_loader(batch_size=self.batch_size):
                if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids=example["input_ids"]
                attention_mask=example["attention_mask"]
                token_type_ids=example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids, label
        else:
            for idx, example in self.BoolQ_loader(batch_size=self.batch_size):
                if idx is None or example is None: break # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_CQA_loader_generator(self):

        self.CQA_loader.shuffle = self.shuffle

        if not self.is_test_CQA:
            for idx, example, label in self.CQA_loader(batch_size=self.batch_size):
                if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids=example["input_ids"]
                attention_mask=example["attention_mask"]
                token_type_ids=example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids, label
        else:
            for idx, example in self.CQA_loader(batch_size=self.batch_size):
                if idx is None or example is None: break # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_RACE_loader_generator(self):

        self.RACE_loader.shuffle = self.shuffle

        if not self.is_test_RACE:
            for idx, example, label in self.RACE_loader(batch_size=self.batch_size):
                if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids=example["input_ids"]
                attention_mask=example["attention_mask"]
                token_type_ids=example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids, label
        else:
            for idx, example in self.RACE_loader(batch_size=self.batch_size):
                if idx is None or example is None: break # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_MPE_loader_generator(self):

        self.MPE_loader.shuffle = self.shuffle

        if not self.is_test_MPE:
            for idx, example, label in self.MPE_loader(batch_size=self.batch_size):
                if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids=example["input_ids"]
                attention_mask=example["attention_mask"]
                token_type_ids=example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids, label
        else:
            for idx, example in self.MPE_loader(batch_size=self.batch_size):
                if idx is None or example is None: break # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_CB_loader_generator(self):

        self.CB_loader.shuffle = self.shuffle

        if not self.is_test_CB:
            for idx, example, label in self.CB_loader(batch_size=self.batch_size):
                if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids=example["input_ids"]
                attention_mask=example["attention_mask"]
                token_type_ids=example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids, label
        else:
            for idx, example in self.CB_loader(batch_size=self.batch_size):
                if idx is None or example is None: break # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_MultiRC_loader_generator(self):

        self.MultiRC_loader.shuffle = self.shuffle

        if not self.is_test_MultiRC:
            for idx, example, label in self.MultiRC_loader(batch_size=self.batch_size):
                if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids=example["input_ids"]
                attention_mask=example["attention_mask"]
                token_type_ids=example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids, label
        else:
            for idx, example in self.MultiRC_loader(batch_size=self.batch_size):
                if idx is None or example is None: break # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_ReCoRD_loader_generator(self):

        self.ReCoRD_loader.shuffle = self.shuffle

        if not self.is_test_ReCoRD:
            for idx, example, label in self.ReCoRD_loader(batch_size=self.batch_size):
                if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids=example["input_ids"]
                attention_mask=example["attention_mask"]
                token_type_ids=example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids, label
        else:
            for idx, example in self.ReCoRD_loader(batch_size=self.batch_size):
                if idx is None or example is None: break # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_RTE_loader_generator(self):

        self.RTE_loader.shuffle = self.shuffle

        if not self.is_test_RTE:
            for idx, example, label in self.RTE_loader(batch_size=self.batch_size):
                if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids, label
        else:
            for idx, example in self.RTE_loader(batch_size=self.batch_size):
                if idx is None or example is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_WiC_loader_generator(self):

        self.WiC_loader.shuffle = self.shuffle

        if not self.is_test_WiC:
            for idx, example, label in self.WiC_loader(batch_size=self.batch_size):
                if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids, label
        else:
            for idx, example in self.WiC_loader(batch_size=self.batch_size):
                if idx is None or example is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_WSC_loader_generator(self):

        self.WSC_loader.shuffle = self.shuffle

        if not self.is_test_WSC:
            for idx, example, label in self.WSC_loader(batch_size=self.batch_size):
                if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids, label
        else:
            for idx, example in self.WSC_loader(batch_size=self.batch_size):
                if idx is None or example is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_SciTail_loader_generator(self):

        self.SciTail_loader.shuffle = self.shuffle

        if not self.is_test_SciTail:
            for idx, example, label in self.SciTail_loader(batch_size=self.batch_size):
                if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids, label
        else:
            for idx, example in self.SciTail_loader(batch_size=self.batch_size):
                if idx is None or example is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_COPA_loader_generator(self):

        self.COPA_loader.shuffle = self.shuffle

        if not self.is_test_COPA:
            for idx, example, label in self.COPA_loader(batch_size=self.batch_size):
                if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids=example["input_ids"]
                attention_mask=example["attention_mask"]
                token_type_ids=example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids, label
        else:
            for idx, example in self.COPA_loader(batch_size=self.batch_size):
                if idx is None or example is None: break # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_BoolQ_loader(self, batch_size, shuffle):

        self.BoolQ_loader.shuffle = shuffle

        if not self.is_test_BoolQ:
            for idx, example, label in self.BoolQ_loader(batch_size=batch_size):
                if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                yield idx, example, label
        else:
            for idx, example in self.BoolQ_loader(batch_size=batch_size):
                if idx is None or example is None: break # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                yield idx, example

    def get_CB_aux_toks(self):
        if self.is_aux_toks:
            aux_toks = [self._default_bert_large_cased_tokenizer(self.head_aux_token),
                        self._default_bert_large_cased_tokenizer(
                            self.format_aux_toks_dict[self._format_dataset_lookup_table("CB")])]
            for idx, example in self.CB_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                batch_size_ = input_ids.shape[0]
                input_ids = tf.concat([tf.convert_to_tensor([aux_toks]*batch_size_, dtype=tf.dtypes.int32), input_ids], axis=1)
                attention_mask = tf.concat([tf.convert_to_tensor([[1, 1]]*batch_size_, dtype=tf.dtypes.int32), attention_mask], axis=1)
                token_type_ids = tf.concat([tf.convert_to_tensor([[0, 0]]*batch_size_, dtype=tf.dtypes.int32), token_type_ids],axis=1)
                yield idx, input_ids, attention_mask, token_type_ids
        else:
            for idx, example in self.CB_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_SNLI_aux_toks(self):
        if self.is_aux_toks:
            aux_toks = [self._default_bert_large_cased_tokenizer(self.head_aux_token),
                        self._default_bert_large_cased_tokenizer(
                            self.format_aux_toks_dict[self._format_dataset_lookup_table("CB")])] # keep CB as it returns the correct format.
            for idx, example in self.SNLI_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                batch_size_ = input_ids.shape[0]
                input_ids = tf.concat([tf.convert_to_tensor([aux_toks]*batch_size_, dtype=tf.dtypes.int32), input_ids], axis=1)
                attention_mask = tf.concat([tf.convert_to_tensor([[1, 1]]*batch_size_, dtype=tf.dtypes.int32), attention_mask], axis=1)
                token_type_ids = tf.concat([tf.convert_to_tensor([[0, 0]]*batch_size_, dtype=tf.dtypes.int32), token_type_ids],axis=1)
                yield idx, input_ids, attention_mask, token_type_ids
        else:
            for idx, example in self.SNLI_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_MPE_aux_toks(self):
        if self.is_aux_toks:
            aux_toks = [self._default_bert_large_cased_tokenizer(self.head_aux_token),
                        self._default_bert_large_cased_tokenizer(
                            self.format_aux_toks_dict[self._format_dataset_lookup_table("MPE")])]
            for idx, example in self.MPE_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                batch_size_ = input_ids.shape[0]
                input_ids = tf.concat([tf.convert_to_tensor([aux_toks]*batch_size_, dtype=tf.dtypes.int32), input_ids], axis=1)
                attention_mask = tf.concat([tf.convert_to_tensor([[1, 1]]*batch_size_, dtype=tf.dtypes.int32), attention_mask], axis=1)
                token_type_ids = tf.concat([tf.convert_to_tensor([[0, 0]]*batch_size_, dtype=tf.dtypes.int32), token_type_ids],axis=1)
                yield idx, input_ids, attention_mask, token_type_ids
        else:
            for idx, example in self.MPE_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_RTE_aux_toks(self):
        if self.is_aux_toks:
            aux_toks = [self._default_bert_large_cased_tokenizer(self.head_aux_token),
                        self._default_bert_large_cased_tokenizer(
                            self.format_aux_toks_dict[self._format_dataset_lookup_table("RTE")])]
            for idx, example in self.RTE_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                batch_size_ = input_ids.shape[0]
                input_ids = tf.concat([tf.convert_to_tensor([aux_toks]*batch_size_, dtype=tf.dtypes.int32), input_ids], axis=1)
                attention_mask = tf.concat([tf.convert_to_tensor([[1, 1]]*batch_size_, dtype=tf.dtypes.int32), attention_mask], axis=1)
                token_type_ids = tf.concat([tf.convert_to_tensor([[0, 0]]*batch_size_, dtype=tf.dtypes.int32), token_type_ids],axis=1)
                yield idx, input_ids, attention_mask, token_type_ids
        else:
            for idx, example in self.RTE_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_MED_aux_toks(self):
        if self.is_aux_toks:
            aux_toks = [self._default_bert_large_cased_tokenizer(self.head_aux_token),
                        self._default_bert_large_cased_tokenizer(
                            self.format_aux_toks_dict[self._format_dataset_lookup_table("RTE")])] # RTE is correct, MED used the same aux tokens.
            for idx, example in self.MED_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                batch_size_ = input_ids.shape[0]
                input_ids = tf.concat([tf.convert_to_tensor([aux_toks]*batch_size_, dtype=tf.dtypes.int32), input_ids], axis=1)
                attention_mask = tf.concat([tf.convert_to_tensor([[1, 1]]*batch_size_, dtype=tf.dtypes.int32), attention_mask], axis=1)
                token_type_ids = tf.concat([tf.convert_to_tensor([[0, 0]]*batch_size_, dtype=tf.dtypes.int32), token_type_ids],axis=1)
                yield idx, input_ids, attention_mask, token_type_ids
        else:
            for idx, example in self.MED_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_SciTail_aux_toks(self):
        if self.is_aux_toks:
            aux_toks = [self._default_bert_large_cased_tokenizer(self.head_aux_token),
                        self._default_bert_large_cased_tokenizer(
                            self.format_aux_toks_dict[self._format_dataset_lookup_table("SciTail")])]
            for idx, example in self.SciTail_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                batch_size_ = input_ids.shape[0]
                input_ids = tf.concat([tf.convert_to_tensor([aux_toks]*batch_size_, dtype=tf.dtypes.int32), input_ids], axis=1)
                attention_mask = tf.concat([tf.convert_to_tensor([[1, 1]]*batch_size_, dtype=tf.dtypes.int32), attention_mask], axis=1)
                token_type_ids = tf.concat([tf.convert_to_tensor([[0, 0]]*batch_size_, dtype=tf.dtypes.int32), token_type_ids],axis=1)
                yield idx, input_ids, attention_mask, token_type_ids
        else:
            for idx, example in self.SciTail_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_COPA_aux_toks(self):
        if self.is_aux_toks:
            aux_toks = [self._default_bert_large_cased_tokenizer(self.head_aux_token),
                        self._default_bert_large_cased_tokenizer(
                            self.format_aux_toks_dict[self._format_dataset_lookup_table("COPA")])]
            for idx, example in self.COPA_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                batch_size_ = input_ids.shape[0]
                input_ids = tf.concat([tf.convert_to_tensor([aux_toks]*batch_size_, dtype=tf.dtypes.int32), input_ids], axis=1)
                attention_mask = tf.concat([tf.convert_to_tensor([[1, 1]]*batch_size_, dtype=tf.dtypes.int32), attention_mask], axis=1)
                token_type_ids = tf.concat([tf.convert_to_tensor([[0, 0]]*batch_size_, dtype=tf.dtypes.int32), token_type_ids],axis=1)
                yield idx, input_ids, attention_mask, token_type_ids
        else:
            for idx, example in self.COPA_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_CQA_aux_toks(self):
        if self.is_aux_toks:
            aux_toks = [self._default_bert_large_cased_tokenizer(self.head_aux_token),
                        self._default_bert_large_cased_tokenizer(
                            self.format_aux_toks_dict[self._format_dataset_lookup_table("CQA")])]
            for idx, example in self.CQA_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                batch_size_ = input_ids.shape[0]
                input_ids = tf.concat([tf.convert_to_tensor([aux_toks]*batch_size_, dtype=tf.dtypes.int32), input_ids], axis=1)
                attention_mask = tf.concat([tf.convert_to_tensor([[1, 1]]*batch_size_, dtype=tf.dtypes.int32), attention_mask], axis=1)
                token_type_ids = tf.concat([tf.convert_to_tensor([[0, 0]]*batch_size_, dtype=tf.dtypes.int32), token_type_ids],axis=1)
                yield idx, input_ids, attention_mask, token_type_ids
        else:
            for idx, example in self.CQA_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_PIQA_aux_toks(self):
        if self.is_aux_toks:
            aux_toks = [self._default_bert_large_cased_tokenizer(self.head_aux_token),
                        self._default_bert_large_cased_tokenizer(
                            self.format_aux_toks_dict[self._format_dataset_lookup_table("CQA")])] # keep CQA here.
            for idx, example in self.PIQA_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                batch_size_ = input_ids.shape[0]
                input_ids = tf.concat([tf.convert_to_tensor([aux_toks]*batch_size_, dtype=tf.dtypes.int32), input_ids], axis=1)
                attention_mask = tf.concat([tf.convert_to_tensor([[1, 1]]*batch_size_, dtype=tf.dtypes.int32), attention_mask], axis=1)
                token_type_ids = tf.concat([tf.convert_to_tensor([[0, 0]]*batch_size_, dtype=tf.dtypes.int32), token_type_ids],axis=1)
                yield idx, input_ids, attention_mask, token_type_ids
        else:
            for idx, example in self.PIQA_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_SIQA_aux_toks(self):
        if self.is_aux_toks:
            aux_toks = [self._default_bert_large_cased_tokenizer(self.head_aux_token),
                        self._default_bert_large_cased_tokenizer(
                            self.format_aux_toks_dict[self._format_dataset_lookup_table("RACE")])] # keep RACE here. for [P+QA] - [C+QA]
            for idx, example in self.SIQA_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                batch_size_ = input_ids.shape[0]
                input_ids = tf.concat([tf.convert_to_tensor([aux_toks]*batch_size_, dtype=tf.dtypes.int32), input_ids], axis=1)
                attention_mask = tf.concat([tf.convert_to_tensor([[1, 1]]*batch_size_, dtype=tf.dtypes.int32), attention_mask], axis=1)
                token_type_ids = tf.concat([tf.convert_to_tensor([[0, 0]]*batch_size_, dtype=tf.dtypes.int32), token_type_ids],axis=1)
                yield idx, input_ids, attention_mask, token_type_ids
        else:
            for idx, example in self.SIQA_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_WSC_aux_toks(self):
        if self.is_aux_toks:
            aux_toks = [self._default_bert_large_cased_tokenizer(self.head_aux_token),
                        self._default_bert_large_cased_tokenizer(
                            self.format_aux_toks_dict[self._format_dataset_lookup_table("WSC")])]
            for idx, example in self.WSC_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                batch_size_ = input_ids.shape[0]
                input_ids = tf.concat([tf.convert_to_tensor([aux_toks]*batch_size_, dtype=tf.dtypes.int32), input_ids], axis=1)
                attention_mask = tf.concat([tf.convert_to_tensor([[1, 1]]*batch_size_, dtype=tf.dtypes.int32), attention_mask], axis=1)
                token_type_ids = tf.concat([tf.convert_to_tensor([[0, 0]]*batch_size_, dtype=tf.dtypes.int32), token_type_ids],axis=1)
                yield idx, input_ids, attention_mask, token_type_ids
        else:
            for idx, example in self.WSC_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_MultiRC_aux_toks(self):
        if self.is_aux_toks:
            aux_toks = [self._default_bert_large_cased_tokenizer(self.head_aux_token),
                        self._default_bert_large_cased_tokenizer(
                            self.format_aux_toks_dict[self._format_dataset_lookup_table("MultiRC")])]
            for idx, example in self.MultiRC_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                batch_size_ = input_ids.shape[0]
                input_ids = tf.concat([tf.convert_to_tensor([aux_toks]*batch_size_, dtype=tf.dtypes.int32), input_ids], axis=1)
                attention_mask = tf.concat([tf.convert_to_tensor([[1, 1]]*batch_size_, dtype=tf.dtypes.int32), attention_mask], axis=1)
                token_type_ids = tf.concat([tf.convert_to_tensor([[0, 0]]*batch_size_, dtype=tf.dtypes.int32), token_type_ids],axis=1)
                yield idx, input_ids, attention_mask, token_type_ids
        else:
            for idx, example in self.MultiRC_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_ReCoRD_aux_toks(self):
        if self.is_aux_toks:
            aux_toks = [self._default_bert_large_cased_tokenizer(self.head_aux_token),
                        self._default_bert_large_cased_tokenizer(
                            self.format_aux_toks_dict[self._format_dataset_lookup_table("ReCoRD")])]
            for idx, example in self.ReCoRD_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                batch_size_ = input_ids.shape[0]
                input_ids = tf.concat([tf.convert_to_tensor([aux_toks]*batch_size_, dtype=tf.dtypes.int32), input_ids], axis=1)
                attention_mask = tf.concat([tf.convert_to_tensor([[1, 1]]*batch_size_, dtype=tf.dtypes.int32), attention_mask], axis=1)
                token_type_ids = tf.concat([tf.convert_to_tensor([[0, 0]]*batch_size_, dtype=tf.dtypes.int32), token_type_ids],axis=1)
                yield idx, input_ids, attention_mask, token_type_ids
        else:
            for idx, example in self.ReCoRD_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_RACE_aux_toks(self):
        if self.is_aux_toks:
            aux_toks = [self._default_bert_large_cased_tokenizer(self.head_aux_token),
                        self._default_bert_large_cased_tokenizer(
                            self.format_aux_toks_dict[self._format_dataset_lookup_table("RACE")])]
            for idx, example in self.RACE_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                batch_size_ = input_ids.shape[0]
                input_ids = tf.concat([tf.convert_to_tensor([aux_toks]*batch_size_, dtype=tf.dtypes.int32), input_ids], axis=1)
                attention_mask = tf.concat([tf.convert_to_tensor([[1, 1]]*batch_size_, dtype=tf.dtypes.int32), attention_mask], axis=1)
                token_type_ids = tf.concat([tf.convert_to_tensor([[0, 0]]*batch_size_, dtype=tf.dtypes.int32), token_type_ids],axis=1)
                yield idx, input_ids, attention_mask, token_type_ids
        else:
            for idx, example in self.RACE_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_ReClor_aux_toks(self):
        if self.is_aux_toks:
            aux_toks = [self._default_bert_large_cased_tokenizer(self.head_aux_token),
                        self._default_bert_large_cased_tokenizer(
                            self.format_aux_toks_dict[self._format_dataset_lookup_table("RACE")])] # RACE provides the correct format for ReClor.
            for idx, example in self.ReClor_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                batch_size_ = input_ids.shape[0]
                input_ids = tf.concat([tf.convert_to_tensor([aux_toks]*batch_size_, dtype=tf.dtypes.int32), input_ids], axis=1)
                attention_mask = tf.concat([tf.convert_to_tensor([[1, 1]]*batch_size_, dtype=tf.dtypes.int32), attention_mask], axis=1)
                token_type_ids = tf.concat([tf.convert_to_tensor([[0, 0]]*batch_size_, dtype=tf.dtypes.int32), token_type_ids],axis=1)
                yield idx, input_ids, attention_mask, token_type_ids
        else:
            for idx, example in self.ReClor_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_DREAM_aux_toks(self):
        if self.is_aux_toks:
            aux_toks = [self._default_bert_large_cased_tokenizer(self.head_aux_token),
                        self._default_bert_large_cased_tokenizer(
                            self.format_aux_toks_dict[self._format_dataset_lookup_table("RACE")])] # RACE provides the correct format for DREAM.
            for idx, example in self.DREAM_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                batch_size_ = input_ids.shape[0]
                input_ids = tf.concat([tf.convert_to_tensor([aux_toks]*batch_size_, dtype=tf.dtypes.int32), input_ids], axis=1)
                attention_mask = tf.concat([tf.convert_to_tensor([[1, 1]]*batch_size_, dtype=tf.dtypes.int32), attention_mask], axis=1)
                token_type_ids = tf.concat([tf.convert_to_tensor([[0, 0]]*batch_size_, dtype=tf.dtypes.int32), token_type_ids],axis=1)
                yield idx, input_ids, attention_mask, token_type_ids
        else:
            for idx, example in self.DREAM_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_StrategyQA_aux_toks(self):
        if self.is_aux_toks:
            aux_toks = [self._default_bert_large_cased_tokenizer(self.head_aux_token),
                        self._default_bert_large_cased_tokenizer(
                            self.format_aux_toks_dict[self._format_dataset_lookup_table("BoolQ")])] # BoolQ provides the correct format for StrategyQA.
            for idx, example in self.StrategyQA_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                batch_size_ = input_ids.shape[0]
                input_ids = tf.concat([tf.convert_to_tensor([aux_toks]*batch_size_, dtype=tf.dtypes.int32), input_ids], axis=1)
                attention_mask = tf.concat([tf.convert_to_tensor([[1, 1]]*batch_size_, dtype=tf.dtypes.int32), attention_mask], axis=1)
                token_type_ids = tf.concat([tf.convert_to_tensor([[0, 0]]*batch_size_, dtype=tf.dtypes.int32), token_type_ids],axis=1)
                yield idx, input_ids, attention_mask, token_type_ids
        else:
            for idx, example in self.StrategyQA_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_WiC_aux_toks(self):
        if self.is_aux_toks:
            aux_toks = [self._default_bert_large_cased_tokenizer(self.head_aux_token),
                        self._default_bert_large_cased_tokenizer(
                            self.format_aux_toks_dict[self._format_dataset_lookup_table("WiC")])]
            for idx, example in self.WiC_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                batch_size_ = input_ids.shape[0]
                input_ids = tf.concat([tf.convert_to_tensor([aux_toks]*batch_size_, dtype=tf.dtypes.int32), input_ids], axis=1)
                attention_mask = tf.concat([tf.convert_to_tensor([[1, 1]]*batch_size_, dtype=tf.dtypes.int32), attention_mask], axis=1)
                token_type_ids = tf.concat([tf.convert_to_tensor([[0, 0]]*batch_size_, dtype=tf.dtypes.int32), token_type_ids],axis=1)
                yield idx, input_ids, attention_mask, token_type_ids
        else:
            for idx, example in self.WiC_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_BoolQ_aux_toks(self):
        if self.is_aux_toks:
            aux_toks = [self._default_bert_large_cased_tokenizer(self.head_aux_token),
                        self._default_bert_large_cased_tokenizer(
                            self.format_aux_toks_dict[self._format_dataset_lookup_table("BoolQ")])]
            for idx, example in self.BoolQ_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                batch_size_ = input_ids.shape[0]
                input_ids = tf.concat([tf.convert_to_tensor([aux_toks]*batch_size_, dtype=tf.dtypes.int32), input_ids], axis=1)
                attention_mask = tf.concat([tf.convert_to_tensor([[1, 1]]*batch_size_, dtype=tf.dtypes.int32), attention_mask], axis=1)
                token_type_ids = tf.concat([tf.convert_to_tensor([[0, 0]]*batch_size_, dtype=tf.dtypes.int32), token_type_ids],axis=1)
                yield idx, input_ids, attention_mask, token_type_ids
        else:
            for idx, example in self.BoolQ_loader(batch_size=self.batch_size):
                if idx is None or example is None: break
                input_ids = example["input_ids"]
                attention_mask = example["attention_mask"]
                token_type_ids = example["token_type_ids"]
                yield idx, input_ids, attention_mask, token_type_ids

    def get_GQA_generator_for_ablation(self):
        # ENT-3 CB
        # CS-1 COPA
        # Y/N-1 BoolQ
        # WIC-1 WiC
        # remember to shuffle the datasets if it is set to true
        self.CB_loader.shuffle = self.shuffle_datasets
        self.COPA_loader.shuffle = self.shuffle_datasets
        self.BoolQ_loader.shuffle = self.shuffle_datasets
        self.WiC_loader.shuffle = self.shuffle_datasets

        self.head_aux_toks_dict = self._head_aux_tokens()
        self.format_aux_toks_dict = self._format_aux_tokens()

        def CB_loader(head_aux_token: str, batch_size=1):
            if self.is_aux_toks:
                aux_toks = [self._default_bert_large_cased_tokenizer(head_aux_token),
                            self._default_bert_large_cased_tokenizer(self.format_aux_toks_dict[self._format_dataset_lookup_table("CB")])]
                for idx, example, label in self.CB_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                    input_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor(aux_toks, dtype=tf.dtypes.int32), axis=0),input_ids], axis=1)
                    attention_mask = tf.concat([tf.expand_dims(tf.convert_to_tensor([1,1], dtype=tf.dtypes.int32), axis=0),attention_mask], axis=1)
                    token_type_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor([0,0], dtype=tf.dtypes.int32), axis=0),token_type_ids], axis=1)
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None
            else:
                for idx, example, label in self.CB_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids = example["input_ids"]
                    attention_mask = example["attention_mask"]
                    token_type_ids = example["token_type_ids"]
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None

        def COPA_loader(head_aux_token: str, batch_size=1):
            if self.is_aux_toks:
                aux_toks = [self._default_bert_large_cased_tokenizer(head_aux_token),
                            self._default_bert_large_cased_tokenizer(self.format_aux_toks_dict[self._format_dataset_lookup_table("COPA")])]
                for idx, example, label in self.COPA_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                    input_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor(aux_toks, dtype=tf.dtypes.int32), axis=0),input_ids], axis=1)
                    attention_mask = tf.concat([tf.expand_dims(tf.convert_to_tensor([1,1], dtype=tf.dtypes.int32), axis=0),attention_mask], axis=1)
                    token_type_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor([0,0], dtype=tf.dtypes.int32), axis=0),token_type_ids], axis=1)
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None
            else:
                for idx, example, label in self.COPA_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids = example["input_ids"]
                    attention_mask = example["attention_mask"]
                    token_type_ids = example["token_type_ids"]
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None

        def BoolQ_loader(head_aux_token: str, batch_size=1):
            if self.is_aux_toks:
                aux_toks = [self._default_bert_large_cased_tokenizer(head_aux_token),
                            self._default_bert_large_cased_tokenizer(self.format_aux_toks_dict[self._format_dataset_lookup_table("BoolQ")])]
                for idx, example, label in self.BoolQ_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                    input_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor(aux_toks, dtype=tf.dtypes.int32), axis=0),input_ids], axis=1)
                    attention_mask = tf.concat([tf.expand_dims(tf.convert_to_tensor([1,1], dtype=tf.dtypes.int32), axis=0),attention_mask], axis=1)
                    token_type_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor([0,0], dtype=tf.dtypes.int32), axis=0),token_type_ids], axis=1)
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None
            else:
                for idx, example, label in self.BoolQ_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids = example["input_ids"]
                    attention_mask = example["attention_mask"]
                    token_type_ids = example["token_type_ids"]
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None

        def WiC_loader(head_aux_token: str, batch_size=1):
            if self.is_aux_toks:
                aux_toks = [self._default_bert_large_cased_tokenizer(head_aux_token),
                           self._default_bert_large_cased_tokenizer(self.format_aux_toks_dict[self._format_dataset_lookup_table("WiC")])]
                for idx, example, label in self.WiC_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                    input_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor(aux_toks, dtype=tf.dtypes.int32), axis=0),input_ids], axis=1)
                    attention_mask = tf.concat([tf.expand_dims(tf.convert_to_tensor([1,1], dtype=tf.dtypes.int32), axis=0),attention_mask], axis=1)
                    token_type_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor([0,0], dtype=tf.dtypes.int32), axis=0),token_type_ids], axis=1)
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None
            else:
                for idx, example, label in self.WiC_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids = example["input_ids"]
                    attention_mask = example["attention_mask"]
                    token_type_ids = example["token_type_ids"]
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None

        self.general_CB_loader = CB_loader(head_aux_token=self.head_aux_toks_dict["entailment-3"])
        self.general_COPA_loader = COPA_loader(head_aux_token=self.head_aux_toks_dict["commonsense"])
        self.general_BoolQ_loader = BoolQ_loader(head_aux_token=self.head_aux_toks_dict["yes/no"])
        self.general_WiC_loader = WiC_loader(head_aux_token=self.head_aux_toks_dict["word-in-context"])

        def get_head_datasets_1(): # entailment-3
            if self.is_diagnostics: print(f"CB")
            self.CB_counter += 1
            input_ids, attention_mask, token_type_ids, label = next(self.general_CB_loader)
            if input_ids is None: # then re-load the dataloader. It will be shuffled again as this is a new call.
                self.general_CB_loader = CB_loader(head_aux_token=self.head_aux_toks_dict["entailment-3"])
                input_ids, attention_mask, token_type_ids, label = next(self.general_CB_loader)
            return input_ids, attention_mask, token_type_ids, label

        def get_head_datasets_3(): # commonsense
            if self.is_diagnostics: print(f"COPA")
            self.COPA_counter += 1
            input_ids, attention_mask, token_type_ids, label = next(self.general_COPA_loader)
            if input_ids is None:  # then re-load the dataloader. It will be shuffled again as this is a new call.
                self.general_COPA_loader = COPA_loader(head_aux_token=self.head_aux_toks_dict["commonsense"])
                input_ids, attention_mask, token_type_ids, label = next(self.general_COPA_loader)
            return input_ids, attention_mask, token_type_ids, label

        def get_head_datasets_5(): # yes/no
            if self.is_diagnostics: print(f"BoolQ")
            self.BoolQ_counter += 1
            input_ids, attention_mask, token_type_ids, label = next(self.general_BoolQ_loader)
            if input_ids is None:  # then re-load the dataloader. It will be shuffled again as this is a new call.
                self.general_BoolQ_loader = BoolQ_loader(head_aux_token=self.head_aux_toks_dict["yes/no"])
                input_ids, attention_mask, token_type_ids, label = next(self.general_BoolQ_loader)
            return input_ids, attention_mask, token_type_ids, label

        def get_head_datasets_6(): # word-in-context
            if self.is_diagnostics: print(f"WiC")
            self.WiC_counter += 1
            input_ids, attention_mask, token_type_ids, label = next(self.general_WiC_loader)
            if input_ids is None:  # then re-load the dataloader. It will be shuffled again as this is a new call.
                self.general_WiC_loader = WiC_loader(head_aux_token=self.head_aux_toks_dict["word-in-context"])
                input_ids, attention_mask, token_type_ids, label = next(self.general_WiC_loader)
            return input_ids, attention_mask, token_type_ids, label

        for i in range(self.num_iterations):

            head_number = random.randrange(1, 5) # 1 to 4 (inclusive)

            binput_ids, battention_mask, btoken_type_ids, blabel = None, None, None, None
            for j in range(self.batch_size):

                if head_number == 1: # entailment-3
                    input_ids, attention_mask, token_type_ids, label = get_head_datasets_1()
                    self.head1_counter += 1
                elif head_number == 2: # commonsense; head_num was 3
                    input_ids, attention_mask, token_type_ids, label = get_head_datasets_3()
                    self.head2_counter += 1
                elif head_number == 3: # yes/no; head_num was 5
                    input_ids, attention_mask, token_type_ids, label = get_head_datasets_5()
                    self.head3_counter += 1
                elif head_number == 4: # word-in-context; head_num was 6
                    input_ids, attention_mask, token_type_ids, label = get_head_datasets_6()
                    self.head4_counter += 1
                else: raise Exception(f"Invalid head number: {head_number}!\nThis should never be reached!")

                if binput_ids is None:
                    #print(input_ids.shape)
                    assert input_ids.shape[0] == 1, f"The batch size should be 1, got {input_ids.shape[0]}!"
                    if self.is_aux_toks:
                        assert input_ids.shape[1] == 2+self.max_seq_len, f"The sequence length should be " \
                                                                         f"{2+self.max_seq_len}, got {input_ids.shape[1]}!"
                    binput_ids, battention_mask, btoken_type_ids, blabel = input_ids, attention_mask, token_type_ids, label
                else:
                    if self.is_aux_toks:
                        assert input_ids.shape[1] == 2+self.max_seq_len, f"The sequence length should be " \
                                                                         f"{2+self.max_seq_len}, got {input_ids.shape[1]}!"
                    binput_ids = tf.concat([binput_ids, input_ids], axis=0) # axis 0 is the batch dimension.
                    battention_mask = tf.concat([battention_mask, attention_mask], axis=0) # axis 0 is the batch dimension.
                    btoken_type_ids = tf.concat([btoken_type_ids, token_type_ids], axis=0) # axis 0 is the batch dimension.
                    blabel = tf.concat([blabel, label], axis=0) # axis 0 is the batch dimension.

            # +2 b/c of the two auxiliary tokens.
            mslen = self.max_seq_len+2 if self.is_aux_toks else self.max_seq_len
            assert binput_ids.shape[0] == self.batch_size and binput_ids.shape[1] == mslen
            assert battention_mask.shape[0] == self.batch_size and battention_mask.shape[1] == mslen
            assert btoken_type_ids.shape[0] == self.batch_size and btoken_type_ids.shape[1] == mslen
            assert blabel.shape[0] == self.batch_size # depending on the head, the second dimension can change.
            # head_number
            head_number_ = tf.cast(tf.convert_to_tensor(head_number), dtype=tf.dtypes.int8)
            yield binput_ids, battention_mask, btoken_type_ids, blabel, head_number_

        print(f"head1_counter: {self.head1_counter}\n"
              f"CB_counter: {self.CB_counter}\n"
              f"head2_counter: {self.head2_counter}\n"
              f"COPA_counter: {self.COPA_counter}\n"
              f"head3_counter: {self.head3_counter}\n"
              f"BoolQ_counter: {self.BoolQ_counter}\n"
              f"head4_counter: {self.head4_counter}\n"
              f"WiC_counter: {self.WiC_counter}\n")

    def get_GQA_generator_default(self):

        # remember to shuffle the datasets if it is set to true
        self.CB_loader.shuffle = self.shuffle_datasets
        self.MPE_loader.shuffle = self.shuffle_datasets
        self.RTE_loader.shuffle = self.shuffle_datasets
        self.SciTail_loader.shuffle = self.shuffle_datasets
        self.COPA_loader.shuffle = self.shuffle_datasets
        self.CQA_loader.shuffle = self.shuffle_datasets
        self.WSC_loader.shuffle = self.shuffle_datasets
        self.MultiRC_loader.shuffle = self.shuffle_datasets
        self.ReCoRD_loader.shuffle = self.shuffle_datasets
        self.RACE_loader.shuffle = self.shuffle_datasets
        self.BoolQ_loader.shuffle = self.shuffle_datasets
        self.WiC_loader.shuffle = self.shuffle_datasets

        self.head_aux_toks_dict = self._head_aux_tokens()
        self.format_aux_toks_dict = self._format_aux_tokens()

        def CB_loader(head_aux_token: str, batch_size=1):
            if self.is_aux_toks:
                aux_toks = [self._default_bert_large_cased_tokenizer(head_aux_token),
                            self._default_bert_large_cased_tokenizer(self.format_aux_toks_dict[self._format_dataset_lookup_table("CB")])]
                for idx, example, label in self.CB_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                    input_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor(aux_toks, dtype=tf.dtypes.int32), axis=0),input_ids], axis=1)
                    attention_mask = tf.concat([tf.expand_dims(tf.convert_to_tensor([1,1], dtype=tf.dtypes.int32), axis=0),attention_mask], axis=1)
                    token_type_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor([0,0], dtype=tf.dtypes.int32), axis=0),token_type_ids], axis=1)
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None
            else:
                for idx, example, label in self.CB_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids = example["input_ids"]
                    attention_mask = example["attention_mask"]
                    token_type_ids = example["token_type_ids"]
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None

        def MPE_loader(head_aux_token: str, batch_size=1):
            if self.is_aux_toks:
                aux_toks = [self._default_bert_large_cased_tokenizer(head_aux_token),
                            self._default_bert_large_cased_tokenizer(self.format_aux_toks_dict[self._format_dataset_lookup_table("MPE")])]
                for idx, example, label in self.MPE_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                    input_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor(aux_toks, dtype=tf.dtypes.int32), axis=0),input_ids], axis=1)
                    attention_mask = tf.concat([tf.expand_dims(tf.convert_to_tensor([1,1], dtype=tf.dtypes.int32), axis=0),attention_mask], axis=1)
                    token_type_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor([0,0], dtype=tf.dtypes.int32), axis=0),token_type_ids], axis=1)
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None
            else:
                for idx, example, label in self.MPE_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids = example["input_ids"]
                    attention_mask = example["attention_mask"]
                    token_type_ids = example["token_type_ids"]
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None

        def RTE_loader(head_aux_token: str, batch_size=1):
            if self.is_aux_toks:
                aux_toks = [self._default_bert_large_cased_tokenizer(head_aux_token),
                            self._default_bert_large_cased_tokenizer(self.format_aux_toks_dict[self._format_dataset_lookup_table("RTE")])]
                for idx, example, label in self.RTE_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                    input_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor(aux_toks, dtype=tf.dtypes.int32), axis=0),input_ids], axis=1)
                    attention_mask = tf.concat([tf.expand_dims(tf.convert_to_tensor([1,1], dtype=tf.dtypes.int32), axis=0),attention_mask], axis=1)
                    token_type_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor([0,0], dtype=tf.dtypes.int32), axis=0),token_type_ids], axis=1)
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None
            else:
                for idx, example, label in self.RTE_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids = example["input_ids"]
                    attention_mask = example["attention_mask"]
                    token_type_ids = example["token_type_ids"]
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None

        def SciTail_loader(head_aux_token: str, batch_size=1):
            if self.is_aux_toks:
                aux_toks = [self._default_bert_large_cased_tokenizer(head_aux_token),
                            self._default_bert_large_cased_tokenizer(self.format_aux_toks_dict[self._format_dataset_lookup_table("SciTail")])]
                for idx, example, label in self.SciTail_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                    input_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor(aux_toks, dtype=tf.dtypes.int32), axis=0),input_ids], axis=1)
                    attention_mask = tf.concat([tf.expand_dims(tf.convert_to_tensor([1,1], dtype=tf.dtypes.int32), axis=0),attention_mask], axis=1)
                    token_type_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor([0,0], dtype=tf.dtypes.int32), axis=0),token_type_ids], axis=1)
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None
            else:
                for idx, example, label in self.SciTail_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids = example["input_ids"]
                    attention_mask = example["attention_mask"]
                    token_type_ids = example["token_type_ids"]
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None

        def COPA_loader(head_aux_token: str, batch_size=1):
            if self.is_aux_toks:
                aux_toks = [self._default_bert_large_cased_tokenizer(head_aux_token),
                            self._default_bert_large_cased_tokenizer(self.format_aux_toks_dict[self._format_dataset_lookup_table("COPA")])]
                for idx, example, label in self.COPA_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                    input_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor(aux_toks, dtype=tf.dtypes.int32), axis=0),input_ids], axis=1)
                    attention_mask = tf.concat([tf.expand_dims(tf.convert_to_tensor([1,1], dtype=tf.dtypes.int32), axis=0),attention_mask], axis=1)
                    token_type_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor([0,0], dtype=tf.dtypes.int32), axis=0),token_type_ids], axis=1)
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None
            else:
                for idx, example, label in self.COPA_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids = example["input_ids"]
                    attention_mask = example["attention_mask"]
                    token_type_ids = example["token_type_ids"]
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None

        def CQA_loader(head_aux_token: str, batch_size=1):
            if self.is_aux_toks:
                aux_toks = [self._default_bert_large_cased_tokenizer(head_aux_token),
                            self._default_bert_large_cased_tokenizer(self.format_aux_toks_dict[self._format_dataset_lookup_table("CQA")])]
                for idx, example, label in self.CQA_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                    input_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor(aux_toks, dtype=tf.dtypes.int32), axis=0),input_ids], axis=1)
                    attention_mask = tf.concat([tf.expand_dims(tf.convert_to_tensor([1,1], dtype=tf.dtypes.int32), axis=0),attention_mask], axis=1)
                    token_type_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor([0,0], dtype=tf.dtypes.int32), axis=0),token_type_ids], axis=1)
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None
            else:
                for idx, example, label in self.CQA_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids = example["input_ids"]
                    attention_mask = example["attention_mask"]
                    token_type_ids = example["token_type_ids"]
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None

        def WSC_loader(head_aux_token: str, batch_size=1):
            if self.is_aux_toks:
                aux_toks = [self._default_bert_large_cased_tokenizer(head_aux_token),
                            self._default_bert_large_cased_tokenizer(self.format_aux_toks_dict[self._format_dataset_lookup_table("WSC")])]
                for idx, example, label in self.WSC_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                    input_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor(aux_toks, dtype=tf.dtypes.int32), axis=0),input_ids], axis=1)
                    attention_mask = tf.concat([tf.expand_dims(tf.convert_to_tensor([1,1], dtype=tf.dtypes.int32), axis=0),attention_mask], axis=1)
                    token_type_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor([0,0], dtype=tf.dtypes.int32), axis=0),token_type_ids], axis=1)
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None
            else:
                for idx, example, label in self.WSC_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids = example["input_ids"]
                    attention_mask = example["attention_mask"]
                    token_type_ids = example["token_type_ids"]
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None

        def MultiRC_loader(head_aux_token: str, batch_size=1):
            if self.is_aux_toks:
                aux_toks = [self._default_bert_large_cased_tokenizer(head_aux_token),
                            self._default_bert_large_cased_tokenizer(self.format_aux_toks_dict[self._format_dataset_lookup_table("MultiRC")])]
                for idx, example, label in self.MultiRC_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                    input_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor(aux_toks, dtype=tf.dtypes.int32), axis=0),input_ids], axis=1)
                    attention_mask = tf.concat([tf.expand_dims(tf.convert_to_tensor([1,1], dtype=tf.dtypes.int32), axis=0),attention_mask], axis=1)
                    token_type_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor([0,0], dtype=tf.dtypes.int32), axis=0),token_type_ids], axis=1)
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None
            else:
                for idx, example, label in self.MultiRC_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids = example["input_ids"]
                    attention_mask = example["attention_mask"]
                    token_type_ids = example["token_type_ids"]
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None

        def ReCoRD_loader(head_aux_token: str, batch_size=1):
            if self.is_aux_toks:
                aux_toks = [self._default_bert_large_cased_tokenizer(head_aux_token),
                            self._default_bert_large_cased_tokenizer(self.format_aux_toks_dict[self._format_dataset_lookup_table("ReCoRD")])]
                for idx, example, label in self.ReCoRD_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                    input_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor(aux_toks, dtype=tf.dtypes.int32), axis=0),input_ids], axis=1)
                    attention_mask = tf.concat([tf.expand_dims(tf.convert_to_tensor([1,1], dtype=tf.dtypes.int32), axis=0),attention_mask], axis=1)
                    token_type_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor([0,0], dtype=tf.dtypes.int32), axis=0),token_type_ids], axis=1)
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None
            else:
                for idx, example, label in self.ReCoRD_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids = example["input_ids"]
                    attention_mask = example["attention_mask"]
                    token_type_ids = example["token_type_ids"]
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None

        def RACE_loader(head_aux_token: str, batch_size=1):
            if self.is_aux_toks:
                aux_toks = [self._default_bert_large_cased_tokenizer(head_aux_token),
                            self._default_bert_large_cased_tokenizer(self.format_aux_toks_dict[self._format_dataset_lookup_table("RACE")])]
                for idx, example, label in self.RACE_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                    input_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor(aux_toks, dtype=tf.dtypes.int32), axis=0),input_ids], axis=1)
                    attention_mask = tf.concat([tf.expand_dims(tf.convert_to_tensor([1,1], dtype=tf.dtypes.int32), axis=0),attention_mask], axis=1)
                    token_type_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor([0,0], dtype=tf.dtypes.int32), axis=0),token_type_ids], axis=1)
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None
            else:
                for idx, example, label in self.RACE_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids = example["input_ids"]
                    attention_mask = example["attention_mask"]
                    token_type_ids = example["token_type_ids"]
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None

        def BoolQ_loader(head_aux_token: str, batch_size=1):
            if self.is_aux_toks:
                aux_toks = [self._default_bert_large_cased_tokenizer(head_aux_token),
                            self._default_bert_large_cased_tokenizer(self.format_aux_toks_dict[self._format_dataset_lookup_table("BoolQ")])]
                for idx, example, label in self.BoolQ_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                    input_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor(aux_toks, dtype=tf.dtypes.int32), axis=0),input_ids], axis=1)
                    attention_mask = tf.concat([tf.expand_dims(tf.convert_to_tensor([1,1], dtype=tf.dtypes.int32), axis=0),attention_mask], axis=1)
                    token_type_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor([0,0], dtype=tf.dtypes.int32), axis=0),token_type_ids], axis=1)
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None
            else:
                for idx, example, label in self.BoolQ_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids = example["input_ids"]
                    attention_mask = example["attention_mask"]
                    token_type_ids = example["token_type_ids"]
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None

        def WiC_loader(head_aux_token: str, batch_size=1):
            if self.is_aux_toks:
                aux_toks = [self._default_bert_large_cased_tokenizer(head_aux_token),
                           self._default_bert_large_cased_tokenizer(self.format_aux_toks_dict[self._format_dataset_lookup_table("WiC")])]
                for idx, example, label in self.WiC_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids, attention_mask, token_type_ids = example["input_ids"], example["attention_mask"], example["token_type_ids"]
                    input_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor(aux_toks, dtype=tf.dtypes.int32), axis=0),input_ids], axis=1)
                    attention_mask = tf.concat([tf.expand_dims(tf.convert_to_tensor([1,1], dtype=tf.dtypes.int32), axis=0),attention_mask], axis=1)
                    token_type_ids = tf.concat([tf.expand_dims(tf.convert_to_tensor([0,0], dtype=tf.dtypes.int32), axis=0),token_type_ids], axis=1)
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None
            else:
                for idx, example, label in self.WiC_loader(batch_size=batch_size):
                    if idx is None or example is None or label is None: break  # technically this line isn't really required here as boolQ dataloader doesn't need to return None.
                    input_ids = example["input_ids"]
                    attention_mask = example["attention_mask"]
                    token_type_ids = example["token_type_ids"]
                    yield input_ids, attention_mask, token_type_ids, label
                yield None, None, None, None 

        self.general_CB_loader = CB_loader(head_aux_token=self.head_aux_toks_dict["entailment-3"])
        self.general_MPE_loader = MPE_loader(head_aux_token=self.head_aux_toks_dict["entailment-3"])

        self.general_RTE_loader = RTE_loader(head_aux_token=self.head_aux_toks_dict["entailment-2"])
        self.general_SciTail_loader = SciTail_loader(head_aux_token=self.head_aux_toks_dict["entailment-2"])

        self.general_COPA_loader = COPA_loader(head_aux_token=self.head_aux_toks_dict["commonsense"])
        self.general_CQA_loader = CQA_loader(head_aux_token=self.head_aux_toks_dict["commonsense"])
        self.general_WSC_loader = WSC_loader(head_aux_token=self.head_aux_toks_dict["commonsense"])

        self.general_MultiRC_loader = MultiRC_loader(head_aux_token=self.head_aux_toks_dict["readingcomprehension"])
        self.general_ReCoRD_loader = ReCoRD_loader(head_aux_token=self.head_aux_toks_dict["readingcomprehension"])
        self.general_RACE_loader = RACE_loader(head_aux_token=self.head_aux_toks_dict["readingcomprehension"])

        self.general_BoolQ_loader = BoolQ_loader(head_aux_token=self.head_aux_toks_dict["yes/no"])

        self.general_WiC_loader = WiC_loader(head_aux_token=self.head_aux_toks_dict["word-in-context"])

        def get_head_datasets_1(): # entailment-3
            rand_ = random.randrange(1,3)
            if rand_ == 1: # CB
                if self.is_diagnostics: print(f"CB")
                self.CB_counter += 1
                input_ids, attention_mask, token_type_ids, label = next(self.general_CB_loader)
                if input_ids is None: # then re-load the dataloader. It will be shuffled again as this is a new call.
                    self.general_CB_loader = CB_loader(head_aux_token=self.head_aux_toks_dict["entailment-3"])
                    input_ids, attention_mask, token_type_ids, label = next(self.general_CB_loader)
                return input_ids, attention_mask, token_type_ids, label
            elif rand_ == 2: # MPE
                if self.is_diagnostics: print(f"MPE")
                self.MPE_counter += 1
                input_ids, attention_mask, token_type_ids, label = next(self.general_MPE_loader)
                if input_ids is None:  # then re-load the dataloader. It will be shuffled again as this is a new call.
                    self.general_MPE_loader = MPE_loader(head_aux_token=self.head_aux_toks_dict["entailment-3"])
                    input_ids, attention_mask, token_type_ids, label = next(self.general_MPE_loader)
                return input_ids, attention_mask, token_type_ids, label

        def get_head_datasets_2(): # entailment-2
            rand_ = random.randrange(1,3)
            if rand_ == 1: # RTE
                if self.is_diagnostics: print(f"RTE")
                self.RTE_counter += 1
                input_ids, attention_mask, token_type_ids, label = next(self.general_RTE_loader)
                if input_ids is None:  # then re-load the dataloader. It will be shuffled again as this is a new call.
                    self.general_RTE_loader = RTE_loader(head_aux_token=self.head_aux_toks_dict["entailment-2"])
                    input_ids, attention_mask, token_type_ids, label = next(self.general_RTE_loader)
                return input_ids, attention_mask, token_type_ids, label
            elif rand_ == 2: # SciTail
                if self.is_diagnostics: print(f"SciTail")
                self.SciTail_counter += 1
                input_ids, attention_mask, token_type_ids, label = next(self.general_SciTail_loader)
                if input_ids is None:  # then re-load the dataloader. It will be shuffled again as this is a new call.
                    self.general_SciTail_loader = SciTail_loader(head_aux_token=self.head_aux_toks_dict["entailment-2"])
                    input_ids, attention_mask, token_type_ids, label = next(self.general_SciTail_loader)
                return input_ids, attention_mask, token_type_ids, label

        def get_head_datasets_3(): # commonsense
            rand_ = random.randrange(1,4)
            if rand_ == 1: # COPA
                if self.is_diagnostics: print(f"COPA")
                self.COPA_counter += 1
                input_ids, attention_mask, token_type_ids, label = next(self.general_COPA_loader)
                if input_ids is None:  # then re-load the dataloader. It will be shuffled again as this is a new call.
                    self.general_COPA_loader = COPA_loader(head_aux_token=self.head_aux_toks_dict["commonsense"])
                    input_ids, attention_mask, token_type_ids, label = next(self.general_COPA_loader)
                return input_ids, attention_mask, token_type_ids, label
            elif rand_ == 2: # CQA
                if self.is_diagnostics: print(f"CQA")
                self.CQA_counter += 1
                input_ids, attention_mask, token_type_ids, label = next(self.general_CQA_loader)
                if input_ids is None:  # then re-load the dataloader. It will be shuffled again as this is a new call.
                    self.general_CQA_loader = CQA_loader(head_aux_token=self.head_aux_toks_dict["commonsense"])
                    input_ids, attention_mask, token_type_ids, label = next(self.general_CQA_loader)
                return input_ids, attention_mask, token_type_ids, label
            elif rand_ == 3: # WSC
                if self.is_diagnostics: print(f"WSC")
                self.WSC_counter += 1
                input_ids, attention_mask, token_type_ids, label = next(self.general_WSC_loader)
                if input_ids is None:  # then re-load the dataloader. It will be shuffled again as this is a new call.
                    self.general_WSC_loader = WSC_loader(head_aux_token=self.head_aux_toks_dict["commonsense"])
                    input_ids, attention_mask, token_type_ids, label = next(self.general_WSC_loader)
                return input_ids, attention_mask, token_type_ids, label

        def get_head_datasets_4(): # readingcomprehension
            rand_ = random.randrange(1,4)
            if rand_ == 1: # MultiRC
                if self.is_diagnostics: print(f"MultiRC")
                self.MultiRC_counter += 1
                input_ids, attention_mask, token_type_ids, label = next(self.general_MultiRC_loader)
                if input_ids is None:  # then re-load the dataloader. It will be shuffled again as this is a new call.
                    self.general_MultiRC_loader = MultiRC_loader(head_aux_token=self.head_aux_toks_dict["readingcomprehension"])
                    input_ids, attention_mask, token_type_ids, label = next(self.general_MultiRC_loader)
                return input_ids, attention_mask, token_type_ids, label
            elif rand_ == 2: # ReCoRD
                if self.is_diagnostics: print(f"ReCoRD")
                self.ReCoRD_counter += 1
                input_ids, attention_mask, token_type_ids, label = next(self.general_ReCoRD_loader)
                if input_ids is None:  # then re-load the dataloader. It will be shuffled again as this is a new call.
                    self.general_ReCoRD_loader = ReCoRD_loader(head_aux_token=self.head_aux_toks_dict["readingcomprehension"])
                    input_ids, attention_mask, token_type_ids, label = next(self.general_ReCoRD_loader)
                return input_ids, attention_mask, token_type_ids, label
            elif rand_ == 3: # RACE
                if self.is_diagnostics: print(f"RACE")
                self.RACE_counter += 1
                input_ids, attention_mask, token_type_ids, label = next(self.general_RACE_loader)
                if input_ids is None:  # then re-load the dataloader. It will be shuffled again as this is a new call.
                    self.general_RACE_loader = RACE_loader(head_aux_token=self.head_aux_toks_dict["readingcomprehension"])
                    input_ids, attention_mask, token_type_ids, label = next(self.general_RACE_loader)
                return input_ids, attention_mask, token_type_ids, label

        def get_head_datasets_5(): # yes/no
            if self.is_diagnostics: print(f"BoolQ")
            self.BoolQ_counter += 1
            input_ids, attention_mask, token_type_ids, label = next(self.general_BoolQ_loader)
            if input_ids is None:  # then re-load the dataloader. It will be shuffled again as this is a new call.
                self.general_BoolQ_loader = BoolQ_loader(head_aux_token=self.head_aux_toks_dict["yes/no"])
                input_ids, attention_mask, token_type_ids, label = next(self.general_BoolQ_loader)
            return input_ids, attention_mask, token_type_ids, label

        def get_head_datasets_6(): # word-in-context
            if self.is_diagnostics: print(f"WiC")
            self.WiC_counter += 1
            input_ids, attention_mask, token_type_ids, label = next(self.general_WiC_loader)
            if input_ids is None:  # then re-load the dataloader. It will be shuffled again as this is a new call.
                self.general_WiC_loader = WiC_loader(head_aux_token=self.head_aux_toks_dict["word-in-context"])
                input_ids, attention_mask, token_type_ids, label = next(self.general_WiC_loader)
            return input_ids, attention_mask, token_type_ids, label

        for i in range(self.num_iterations):

            head_number = random.randrange(1, len(self.head_aux_toks_dict.keys())+1) # 1 to 6

            binput_ids, battention_mask, btoken_type_ids, blabel = None, None, None, None
            for j in range(self.batch_size):

                if head_number == 1: # entailment-3
                    input_ids, attention_mask, token_type_ids, label = get_head_datasets_1()
                    self.head1_counter += 1
                elif head_number == 2: # entailment-2
                    input_ids, attention_mask, token_type_ids, label = get_head_datasets_2()
                    self.head2_counter += 1
                elif head_number == 3: # commonsense
                    input_ids, attention_mask, token_type_ids, label = get_head_datasets_3()
                    self.head3_counter += 1
                elif head_number == 4: # readingcomprehension
                    input_ids, attention_mask, token_type_ids, label = get_head_datasets_4()
                    self.head4_counter += 1
                elif head_number == 5: # yes/no
                    input_ids, attention_mask, token_type_ids, label = get_head_datasets_5()
                    self.head5_counter += 1
                elif head_number == 6: # word-in-context
                    input_ids, attention_mask, token_type_ids, label = get_head_datasets_6()
                    self.head6_counter += 1
                else: raise Exception(f"Invalid head number: {head_number}!\nThis should never be reached!")

                if binput_ids is None:
                    #print(input_ids.shape)
                    assert input_ids.shape[0] == 1, f"The batch size should be 1, got {input_ids.shape[0]}!"
                    if self.is_aux_toks:
                        assert input_ids.shape[1] == 2+self.max_seq_len, f"The sequence length should be " \
                                                                         f"{2+self.max_seq_len}, got {input_ids.shape[1]}!"
                    binput_ids, battention_mask, btoken_type_ids, blabel = input_ids, attention_mask, token_type_ids, label
                else:
                    if self.is_aux_toks:
                        assert input_ids.shape[1] == 2+self.max_seq_len, f"The sequence length should be " \
                                                                         f"{2+self.max_seq_len}, got {input_ids.shape[1]}!"
                    binput_ids = tf.concat([binput_ids, input_ids], axis=0) # axis 0 is the batch dimension.
                    battention_mask = tf.concat([battention_mask, attention_mask], axis=0) # axis 0 is the batch dimension.
                    btoken_type_ids = tf.concat([btoken_type_ids, token_type_ids], axis=0) # axis 0 is the batch dimension.
                    blabel = tf.concat([blabel, label], axis=0) # axis 0 is the batch dimension.

            # +2 b/c of the two auxiliary tokens.
            mslen = self.max_seq_len+2 if self.is_aux_toks else self.max_seq_len
            assert binput_ids.shape[0] == self.batch_size and binput_ids.shape[1] == mslen
            assert battention_mask.shape[0] == self.batch_size and battention_mask.shape[1] == mslen
            assert btoken_type_ids.shape[0] == self.batch_size and btoken_type_ids.shape[1] == mslen
            assert blabel.shape[0] == self.batch_size # depending on the head, the second dimension can change.
            # head_number
            head_number_ = tf.cast(tf.convert_to_tensor(head_number), dtype=tf.dtypes.int8)
            yield binput_ids, battention_mask, btoken_type_ids, blabel, head_number_

        print(f"head1_counter: {self.head1_counter}\n"
              f"CB_counter: {self.CB_counter}\tMPE_counter: {self.MPE_counter}\n"
              f"head2_counter: {self.head2_counter}\n"
              f"RTE_counter: {self.RTE_counter}\tSciTail_counter: {self.SciTail_counter}\n"
              f"head3_counter: {self.head3_counter}\n"
              f"COPA_counter: {self.COPA_counter}\tCQA_counter: {self.CQA_counter}\tWSC_counter: {self.WSC_counter}\n"
              f"head4_counter: {self.head4_counter}\n"
              f"MultiRC_counter: {self.MultiRC_counter}\tReCoRD_counter: {self.ReCoRD_counter}\tRACE_counter: {self.RACE_counter}\n"
              f"head5_counter: {self.head5_counter}\n"
              f"BoolQ_counter: {self.BoolQ_counter}\n"
              f"head6_counter: {self.head6_counter}\n"
              f"WiC_counter: {self.WiC_counter}\n")

    def _head_aux_tokens(self) -> dict:
        return {"entailment-3":"[ENT-3]", "entailment-2":"[ENT-2]",
                "commonsense":"[CS]", "readingcomprehension":"[RC]",
                "yes/no":"[Y/N]", "word-in-context":"[WIC]"}

    def _format_aux_tokens(self) -> dict:
        return {"text-wordword":"[T+WW]", "passage-questionanswer":"[P+QA]",
                "premise-hypothesis":"[P+H]", "passage-question":"[P+Q]",
                "stem-text":"[S+T]", "text1text2-word":"[TT+W]",
                "premise-hypothesisanswer":"[P+HA]"}

    def _format_dataset_lookup_table(self, dataset: str) -> str:
        if dataset == "RTE" or dataset == "SciTail": return "premise-hypothesis"
        elif dataset == "CB" or dataset == "MPE": return "premise-hypothesisanswer" # assumes sigmoid mode
        elif dataset == "BoolQ": return "passage-question"
        elif dataset == "CQA": return "stem-text"
        elif dataset == "WiC": return "text1text2-word"
        elif dataset  == "WSC": return "text-wordword"
        elif dataset == "COPA" or dataset == "MultiRC" or dataset == "ReCoRD" or dataset == "RACE":
            return "passage-questionanswer"
        else: raise Exception(f"Invalid dataset: {dataset}!")

    def _default_bert_large_cased_tokenizer(self, aux_tok: str) -> int:
        # convert each token to an unused idx.
        if aux_tok == "[ENT-3]": return 1#"[unused1]"
        elif aux_tok == "[ENT-2]": return 2#"[unused2]"
        elif aux_tok == "[CS]": return 3#"[unused3]"
        elif aux_tok == "[RC]": return 4#"[unused4]"
        elif aux_tok == "[Y/N]": return 5#"[unused5]"
        elif aux_tok == "[WIC]": return 6#"[unused6]"
        elif aux_tok == "[T+WW]": return 7#"[unused7]"
        elif aux_tok == "[P+QA]": return 8#"[unused8]"
        elif aux_tok == "[P+H]": return 9#"[unused9]"
        elif aux_tok == "[P+HA]": return 10#"[unused10]"
        elif aux_tok == "[P+Q]": return 11#"[unused11]"
        elif aux_tok == "[S+T]": return 12#"[unused12]"
        elif aux_tok == "[TT+W]": return 13#"[unused13]"
        else: raise Exception(f"invalid aux_token: {aux_tok}!")

    def get_generalQA_generator(self, batch_size:int, head_strategy:str="uniform", dataset_strategy:str="uniform",
                                shuffle_datasets:bool=True, type:str="default", num_iterations: int=400000,
                                is_aux_toks: bool=True):

        self.batch_size = batch_size
        self.head_strategy = head_strategy # currently not in use!
        self.dataset_strategy = dataset_strategy # currently not in use!
        self.shuffle_datasets = shuffle_datasets
        self.type = type
        self.num_iterations = num_iterations
        self.is_aux_toks = is_aux_toks

        generator = None

        #self.general_CB_loader = CB_loader(head_aux_token=self.head_aux_toks_dict["entailment-3"])
        #self.general_MPE_loader = MPE_loader(head_aux_token=self.head_aux_toks_dict["entailment-3"])

        #self.general_RTE_loader = RTE_loader(head_aux_token=self.head_aux_toks_dict["entailment-2"])
        #self.general_SciTail_loader = SciTail_loader(head_aux_token=self.head_aux_toks_dict["entailment-2"])

        #self.general_COPA_loader = COPA_loader(head_aux_token=self.head_aux_toks_dict["commonsense"])
        #self.general_CQA_loader = CQA_loader(head_aux_token=self.head_aux_toks_dict["commonsense"])
        #self.general_WSC_loader = WSC_loader(head_aux_token=self.head_aux_toks_dict["commonsense"])

        #self.general_MultiRC_loader = MultiRC_loader(head_aux_token=self.head_aux_toks_dict["readingcomprehension"])
        #self.general_ReCoRD_loader = ReCoRD_loader(head_aux_token=self.head_aux_toks_dict["readingcomprehension"])
        #self.general_RACE_loader = RACE_loader(head_aux_token=self.head_aux_toks_dict["readingcomprehension"])

        #self.general_BoolQ_loader = BoolQ_loader(head_aux_token=self.head_aux_toks_dict["yes/no"])

        #self.general_WiC_loader = WiC_loader(head_aux_token=self.head_aux_toks_dict["word-in-context"])

        if self.type == "default":

            assert self.is_CB is True and self.is_MPE is True and self.is_RTE is True and self.is_COPA is True \
            and self.is_CQA is True and self.is_WSC is True and self.is_MultiRC is True and self.is_ReCoRD is True \
            and self.is_RACE is True and self.is_BoolQ is True and self.is_WiC is True and self.is_SciTail is True

            # note: is_test is always False.

            generator = tf.data.Dataset.from_generator(self.get_GQA_generator_default,
                                                       output_types=(tf.dtypes.int32, # input_ids
                                                                     tf.dtypes.int32, # attention_mask
                                                                     tf.dtypes.int32, # token_type_ids
                                                                     tf.dtypes.int8,  # label (ones and zeroes).
                                                                     tf.dtypes.int8)) # head_num

        elif self.type == "ablation":
            assert self.is_CB is True and self.is_COPA is True and self.is_BoolQ is True and self.is_WiC is True

            generator = tf.data.Dataset.from_generator(self.get_GQA_generator_for_ablation,
                                                        output_types=(tf.dtypes.int32, # input_ids
                                                                      tf.dtypes.int32, # attention_mask
                                                                      tf.dtypes.int32, # token_type_ids
                                                                      tf.dtypes.int8,  # label (ones and zeroes)
                                                                      tf.dtypes.int8)) # head_num

        elif self.type == "CB_test":
            assert self.is_test_CB
            assert self.is_CB
            assert self.shuffle_datasets is False
            self.CB_loader.shuffle = self.shuffle_datasets
            head_aux_toks_dict = self._head_aux_tokens()
            self.head_aux_token = head_aux_toks_dict["entailment-3"]
            self.format_aux_toks_dict = self._format_aux_tokens()
            generator = tf.data.Dataset.from_generator(self.get_CB_aux_toks,
                                                       output_types=(tf.dtypes.int32,  # idx
                                                                     tf.dtypes.int32,  # input_ids
                                                                     tf.dtypes.int32,  # attention_mask
                                                                     tf.dtypes.int32))  # token_type_ids
        elif self.type == "SNLI_test":
            assert self.is_test_SNLI
            assert self.is_SNLI
            assert self.shuffle_datasets is False
            self.SNLI_loader.shuffle = self.shuffle_datasets
            head_aux_toks_dict = self._head_aux_tokens()
            self.head_aux_token = head_aux_toks_dict["entailment-3"]
            self.format_aux_toks_dict = self._format_aux_tokens()
            generator = tf.data.Dataset.from_generator(self.get_SNLI_aux_toks,
                                                       output_types=(tf.dtypes.int32,  # idx
                                                                     tf.dtypes.int32,  # input_ids
                                                                     tf.dtypes.int32,  # attention_mask
                                                                     tf.dtypes.int32))  # token_type_ids

        elif self.type == "MPE_test":
            assert self.is_test_MPE
            assert self.is_MPE
            assert self.shuffle_datasets is False
            self.MPE_loader.shuffle = self.shuffle_datasets
            head_aux_toks_dict = self._head_aux_tokens()
            self.head_aux_token = head_aux_toks_dict["entailment-3"]
            self.format_aux_toks_dict = self._format_aux_tokens()
            generator = tf.data.Dataset.from_generator(self.get_MPE_aux_toks,
                                                       output_types=(tf.dtypes.int32,  # idx
                                                                     tf.dtypes.int32,  # input_ids
                                                                     tf.dtypes.int32,  # attention_mask
                                                                     tf.dtypes.int32))  # token_type_ids
        elif self.type == "RTE_test":
            assert self.is_test_RTE
            assert self.is_RTE
            assert self.shuffle_datasets is False
            self.RTE_loader.shuffle = self.shuffle_datasets
            head_aux_toks_dict = self._head_aux_tokens()
            self.head_aux_token = head_aux_toks_dict["entailment-2"]
            self.format_aux_toks_dict = self._format_aux_tokens()
            generator = tf.data.Dataset.from_generator(self.get_RTE_aux_toks,
                                                       output_types=(tf.dtypes.int32,  # idx
                                                                     tf.dtypes.int32,  # input_ids
                                                                     tf.dtypes.int32,  # attention_mask
                                                                     tf.dtypes.int32))  # token_type_ids
        elif self.type == "MED_test":
            assert self.is_test_MED
            assert self.is_MED
            assert self.shuffle_datasets is False
            self.MED_loader.shuffle = self.shuffle_datasets
            head_aux_toks_dict = self._head_aux_tokens()
            self.head_aux_token = head_aux_toks_dict["entailment-2"]
            self.format_aux_toks_dict = self._format_aux_tokens()
            generator = tf.data.Dataset.from_generator(self.get_MED_aux_toks,
                                                       output_types=(tf.dtypes.int32,  # idx
                                                                     tf.dtypes.int32,  # input_ids
                                                                     tf.dtypes.int32,  # attention_mask
                                                                     tf.dtypes.int32))  # token_type_ids

        elif self.type == "COPA_test":
            assert self.is_test_COPA
            assert self.is_COPA
            assert self.shuffle_datasets is False
            self.COPA_loader.shuffle = self.shuffle_datasets
            head_aux_toks_dict = self._head_aux_tokens()
            self.head_aux_token = head_aux_toks_dict["commonsense"]
            self.format_aux_toks_dict = self._format_aux_tokens()
            generator = tf.data.Dataset.from_generator(self.get_COPA_aux_toks,
                                                       output_types=(tf.dtypes.int32,  # idx
                                                                     tf.dtypes.int32,  # input_ids
                                                                     tf.dtypes.int32,  # attention_mask
                                                                     tf.dtypes.int32))  # token_type_ids
        elif self.type == "CQA_test":
            assert self.is_test_CQA
            assert self.is_CQA
            assert self.shuffle_datasets is False
            self.CQA_loader.shuffle = self.shuffle_datasets
            head_aux_toks_dict = self._head_aux_tokens()
            self.head_aux_token = head_aux_toks_dict["commonsense"]
            self.format_aux_toks_dict = self._format_aux_tokens()
            generator = tf.data.Dataset.from_generator(self.get_CQA_aux_toks,
                                                       output_types=(tf.dtypes.string,  # idx
                                                                     tf.dtypes.int32,  # input_ids
                                                                     tf.dtypes.int32,  # attention_mask
                                                                     tf.dtypes.int32))  # token_type_ids

        elif self.type == "PIQA_test":
            assert self.is_test_PIQA
            assert self.is_PIQA
            assert self.shuffle_datasets is False
            self.PIQA_loader.shuffle = self.shuffle_datasets
            head_aux_toks_dict = self._head_aux_tokens()
            self.head_aux_token = head_aux_toks_dict["commonsense"]
            self.format_aux_toks_dict = self._format_aux_tokens()
            generator = tf.data.Dataset.from_generator(self.get_PIQA_aux_toks,
                                                       output_types=(tf.dtypes.int32,  # idx
                                                                     tf.dtypes.int32,  # input_ids
                                                                     tf.dtypes.int32,  # attention_mask
                                                                     tf.dtypes.int32))  # token_type_ids

        elif self.type == "SIQA_test":
            assert self.is_test_SIQA
            assert self.is_SIQA
            assert self.shuffle_datasets is False
            self.SIQA_loader.shuffle = self.shuffle_datasets
            head_aux_toks_dict = self._head_aux_tokens()
            self.head_aux_token = head_aux_toks_dict["commonsense"]
            self.format_aux_toks_dict = self._format_aux_tokens()
            generator = tf.data.Dataset.from_generator(self.get_SIQA_aux_toks,
                                                       output_types=(tf.dtypes.int32,  # idx
                                                                     tf.dtypes.int32,  # input_ids
                                                                     tf.dtypes.int32,  # attention_mask
                                                                     tf.dtypes.int32))  # token_type_ids

        elif self.type == "WSC_test":
            assert self.is_test_WSC
            assert self.is_WSC
            assert self.shuffle_datasets is False
            self.WSC_loader.shuffle = self.shuffle_datasets
            head_aux_toks_dict = self._head_aux_tokens()
            self.head_aux_token = head_aux_toks_dict["commonsense"]
            self.format_aux_toks_dict = self._format_aux_tokens()
            generator = tf.data.Dataset.from_generator(self.get_WSC_aux_toks,
                                                       output_types=(tf.dtypes.int32,  # idx
                                                                     tf.dtypes.int32,  # input_ids
                                                                     tf.dtypes.int32,  # attention_mask
                                                                     tf.dtypes.int32))  # token_type_ids
        elif self.type == "MultiRC_test":
            assert self.is_test_MultiRC
            assert self.is_MultiRC
            assert self.shuffle_datasets is False
            self.MultiRC_loader.shuffle = self.shuffle_datasets
            head_aux_toks_dict = self._head_aux_tokens()
            self.head_aux_token = head_aux_toks_dict["readingcomprehension"]
            self.format_aux_toks_dict = self._format_aux_tokens()
            generator = tf.data.Dataset.from_generator(self.get_MultiRC_aux_toks,
                                                       output_types=(tf.dtypes.int32,  # idx
                                                                     tf.dtypes.int32,  # input_ids
                                                                     tf.dtypes.int32,  # attention_mask
                                                                     tf.dtypes.int32))  # token_type_ids
        elif self.type == "ReCoRD_test":
            assert self.is_test_ReCoRD
            assert self.is_ReCoRD
            assert self.shuffle_datasets is False
            self.ReCoRD_loader.shuffle = self.shuffle_datasets
            head_aux_toks_dict = self._head_aux_tokens()
            self.head_aux_token = head_aux_toks_dict["readingcomprehension"]
            self.format_aux_toks_dict = self._format_aux_tokens()
            generator = tf.data.Dataset.from_generator(self.get_ReCoRD_aux_toks,
                                                       output_types=(tf.dtypes.int32,  # idx
                                                                     tf.dtypes.int32,  # input_ids
                                                                     tf.dtypes.int32,  # attention_mask
                                                                     tf.dtypes.int32))  # token_type_ids
        elif self.type == "RACE_test":
            assert self.is_test_RACE
            assert self.is_RACE
            assert self.shuffle_datasets is False
            self.RACE_loader.shuffle = self.shuffle_datasets
            head_aux_toks_dict = self._head_aux_tokens()
            self.head_aux_token = head_aux_toks_dict["readingcomprehension"]
            self.format_aux_toks_dict = self._format_aux_tokens()
            generator = tf.data.Dataset.from_generator(self.get_RACE_aux_toks,
                                                       output_types=(tf.dtypes.string,  # idx
                                                                     tf.dtypes.int32,  # input_ids
                                                                     tf.dtypes.int32,  # attention_mask
                                                                     tf.dtypes.int32))  # token_type_ids

        elif self.type == "ReClor_test":
            assert self.is_test_ReClor
            assert self.is_ReClor
            assert self.shuffle_datasets is False
            self.ReClor_loader.shuffle = self.shuffle_datasets
            head_aux_toks_dict = self._head_aux_tokens()
            self.head_aux_token = head_aux_toks_dict["readingcomprehension"]
            self.format_aux_toks_dict = self._format_aux_tokens()
            generator = tf.data.Dataset.from_generator(self.get_ReClor_aux_toks,
                                                       output_types=(tf.dtypes.string,  # idx
                                                                     tf.dtypes.int32,  # input_ids
                                                                     tf.dtypes.int32,  # attention_mask
                                                                     tf.dtypes.int32))  # token_type_ids

        elif self.type == "DREAM_test":
            assert self.is_test_DREAM
            assert self.is_DREAM
            assert self.shuffle_datasets is False
            self.DREAM_loader.shuffle = self.shuffle_datasets
            head_aux_toks_dict = self._head_aux_tokens()
            self.head_aux_token = head_aux_toks_dict["readingcomprehension"]
            self.format_aux_toks_dict = self._format_aux_tokens()
            generator = tf.data.Dataset.from_generator(self.get_DREAM_aux_toks,
                                                       output_types=(tf.dtypes.string,  # idx
                                                                     tf.dtypes.int32,  # input_ids
                                                                     tf.dtypes.int32,  # attention_mask
                                                                     tf.dtypes.int32))  # token_type_ids

        elif self.type == "StrategyQA_test":
            assert self.is_test_StrategyQA
            assert self.is_StrategyQA
            assert self.shuffle_datasets is False
            self.StrategyQA_loader.shuffle = self.shuffle_datasets
            head_aux_toks_dict = self._head_aux_tokens()
            self.head_aux_token = head_aux_toks_dict["yes/no"]
            self.format_aux_toks_dict = self._format_aux_tokens()
            generator = tf.data.Dataset.from_generator(self.get_StrategyQA_aux_toks,
                                                       output_types=(tf.dtypes.string,  # idx
                                                                     tf.dtypes.int32,  # input_ids
                                                                     tf.dtypes.int32,  # attention_mask
                                                                     tf.dtypes.int32))  # token_type_ids

        elif self.type == "BoolQ_test":
            assert self.is_test_BoolQ
            assert self.is_BoolQ
            assert self.shuffle_datasets is False
            #aux_toks = [self._default_bert_large_cased_tokenizer(head_aux_token),
            #            self._default_bert_large_cased_tokenizer(
            #                self.format_aux_toks_dict[self._format_dataset_lookup_table("BoolQ")])]
            self.BoolQ_loader.shuffle = self.shuffle_datasets
            #self.head_number = 5
            head_aux_toks_dict = self._head_aux_tokens()
            self.head_aux_token = head_aux_toks_dict["yes/no"]
            self.format_aux_toks_dict = self._format_aux_tokens()
            generator = tf.data.Dataset.from_generator(self.get_BoolQ_aux_toks,
                                                       output_types=(tf.dtypes.int32, # idx
                                                                     tf.dtypes.int32, # input_ids
                                                                     tf.dtypes.int32, # attention_mask
                                                                     tf.dtypes.int32)) # token_type_ids
                                                                     #tf.dtypes.int8,  # label (ones and zeroes).
                                                                     #tf.dtypes.int8)) # head_num
        elif self.type == "WiC_test":
            assert self.is_test_WiC
            assert self.is_WiC
            assert self.shuffle_datasets is False
            self.WiC_loader.shuffle = self.shuffle_datasets
            head_aux_toks_dict = self._head_aux_tokens()
            self.head_aux_token = head_aux_toks_dict["word-in-context"]
            self.format_aux_toks_dict = self._format_aux_tokens()
            generator = tf.data.Dataset.from_generator(self.get_WiC_aux_toks,
                                                       output_types=(tf.dtypes.int32,  # idx
                                                                     tf.dtypes.int32,  # input_ids
                                                                     tf.dtypes.int32,  # attention_mask
                                                                     tf.dtypes.int32))  # token_type_ids
        elif self.type == "SciTail_test":
            assert self.is_test_SciTail
            assert self.is_SciTail
            assert self.shuffle_datasets is False
            self.SciTail_loader.shuffle = self.shuffle_datasets
            head_aux_toks_dict = self._head_aux_tokens()
            self.head_aux_token = head_aux_toks_dict["entailment-2"]
            self.format_aux_toks_dict = self._format_aux_tokens()
            generator = tf.data.Dataset.from_generator(self.get_SciTail_aux_toks,
                                                       output_types=(tf.dtypes.int32,  # idx
                                                                     tf.dtypes.int32,  # input_ids
                                                                     tf.dtypes.int32,  # attention_mask
                                                                     tf.dtypes.int32))  # token_type_ids
        else: raise Exception(f"Invlaid type: {self.type}!")


        return generator


    def get_generators(self, batch_size: int, shuffle: bool, type: str):

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.type = type

        generator = None

        if type == "BoolQ":
            if not self.is_test_BoolQ:
                generator = tf.data.Dataset.from_generator(self.get_BoolQ_loader_generator,
                                                           output_types=(tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int8))
            else:
                generator = tf.data.Dataset.from_generator(self.get_BoolQ_loader_generator,
                                                           output_types=(tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32))
        elif type == "CB":
            if not self.is_test_CB:
                generator = tf.data.Dataset.from_generator(self.get_CB_loader_generator,
                                                           output_types=(tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int8))
            else:
                generator = tf.data.Dataset.from_generator(self.get_CB_loader_generator,
                                                           output_types=(tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32))

        elif type == "COPA":
            if not self.is_test_COPA:
                generator = tf.data.Dataset.from_generator(self.get_COPA_loader_generator,
                                                           output_types=(tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int8))
            else:
                generator = tf.data.Dataset.from_generator(self.get_COPA_loader_generator,
                                                           output_types=(tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32))

        elif type == "MultiRC":
            if not self.is_test_MultiRC:
                generator = tf.data.Dataset.from_generator(self.get_MultiRC_loader_generator,
                                                           output_types=(tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int8))
            else:
                generator = tf.data.Dataset.from_generator(self.get_MultiRC_loader_generator,
                                                           output_types=(tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32))

        elif type == "ReCoRD":
            if not self.is_test_ReCoRD:
                generator = tf.data.Dataset.from_generator(self.get_ReCoRD_loader_generator,
                                                           output_types=(tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int8))
            else:
                generator = tf.data.Dataset.from_generator(self.get_ReCoRD_loader_generator,
                                                           output_types=(tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32))

        elif type == "RTE":
            if not self.is_test_RTE:
                generator = tf.data.Dataset.from_generator(self.get_RTE_loader_generator,
                                                           output_types=(tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int8))
            else:
                generator = tf.data.Dataset.from_generator(self.get_RTE_loader_generator,
                                                           output_types=(tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32))

        elif type == "WiC":
            if not self.is_test_WiC:
                generator = tf.data.Dataset.from_generator(self.get_WiC_loader_generator,
                                                           output_types=(tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int8))
            else:
                generator = tf.data.Dataset.from_generator(self.get_WiC_loader_generator,
                                                           output_types=(tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32))

        elif type == "WSC":
            if not self.is_test_WSC:
                generator = tf.data.Dataset.from_generator(self.get_WSC_loader_generator,
                                                           output_types=(tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int8))
            else:
                generator = tf.data.Dataset.from_generator(self.get_WSC_loader_generator,
                                                           output_types=(tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32))

        elif type == "SciTail":
            if not self.is_test_SciTail:
                generator = tf.data.Dataset.from_generator(self.get_SciTail_loader_generator,
                                                           output_types=(tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int8))
            else:
                generator = tf.data.Dataset.from_generator(self.get_SciTail_loader_generator,
                                                           output_types=(tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32))

        elif type == "CQA":
            if not self.is_test_CQA:
                generator = tf.data.Dataset.from_generator(self.get_CQA_loader_generator,
                                                           output_types=(tf.dtypes.string,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int8))
            else:
                generator = tf.data.Dataset.from_generator(self.get_CQA_loader_generator,
                                                           output_types=(tf.dtypes.string,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32))

        elif type == "RACE":
            if not self.is_test_RACE:
                generator = tf.data.Dataset.from_generator(self.get_RACE_loader_generator,
                                                           output_types=(tf.dtypes.string,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int8))
            else:
                generator = tf.data.Dataset.from_generator(self.get_RACE_loader_generator,
                                                           output_types=(tf.dtypes.string,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32))

        elif type == "MPE":
            if not self.is_test_MPE:
                generator = tf.data.Dataset.from_generator(self.get_MPE_loader_generator,
                                                           output_types=(tf.dtypes.string,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int8))
            else:
                generator = tf.data.Dataset.from_generator(self.get_MPE_loader_generator,
                                                           output_types=(tf.dtypes.string,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32,
                                                                         tf.dtypes.int32))

        return generator



if __name__ == "__main__":

    random.seed(24)
    batch_size = 8
    shuffle_datasets = True
    is_test = False
    num_iterations = 10000
    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
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
                                  max_seq_len=512, is_token_type_ids=True, is_diagnostics=True)

    for inp_ids, attn_mask, token_type_ids, label, head_number in dloader.get_generalQA_generator(batch_size, head_strategy="uniform",
                                                                                     dataset_strategy="uniform",
                                                                                     shuffle_datasets=shuffle_datasets,
                                                                                     type="default",
                                                                                     num_iterations=num_iterations,
                                                                                     is_aux_toks=True):

    #for inp_ids, attn_mask, token_type_ids, head_number in dloader.get_generalQA_generator(batch_size,
    #                                                                                    head_strategy="uniform",
    #                                                                                    dataset_strategy="uniform",
    #                                                                                    shuffle_datasets=shuffle_datasets,
    #                                                                                    type="BoolQ_test",
    #                                                                                    num_iterations=num_iterations,
    #                                                                                    is_aux_toks=True):
        print(f"inp_ids: {inp_ids}\n"
              f"attn_mask: {attn_mask}\n"
              f"attn_mask.shape: {attn_mask.shape}\n"
              f"token_type_ids: {token_type_ids}"
              f"example: {tokenizer.batch_decode(inp_ids)}\n"
              f"label: {label}\n"
              f"head_number: {head_number}")

    '''
    batch_size = 4
    shuffle = False
    is_test = True
    type_ = "MPE"

    tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
    dloader = general_data_loader(filepath_BoolQ="/large_data/SuperGlue/BoolQ/train.jsonl",
                                  filepath_CB="/large_data/SuperGlue/CB/train.jsonl",
                                  filepath_COPA="/large_data/SuperGlue/COPA/train.jsonl",
                                  filepath_MultiRC="/large_data/SuperGlue/MultiRC/train.jsonl",
                                  filepath_ReCoRD="/large_data/SuperGlue/ReCoRD/val.jsonl",
                                  filepath_RTE="/large_data/SuperGlue/RTE/train.jsonl",
                                  filepath_WiC="/large_data/SuperGlue/WiC/train.jsonl",
                                  filepath_WSC="/large_data/SuperGlue/WSC/train.jsonl",
                                  filepath_CQA="/large_data/CommonsenseQA/dev_rand_split.jsonl",
                                  filepath_RACE_middle="/large_data/RACE/RACE/train/middle/",
                                  filepath_RACE_high="/large_data/RACE/RACE/train/high/",
                                  filepath_SciTail="/large_data/SciTail/SciTailV1.1/predictor_format/"
                                                   "scitail_1.0_structure_dev.jsonl",
                                  filepath_MPE="/large_data/MPE/MultiPremiseEntailment-master/data/MPE/mpe_train.txt",
                                  tokenizer=tokenizer,
                 is_test_BoolQ=is_test, is_test_CB=is_test, is_test_COPA=is_test, is_test_MultiRC=is_test,
                 is_test_ReCoRD=is_test, is_test_RTE=is_test, is_test_WiC=is_test, is_test_WSC=is_test,
                 is_test_CQA=is_test, is_test_RACE=is_test, is_test_SciTail=is_test, is_test_MPE=is_test,
                 is_BoolQ=False, is_CB=False, is_COPA=False, is_MultiRC=False, is_ReCoRD=False, is_RTE=False,
                 is_WiC=False, is_WSC=False, is_CQA=False, is_RACE=False, is_SciTail=False, is_MPE=True,
                 error_function_cb="sigmoid")
                 #error_function_cb="softmax")

    if not is_test:
        for idx, inp_ids, attn_mask, token_type_ids, label in dloader.get_generators(batch_size=batch_size, shuffle=shuffle, type=type_):
            print(f"idx: {idx}\n"
                  f"inp_ids: {inp_ids}\n"
                  f"attn_mask: {attn_mask}\n"
                  f"token_type_ids: {token_type_ids}"
                  f"example: {tokenizer.batch_decode(inp_ids)}\n"
                  f"label: {label}")

    else:
        for idx, inp_ids, attn_mask, token_type_ids, in dloader.get_generators(batch_size=batch_size, shuffle=shuffle, type=type_):
            print(f"idx: {idx}\n"
                  f"inp_ids: {inp_ids}\n"
                  f"attn_mask: {attn_mask}\n"
                  f"token_type_ids: {token_type_ids}"
                  f"example: {tokenizer.batch_decode(inp_ids)}\n")
    '''
    '''
    if not is_test:
        for idx, example, label in dloader.get_BoolQ_loader(batch_size=batch_size, shuffle=shuffle):
            print(f"idx: {idx}\n"
                  f"example: {example}\n"
                  #f"example-input_ids: {example['input_ids']}\n"
                  f"example: {tokenizer.batch_decode(example['input_ids'])}\n"
                  f"label: {label}")

    else:
        for idx, example in dloader.get_BoolQ_loader(batch_size=batch_size, shuffle=shuffle):
            print(f"idx: {idx}\n"
                  f"example: {example}\n"
                  f"example: {tokenizer.batch_decode(example['input_ids'])}\n")
    '''