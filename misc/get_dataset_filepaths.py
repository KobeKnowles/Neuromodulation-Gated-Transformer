def get_dataset_filepaths():

    filepaths_BoolQ = {"BoolQ_train": "/large_data/SuperGlue/BoolQ/train.jsonl",
                       "BoolQ_val": "/large_data/SuperGlue/BoolQ/val.jsonl",
                       "BoolQ_test": "/large_data/SuperGlue/BoolQ/test.jsonl"}

    filepaths_CB = {"CB_train": "/large_data/SuperGlue/CB/train.jsonl",
                    "CB_val": "/large_data/SuperGlue/CB/val.jsonl",
                    "CB_test": "/large_data/SuperGlue/CB/test.jsonl"}

    filepaths_COPA = {"COPA_train": "/large_data/SuperGlue/COPA/train.jsonl",
                      "COPA_val": "/large_data/SuperGlue/COPA/val.jsonl",
                      "COPA_test": "/large_data/SuperGlue/COPA/test.jsonl"}

    filepaths_MultiRC = {"MultiRC_train": "/large_data/SuperGlue/MultiRC/train.jsonl",
                         "MultiRC_val": "/large_data/SuperGlue/MultiRC/val.jsonl",
                         "MultiRC_test": "/large_data/SuperGlue/MultiRC/test.jsonl"}

    filepaths_ReCoRD = {"ReCoRD_train": "/large_data/SuperGlue/ReCoRD/train.jsonl",
                        "ReCoRD_val": "/large_data/SuperGlue/ReCoRD/val.jsonl",
                        "ReCoRD_test": "/large_data/SuperGlue/ReCoRD/test.jsonl"}

    filepaths_RTE = {"RTE_train": "/large_data/SuperGlue/RTE/train.jsonl",
                     "RTE_val": "/large_data/SuperGlue/RTE/val.jsonl",
                     "RTE_test": "/large_data/SuperGlue/RTE/test.jsonl"}

    filepaths_WiC = {"WiC_train": "/large_data/SuperGlue/WiC/train.jsonl",
                     "WiC_val": "/large_data/SuperGlue/WiC/val.jsonl",
                     "WiC_test": "/large_data/SuperGlue/WiC/test.jsonl"}

    filepaths_WSC = {"WSC_train": "/large_data/SuperGlue/WSC/train.jsonl",
                     "WSC_val": "/large_data/SuperGlue/WSC/val.jsonl",
                     "WSC_test": "/large_data/SuperGlue/WSC/test.jsonl"}

    return filepaths_BoolQ, filepaths_CB, filepaths_COPA, filepaths_MultiRC, filepaths_ReCoRD, \
           filepaths_RTE, filepaths_WiC, filepaths_WSC