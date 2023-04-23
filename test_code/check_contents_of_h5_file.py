import numpy as np
import h5py

import tensorflow as tf
from tensorflow.python.keras.saving import hdf5_format

if __name__ == "__main__":
    hf = h5py.File('/data/kkno604/NGT_experiments_updated/general_experiments/'
                   'gating-end/exp1/Checkpoints/iteration50000/tf_model.h5', 'r')
    #hf = h5py.File('/data/kkno604/NGT_experiments_updated/superGLUE-experiments/BoolQ/bert-large/gating-end/'
    #               'exp1/Checkpoints/epoch1/tf_model.h5', 'r')

    #print(hf.keys())
    #print(np.array(hf.get("head1_dense")))
    print(hf["head2_dense"]["kernel:0"])
    print(hf["head2_dense"]["kernel:0"])
    print(np.array(hf["head2_dense"]["kernel:0"]))
    #print(np.array(hf.get("bert")))
    #print(hf["bert"]["tf_bert_model"])

    #print(hf["head1_dense"][:])
    #print(hf.attrs.get("head1_dense"))

    set_ = set(hdf5_format.load_attributes_from_hdf5_group(hf, "layer_names"))
    print(set_)

    hf.close()