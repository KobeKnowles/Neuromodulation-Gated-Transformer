# Neuromodulation Gated Transformer

This repository contains the code used to conduct the experiments for our ICLR 2023 Tiny Paper submission for the 
paper titled "Neuromodulation Gated Transformer". This repository does not include the implementation of the 
model/architecture, which is instead stored in another GitHub repository (https://github.com/KobeKnowles/transformers-NGT).

## Data Loaders

All datasets are loaded via the files in the data_loaders folder. 

## Get Results

The code used to evaluate the predictions (i.e., get the metric scores) made by all models is in the get_test_results_final folder.

## Misc

The misc folder includes complementary code that is used throughout this repository.

## Training files

All files used for training (and prediction generation) are in the training_files folder.

### SuperGLUE Benchmark (Single Model)

The training and prediction files used to generate the results for Table 1 and 4 are in the superGLUE_10_epochs folder.

### Multi-Task Training Regime (other experiments not included Neuromodulation Gated Transformer paper)

The training and prediction files for the multi-task training regime are in the general_experiments folder.

### Ablation Study (other experiments not included Neuromodulation Gated Transformer paper)

The training and prediction files for the ablation study are in the ablation_aux_toks folder.

### Neuromodulation Probe (other experiments not included Neuromodulation Gated Transformer paper)

The code for the neuromodulation probe is in the probe_neuromodulation folder.

### Gating Start vs. End

The experiments regarding the comparison of the start and end positions is in the gating_start_vs_end folder.

## Dependencies

The exact environment we used to conduct experiments is shown in the dependnecies.txt file. The important modules are
a working TensorFlow 2.4.1 environment (including all relevant dependencies) and a custom 
transformers library with neuromodulation support (https://github.com/KobeKnowles/transformers-NGT).

The following can be used to install the custom transformers library:

pip install git+https://github.com/KobeKnowles/transformers-NGT@neuromodulation-gating

## Dataset Install Instructions

The dataset_download_instructions.txt includes instruction on how to download all datasets used in the experiments. 

## Custom Transformers Module for Neuromodulation

The custom transformers module with support for neuromodulation can be found at https://github.com/KobeKnowles/transformers-NGT. 
To view the changes to the code go to src/transformers/models/bert. We have modified the modeling_tf_bert.py and 
configuration_bert.py files with support for neuromodulation. For modeling_tf_bert.py go directly to line 530 (the 
class TfBertEncoder) to see the modifications to the code. Additionally, in the configuration_bert.py file we added
support for parameters relevant to neuromodulation that will be used by the bert model in modeling_tf_bert.py. 
Some of the changes made are for additional experiments not included in the Neuromodulation Gated Transformer paper.

## Instructions on How to Run Models

A bert model can be loaded nearly identical to that of the vanilla transformer library. Differences are in the bert config function, namely additional variables. Additional variables in the bert config file are: gating_block_num_layers (the number of layers in a gating block), gating_block_end (if there is a gating block at the end position), gating_block_end_position (the layer the end gating block is inserted after), gating_block_middle (if there is a gating block at the middle position), gating_block_middle_position (the layer the middle gating block is inserted after), gating_block_start (if there is a gating block at the end position), gating_block_start_position (the layer the start gating block is inserted after), nm_gating (if the gating block performs gating or acts as additional layers), cls_dense_layer_number_of_options (the number of output options), is_diagnostics (diagnostic details are printed out during execution), max_seq_len (the maximum sequence length of the input), num_aux_toks (the number of auxiliary tokens; should be set to 0 for the models used in the ICLR 2023 TinyPaper). 







