# Neuromodulation Gated Transformer

This repository contains all of the code used to conduct the experiments for our ICLR 2023 Tiny Paper submission for the paper titled "Neuromodulation Gated Transformer". This repository does not include the implementation of the model/architecture, which 
is instead stored in another anonymous GitHub repository (https://anonymous.4open.science/r/XXXXX-transformers-XXXXXX/; make sure to open this link in a new tab or window for viewing, otherwise the files don't show).
We note that a limitation of the anonymous GitHub repository is that the code can't be downloaded or cloned.

## Data Loaders

All datasets are loaded via the files in the data_loaders folder. 

## Get Results

The code used to evaluate the predictions (i.e., get the metric scores) made by all models is in the get_test_results_final folder.

## Misc

The misc folder includes complementary code that is used throughout this repository.

## Training files

All files used for training (and prediction generation) are displayed in the training_files folder.

### SuperGLUE Benchmark (Single Model)

The training and prediction files used to genreate the results for Table 1 and 4 are in the superGLUE_10_epochs folder.

### Multi-Task Training Regime (other experiments not included in the submission).

The training and prediction files for the multi-task training regime are in the general_experiments folder.

### Ablation Study (other experiments not included in the submission)

The training and prediction files for the ablation study are in the ablation_aux_toks folder.

### Neuromodulation Probe (other experiments not included in the submission)

The code for the neuromodulation probe is in the probe_neuromodulation folder.

### Gating Start vs. End

The experiments regarding the comparison of the start and end positions is in the gating_start_vs_end folder.

## Dependencies

The exact environment we used to conduct experiments is shown in the dependnecies.txt file. The important modules are
a working TensorFlow 2.4.1 environment (including all relevant dependencies), pyplot to reproduce Figure 2, and a custom 
transformers module with neuromodulation support (https://anonymous.4open.science/r/XXXXX-transformers-XXXXXX/).

To install the custom transformers package you need to run:

pip install git+git_repository@branch

This currently doesn't work with the anonymized version of the repository (as you can't download or clone it).
This will be updated in the non-anonymized repository to include instructions on how to install the module.

## Dataset Install Instructions

The dataset_download_instructions.txt includes instruction on how to download all datasets used in the experiments. 

## Custom Transformers Module for Neuromodulation

The custom transformers module with support for neuromodulation can be found at https://anonymous.4open.science/r/XXXXX-transformers-XXXXXX/. 
To view the changes to the code go to src/transformers/models/bert. We have modified the modeling_tf_bert.py and 
configuration_bert.py files with support for neuromodulation. For modeling_tf_bert.py go directly to line 530 (the 
class TfBertEncoder) to see the modifications to the code. Additionally, in the configuration_bert.py file we added
support for parameters relevant to neuromodulation that will be used by the bert model in modeling_tf_bert.py. Some of the changes made are for additional experiments not included in the submitted paper.

## Instructions on How to Run Models

This will be provided in the non-anonymized version when the code can actually be downloaded.

## Seeds for each experiment

TODO. 







