# ProtoAL: Interpretable deep active learning with prototypes for medical imaging

This repository implements the ProtoAL model presented in the [ProtoAL: Interpretable deep active learning with prototypes for medical imaging](https://arxiv.org/2404.04736) paper.

<center><img src="proposal_fig.jpg" alt="drawing" width="600"/></center>

## Table of Contents
- [Directory Structure](#directory-structure)
- [Dataset](#dataset)
- [Generating Dataset CSVs](#generating-dataset-csvs) 
- [Environment Configuration](#environment-configuration)
- [Training](#training)
- [Local and Global Analysis](#local-and-global-analysis)
- [Acknowledgment](#acknowledgment)
- [Citation](#citation)

## Directory Structure

- `/dal`: Deep Active Learning O-medal routines.
- `/dataset`: The Messidor dataset and other utility functions.
- `/model`: The ProtoPNet model code and model utilities.
- `/runners`: Train, global analysis, and local analysis routines.
- `config.ini`: Default parameter configuration (values used in the paper).
- `config_parser.py`: Configuration parser.
- `inference.py`: Code to run local or global analysis.
- `main_al.py`: Main code to run the training routines of the ProtoAL model.

## Dataset

The Messidor dataset is available on the [Messidor download page](https://www.adcis.net/en/third-party/messidor/). Before use, it is necessary to correct the errors listed in the errata section of the page.

The dataset directory should be provided as an argument for `--basedata_dir`.

## Generating Dataset CSVs

The following code excerpt from `main_al.py` is responsible for loading the dataset. When `load_split` is set to `false`, it will generate the CSV files and dataframes. Once generated, `load_split` can be set to `true`. The CSV files used in the paper were generated with `seed = 1`.

```python 
train_dataframe, val_dataframe, test_dataframe, _, _ = get_dataset(
        config, only_dataframe=True, load_split=False)
    dataframes = {'train': train_dataframe,
                  'val': val_dataframe,
                  'test': test_dataframe}
```

## Environment Configuration

Create a virtual environment using Python:

```sh
python -m venv /path/to/new/virtual/environment
```

Example:

```sh
python -m venv .venvs/protoal
```

Activate the virtual environment:

```sh
source .venvs/protoal/bin/activate
```

Install the packages from the `requirements.txt` file:

```sh
pip install -r requirements.txt
```

## Train

To execute the training routine of the ProtoAL model, use the following command:

> [!NOTE]
> The execution tracking is done using [Weights & Biases](https://wandb.ai/). If you don't have an account, comment out the W&B calls in the code.

```sh
python main_al.py --dataset messidor --basedata_dir <messidor_dataset_directory>
```

For a complete list of parameters, check the `config_parser.py` file.

At the end of the execution, a CSV log file will be saved in the `grid_csv` directory. The run iterations, along with model saves and pushed prototypes, will be saved in the `saved_models` directory.

## Local and Global Analysis

To perform local analysis with an image file, run:

```sh
python inference.py --dataset messidor --infer_mode local --load_model saved_models/<run_dir>/<iter_dir>/model_weights.pth --load_model_dir saved_models/<run_dir>/<iter_dir>/img/ --image_label <label_value> --save_dir_path results/ --image_path <test_image_file> > results/output.log
```

To run global analysis:

```sh
python inference.py --dataset messidor --infer_mode global --load_model saved_models/<run_dir>/<iter_dir>/model_weights.pth --load_model_dir saved_models/<run_dir>/<iter_dir>/img/
```

## Acknowledgment

This repository contains modified source code from [cfchen-duke/ProtoPNet](https://github.com/cfchen-duke/ProtoPNet) ([MIT License](https://github.com/cfchen-duke/ProtoPNet/blob/81bf2b70cb60e4f36e25e8be386eb616b7459321/LICENSE)) by Chaofan Chen, Oscar Li, Chaofan Tao, Alina Jade Barnett, and Cynthia Rudin.

This repository contains modified source code from [adgaudio/O-MedAL](https://github.com/adgaudio/O-MedAL) ([MIT License](https://github.com/adgaudio/O-MedAL/blob/main/LICENSE)) by Asim Smailagic, Pedro Costa, Alex Gaudio, Kartik Khandelwal, Mostafa Mirshekari, Jonathon Fagert, Devesh Walawalkar, Susu Xu, Adrian Galdran, Pei Zhang, Aurélio Campilho, and Hae Young Noh.

## Citation

```bibtex
@article{santos2024protoal,
  title={ProtoAL: Interpretable Deep Active Learning with prototypes for medical imaging},
  author={Santos, Iury B. de A. and de Carvalho, André CPLF},
  journal={arXiv preprint arXiv:2404.04736},
  year={2024}
}
```