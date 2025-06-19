# Cell MOA Classifier

A PyTorch-based ResNet (CNN) model for classifying fluorescence microscopy images of drug-treated breast cancer cells by their **mechanism of action (MOA)**.

## Overview

This project builds a deep learning model to classify cell images based on how different chemical compounds affect them. It uses a ResNet-18 convolutional neural network trained on fluorescence microscopy images labeled with one of 13 biological mechanisms of action (e.g., DNA damage, actin disruption, kinase inhibition). 

## Development Environment Requirements

- Python 3.10+
- Conda (recommended: Miniconda or Anaconda)
- PyTorch 2.6.0 (GPU support via CUDA 12.4 for GPU acceleration)
- RDKit installed via Conda from `conda-forge`
- All dependencies are specified in `NN_env.yml` 

## Installation / Environment Setup

### 1. Clone the repository:
 
 ```bash
git clone https://github.com/Blake-De/cell-moa-classifier.git
```

Navigate into the project directory:

```bash
cd cell-moa-classifier
```

### 2. Create and activate the conda environment:

```bash
conda env create -f NN_env.yml
conda activate NN_env
```

## Usage 

### Train the Model:

```bash
python train_NN.py --train_data_dir /path/to/train --batch_size 20 --max_epochs 10 --out model.pth
```

### Command-Line Arguments

| Argument            | Type   | Description                                | Default                             |
|---------------------|--------|--------------------------------------------|-------------------------------------|
| `--max_epochs`      | int    | Maximum number of training epochs          | `1`                                 |
| `--batch_size`      | int    | Number of samples per batch                | `20`                                |
| `--train_data_dir`  | str    | Path to training data directory            | `gs://mscbio2066-data/trainimgs`    |
| `--out`             | str    | File path to save the trained model        | `model.pth`                         |


## Model Performance

The following results were obtained by running the model with this command:

```bash
python train_model.py --train_data_dir ./data --max_epochs 5 --batch_size 32 --out model_final.pth
```

| Metric          | Value      |
| --------------- | ---------- |
| **F1 Micro**    | 0.957      |
| **F1 Macro**    | 0.952      |
| **Accuracy**    | 0.957      |
| **PredictTime** | 13.463 sec |




