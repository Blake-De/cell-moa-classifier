# Cell MOA Classifier

A PyTorch-based ResNet (CNN) model for classifying fluorescence microscopy images of drug-treated breast cancer cells by their **mechanism of action (MOA)**.

## Overview

This project builds a deep learning model to classify cell images based on how different chemical compounds affect them. It uses a ResNet-18 convolutional neural network trained on fluorescence microscopy images labeled with one of 13 biological mechanisms of action (e.g., DNA damage, actin disruption, kinase inhibition). 

### Key Features:

- ResNet-18 backbone with transfer learning
- Multi-class classification (13 MOA classes)
- Weighted sampling for class imbalance
- Data augmentation: horizontal flips and random rotation
- Exported as a TorchScript model for inference
- Fully reproducible with Conda environment YAML

## Data

- **Source**: The dataset used here is a preprocessed version of the Broad Institute MCF7 Cell Imaging Dataset
- Each training sample consists of 3 grayscale 512×512 images representing:
  - DNA
  - F-actin
  - B-tubulin fluorescent channels
- **Training set**: ~14,430 examples
- **Test set**: 1,718 examples (evaluated separately)

## Development Environment Requirements

- Python 3.10+
- Conda (recommended: Miniconda or Anaconda)
- PyTorch 2.6.0 (GPU support via CUDA 12.4 for GPU acceleration)
- RDKit installed via Conda from `conda-forge`
- All dependencies are specified in `NN_env.yaml` 

## Installation

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
conda env create -f NN_env.yaml
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

## Output

- After training, the model is saved in **TorchScript format** as `model.pth`
  - Loadable via `torch.jit.load()` for evaluation or inference
- Console logs are printed during training:
  - Loss is reported approximately every 100 steps (can be changed easily)
  - Epoch time is displayed at the end of each epoch

**Example log output:** <br>
Epoch 0 Step 100: loss = 0.712 <br>
Epoch 0 Step 200: loss = 0.634 <br>
... <br>
Epoch time: 54.31 seconds <br>

## Model Performance

The following results were obtained by running the model with this command:

```bash
python train_NN.py --train_data_dir ./data --max_epochs 5 --batch_size 32 --out model.pth
```

| Metric          | Value      |
| --------------- | ---------- |
| **F1 Micro**    | 0.957      |
| **F1 Macro**    | 0.952      |
| **Accuracy**    | 0.957      |
| **PredictTime** | 13.463 sec |

## Project Structure

```bash
cell-moa-classifier/
├── train_NN.py         # Main training script  
├── NN_env.yaml         # Conda environment file  
├── README.md           # Project documentation  
├── .gitignore          # Git ignore rules  
└── model.pth           # Trained model output (not tracked in Git)
```

## Author & Acknowledgments

Blake Degioanni  
[GitHub](https://github.com/Blake-De) • [LinkedIn](https://www.linkedin.com/in/blake-degioanni)  

## Notes

This project is considered complete. Minor updates may be made to improve code documentation (e.g., adding docstrings or clarifying comments).
