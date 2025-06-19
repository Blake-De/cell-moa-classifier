# Cell MOA Classifier

A PyTorch-based ResNet (CNN) model for classifying fluorescence microscopy images of drug-treated breast cancer cells by their **mechanism of action (MOA)**.

## Overview

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

Train the Model
```bash
python train_NN.py --train_data_dir /path/to/train --batch_size 20 --max_epochs 10 --out model.pth
