# Synthesis-Properties-Database-for-Nanomaterials
Our goal is to create a database of nanomaterials containing detailed synthesis routes and corresponding product properties (size, morphology, and optical properties). Fine-tune the Qwen3-14b model using LoRA within the LLaMA-Factory framework to achieve the goal.

## Project Structure
```
main/
├── train.py                       # Single-GPU training script
├── train_config.yaml              # Training configuration file
├── test_model_optimized.py        # Multi-threaded inference script
├── requirements.txt               # Python dependencies
├── Qwen3-14B/                     # Pre-trained model directory (Please download from https://huggingface.co/Qwen/Qwen3-14B)
├── dataset_info.json              # Training Set Format
├── raw_data_filtered_8192.json    # Examples in the training set
├── test_labels.json               # Test set
├── results/                       # The results of the test set in the paper.
├── saves/                         # Model save directory
└── inverse_design/                # Use of this database for inverse design of nanomaterials
```

## Environment Requirements

- Python 3.8+
- CUDA 11.8+ or 12.x
- GPU memory >= 24GB (recommended)

## Installation Steps

### 1. Create a Conda Environment

```bash
conda create -n llamafac python=3.12
conda activate llamafac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install LLaMA-Factory

```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .
```

### 4. Install Flash Attention (optional, improves training speed)

```bash
pip install flash-attn --no-build-isolation
```

## Data Preparation

- Place the training data in the `main` directory. For the file format and name, refer to `raw_data_filtered_8192.json`

## Model Preparation

- Download the Qwen3-14B pretrained model to the `Qwen3-14B/` directory

## Training Configuration

Key configuration items are in `train_config.yaml`:

- **Model Configuration**: model path, Flash Attention, Liger Kernel
- **Training Method**: LoRA fine-tuning parameters
- **Data Configuration**: dataset path, template, sequence length
- **Training Parameters**: batch size, learning rate, number of training epochs

## Starting Training

### Using LLaMA-Factory CLI

```bash
llamafactory-cli train train_config.yaml
```

## Model Saving

- The fine-tuned results reported in the paper are demonstrated in the `saves/` directory
https://huggingface.co/Kai-gu/Qwen3-14B-finetune/tree/main

## Model Inference

### Prepare Test Set
- Refer to `test_labels.json` for the file format and name

### Start the Inference Script

```bash
python test_model_optimized.py
```

### Inference Results
- The best results reported in the paper are located in the `results/` directory

## Database Storage
- https://huggingface.co/datasets/Kai-gu/Synthesis-Properties-Database-for-Nanomaterials

## Citation
If you find this project helpful, please cite our article:
```
DOI：10.1021/acsnano.6c03070
```
```
@article{doi:10.1021/acsnano.6c03070,
author = {Gu, Kai and Liang, Yingping and Peng, Senliang and Guo, Aotian and Fu, Ying and Zhong, Haizheng},
title = {A Large-Scale Nanocrystal Database with Aligned Synthesis and Properties, Enabling Generative Inverse Design},
journal = {ACS Nano},
volume = {20},
number = {24},
pages = {17413-17422},
year = {2026},
doi = {10.1021/acsnano.6c03070},
    note ={PMID: 42253088},

URL = {     
        https://doi.org/10.1021/acsnano.6c03070
},
eprint = {    
        https://doi.org/10.1021/acsnano.6c03070
}
}
```
If you have any questions, please contact kai_gu94@163.com
