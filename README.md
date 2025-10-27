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
└── saves/                         # Model save directory
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

- Place the training data in the `main` directory. For the file format and name, refer to raw_data_filtered_8192.json.

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

- The fine-tuned results reported in the paper are demonstrated in the saves/ directory
