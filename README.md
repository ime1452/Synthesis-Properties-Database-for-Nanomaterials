# Synthesis-Properties-Database-for-Nanomaterials
Our goal is to create a database of nanomaterials containing detailed synthesis routes and corresponding product properties (size, morphology, and optical properties). Fine-tune the Qwen3-14b model using LoRA within the LLaMA-Factory framework to achieve the goal.

## Project Structure
```
main/
├── train.py              # Single-GPU training script
├── train_config.yaml     # Training configuration file
├── requirements.txt      # Python dependencies
├── Qwen3-14B/            # Pre-trained model directory (Please download from https://huggingface.co/Qwen/Qwen3-14B)
├── dataset_info.json     # Training Set Format
└── saves/                # Model save directory
```
