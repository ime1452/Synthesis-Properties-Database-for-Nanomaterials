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
└── saves/                   # Model save directory
```
