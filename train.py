#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import subprocess
import sys

CONFIG_FILE = "/root/autodl-tmp/train_config.yaml"

CONDA_ACTIVATE = "/root/miniconda3/bin/activate"
CONDA_ENV = "llamafac"


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Configuration file does not exist: {CONFIG_FILE}")
        return 1

    train_cmd = (
        f"/bin/bash -c 'source {CONDA_ACTIVATE} {CONDA_ENV} && "
        f"llamafactory-cli train {CONFIG_FILE}'"
    )

    print("Begin single-card training...")

    try:
        subprocess.run(train_cmd, shell=True, check=True, capture_output=False, text=True)
        print("Training complete!")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"Training failed, exit code: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("Training was interrupted by the user.")
        return 1
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())