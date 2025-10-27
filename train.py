#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单卡训练脚本（精简版）
参考多卡并行脚本风格，使用 LlamaFactory CLI 与最终配置文件。
移除旧版中的各类测试与冗余打印，仅保留必要的训练入口。
"""

import os
import subprocess
import sys

# 使用的训练配置文件（单卡）
CONFIG_FILE = "/root/autodl-tmp/train_config.yaml"

# 可选：如果需要通过 conda 环境运行，请保持与并行脚本一致
CONDA_ACTIVATE = "/root/miniconda3/bin/activate"
CONDA_ENV = "llamafac"


def main() -> int:
    # 设置单卡相关环境变量（保持精简）
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if not os.path.exists(CONFIG_FILE):
        print(f"错误: 配置文件不存在: {CONFIG_FILE}")
        return 1

    # 构建训练命令（通过 bash 激活 conda 环境后调用 CLI）
    train_cmd = (
        f"/bin/bash -c 'source {CONDA_ACTIVATE} {CONDA_ENV} && "
        f"llamafactory-cli train {CONFIG_FILE}'"
    )

    print("开始单卡训练...")

    try:
        subprocess.run(train_cmd, shell=True, check=True, capture_output=False, text=True)
        print("训练完成！")
        return 0
    except subprocess.CalledProcessError as e:
        print(f"训练失败，退出码: {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("训练被用户中断")
        return 1
    except Exception as e:
        print(f"训练过程中发生错误: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())