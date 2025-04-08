#!/usr/bin/env python3
import sys
from llamafactory.cli import run_exp
import os

def main():
    # 直接调用run_exp()函数
    os.environ['ALLOW_EXTRA_ARGS'] = 'true'
    sys.argv = [
        "llamafactory-cli",  # 程序名称
        "/root/autodl-tmp/Explicit-Memory/LLaMA-Factory/examples/train_full/llama3_M3_full_sft.yaml"  # 配置文件路径
    ]
    run_exp()

if __name__ == "__main__":
    main()