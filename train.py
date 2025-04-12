#!/usr/bin/env python3
import sys
from llamafactory.cli import run_exp
import os

def main():
    os.environ['ALLOW_EXTRA_ARGS'] = 'true'
    sys.argv = [
        "llamafactory-cli",
        "/root/autodl-tmp/Explicit-Memory/LLaMA-Factory/examples/train_full/llama3_M3_full_sft.yaml"
    ]
    run_exp()

if __name__ == "__main__":
    main()