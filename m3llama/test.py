from .m3_model import M3_LlamaForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM
import os
import sys

pretrained_model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/meta-llama/Llama-3.2-3B-Instruct").half()
config = pretrained_model.config
model = M3_LlamaForCausalLM(config=config).half()
model.model.load_state_dict(pretrained_model.state_dict(), strict=False)

if not os.path.exists("/root/autodl-tmp/Explicit-Memory/model"):
    os.makedirs("/root/autodl-tmp/Explicit-Memory/model")
model.save_pretrained("/root/autodl-tmp/Explicit-Memory/model/m3-llama-3.2-3b-instruct")
