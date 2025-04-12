import sys
sys.path.append('/root/autodl-tmp/Explicit-Memory/m3llama')
from m3_model import M3_LlamaForCausalLM
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import os

pretrained_model = AutoModelForCausalLM.from_pretrained("/root/autodl-tmp/meta-llama/Llama-3.2-1B-Instruct").half()
config = pretrained_model.config
model = M3_LlamaForCausalLM(config=config).half()
model.model.load_state_dict(pretrained_model.state_dict(), strict=False)
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/meta-llama/Llama-3.2-1B-Instruct")
if not os.path.exists("/root/autodl-tmp/Explicit-Memory/model"):
    os.makedirs("/root/autodl-tmp/Explicit-Memory/model")
model.save_pretrained("/root/autodl-tmp/Explicit-Memory/model/m3-llama-3.2-1b-instruct")
tokenizer.save_pretrained("/root/autodl-tmp/Explicit-Memory/model/m3-llama-3.2-1b-instruct")