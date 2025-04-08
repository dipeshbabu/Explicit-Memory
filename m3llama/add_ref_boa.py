from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
tokenizer.add_special_tokens({'additional_special_tokens': ['<|ref_bos|>']})

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B")
model.resize_token_embeddings(len(tokenizer))

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")

# 添加ref bos token
tokenizer.add_special_tokens({'additional_special_tokens': ['<|ref_bos|>']})
tokenizer.ref_bos_id = tokenizer.convert_tokens_to_ids('<|ref_bos|>')
# 重写构建输入的方法
def build_inputs_with_special_tokens(self, token_ids):
    return [self.ref_bos_id] + [self.bos_token_id] + token_ids + [self.eos_token_id]

tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens.__get__(tokenizer)


