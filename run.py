from utils import load_model, generate
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import M3_LlamaConfig
from memory import MemoryKVCache
from accelerate import Accelerator

if __name__ == "__main__":
    accelerator = Accelerator()
    query = "1+1="
    model, tokenizer, retrieval_model = load_model()
    # print(model)
    # print(tokenizer)
    # print(model.config)
    cache = MemoryKVCache(model=model, tokenizer=tokenizer, retrieval_model=retrieval_model, config=model.config)
    model, tokenizer, retrieval_model, query, cache = accelerator.prepare(model, tokenizer, retrieval_model, query, cache)
    cache.load_from_disk("/root/Explicit-Memory")

    response = generate(query, model, tokenizer, cache)
    print(response)
    