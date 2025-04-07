from utils import load_model, generate, generate_with_cache
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import M3_LlamaConfig
from memory import Base_Memory_3, M3_cache
from accelerate import Accelerator
import torch

if __name__ == "__main__":
    accelerator = Accelerator()
    query = ["1+1=2, 2+2=4, 3+3="*100, "3+3=6, 4+4="*100]
    model, tokenizer, retrieval_model = load_model("/root/autodl-tmp/Explicit-Memory/model/m3_llama_3.2_3b")
    # print(model)
    # print(tokenizer)
    # print(model.config)

    # response = generate(query, model, tokenizer)
    # print(response)

    memory_processor = Base_Memory_3(model=model, tokenizer=tokenizer, retrieval_model=retrieval_model, config=model.config)
    memory_processor.load_from_disk("/root/autodl-tmp/Explicit-Memory/memory")
    query = tokenizer(query, return_tensors="pt", padding='longest', truncation=True, padding_side='left').input_ids.to(model.device)
    memories = memory_processor.preprocess(query)
    output = model(query, memories=memories, memory_processor=memory_processor)
    token_indices = torch.argmax(output.logits, dim=-1)
    # Convert token indices to text
    print(output.logits.shape)
    decoded_text = tokenizer.decode(token_indices[:, -1], skip_special_tokens=True)
    print(decoded_text)
    
    # print(response)
    