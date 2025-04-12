from m3llama.utils import load_model, generate, generate_with_cache
from transformers import AutoTokenizer, AutoModelForCausalLM
from m3llama.memory import Base_Memory_3, M3_cache
from accelerate import Accelerator
import torch

if __name__ == "__main__":
    accelerator = Accelerator()
    # query = ["1+1=2, 2+2=4, 3+3="*100, "3+3=6, 4+4="*100]
    query = [
    [
        {
            "role": "user",
            "content": "4+4=",
        },
        {
            "role": "assistant",
            "content": "8",
        }
    ],
    [
        {
            "role": "user",
            "content": "1+1=2, 2+2=4, 3+3=6"*100+"4+4=",
        },
        {
            "role": "assistant",
            "content": "8",
        }
    ]
]
    model, tokenizer, retrieval_model = load_model("/root/autodl-tmp/model/m3-llama-3.2-3b-instruct")
    # print(model)
    # print(tokenizer)
    # print(model.config)

    # response = generate(query, model, tokenizer)
    # print(response)

    memory_processor = Base_Memory_3(model=model, tokenizer=tokenizer, retrieval_model=retrieval_model, config=model.config)
    memory_processor.load_from_disk("/root/autodl-tmp/memory")
    formatted_query = tokenizer.apply_chat_template(query, tokenize=False)
    print(formatted_query)
    tokens = tokenizer(formatted_query, return_tensors="pt", padding='longest', truncation=True, padding_side='left')
    query = tokens.input_ids.to(model.device)
    memories = memory_processor.preprocess(query)
    model.eval()
    output = model(query, memories=memories, memory_processor=memory_processor, attention_mask=tokens.attention_mask.to(model.device))
    token_indices = torch.argmax(output.logits, dim=-1)
    # Convert token indices to text
    print(output.logits.shape)
    decoded_text = tokenizer.decode(token_indices[:, -1], skip_special_tokens=True)
    print(decoded_text)
    
    # print(response)
    