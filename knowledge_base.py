from m3llama.utils import load_model
from m3llama.memory import Base_Memory_3, M3_cache
import json
from tqdm import tqdm

def test_knowledge_base():
    model, tokenizer, retrieval_model = load_model(model_path="/root/autodl-tmp/meta-llama/Llama-3.2-3B")
    cache = M3_cache()
    memory_processor = Base_Memory_3(model=model, tokenizer=tokenizer, retrieval_model=retrieval_model, config=model.config)
    # print(model)
    # print(tokenizer)
    # print(model.config)
    knowledge_base = [
        "1+1=2",
        "2+2=4",
        "3+3=6",
        "4+4=8",
        "5+5=10"
    ]
    save_path = "/root/autodl-tmp/memory"

    memory_processor.process_knowledge_base(knowledge_base, save_path)

    query = "1+1="

    _, indices = memory_processor.retrieve_memory(query, 2)
    indices = indices[0]
    memory_processor._load_memory_chunk_from_disk(save_path, indices)
    print([memory_processor.memory_chunks[i].text for i in range(len(memory_processor.memory_chunks))])
    print(memory_processor.memory_chunks[0].key_states[0].shape)

def run_knowledge_base():
    model, tokenizer, retrieval_model = load_model(model_path="/root/autodl-tmp/Explicit-Memory/model/m3-llama-3.2-1b-instruct", retrieval_model_path="/root/autodl-tmp/Explicit-Memory/model/BAAI/bge-m3")
    memory_processor = Base_Memory_3(model=model, tokenizer=tokenizer, retrieval_model=retrieval_model, config=model.config)
    # print(model)
    # print(tokenizer)
    # print(model.config)
    
    with open('/root/autodl-tmp/Explicit-Memory/LLaMA-Factory/data/sftdata.json', 'r') as f:
        data = json.load(f)
    knowledge_base = []
    for i in tqdm(range(len(data))):
        text = ""
        for conversation in data[i]['conversations']:
            text += conversation['value']
        knowledge_base.append(text)

    save_path = "/root/autodl-tmp/Explicit-Memory/memory"

    memory_processor.process_knowledge_base(knowledge_base, save_path)

run_knowledge_base()


    
