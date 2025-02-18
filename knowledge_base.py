from utils import load_model
from memory import MemoryKVCache
from accelerate import Accelerator

def test_knowledge_base():
    accelerator = Accelerator()
    model, tokenizer, retrieval_model = load_model()
    cache = MemoryKVCache(model=model, tokenizer=tokenizer, retrieval_model=retrieval_model, config=model.config)
    # print(model)
    # print(tokenizer)
    # print(model.config)
    knowledge_base = [
        "1+1=2",
        "2+2=4 ",
        "3+3=6",
        "4+4=8",
        "5+5=10"
    ]
    model, tokenizer, retrieval_model, cache, knowledge_base = accelerator.prepare(model, tokenizer, retrieval_model, cache, knowledge_base)
    save_path = "./"

    cache.process_knowledge_base(knowledge_base, save_path)

    query = "1+1="

    _, indices = cache.retrieve_memory(query, 2)
    indices = indices[0]
    retrieved_knowledge = [cache.memory_chunks[i].text for i in indices]
    print(retrieved_knowledge)

test_knowledge_base()


    
