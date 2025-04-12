import sys
sys.path.append('/root/autodl-tmp/Explicit-Memory/m3llama')
from m3_model import M3_LlamaForCausalLM
from memory import MemoryKVCache
from m3_config import M3_LlamaConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama import modeling_llama
from accelerate import Accelerator
from retriever import Retriever
import torch
import torch.nn.functional as F

def load_model(model_path: str="/root/autodl-tmp/meta-llama/Llama-3.1-8B", retrieval_model_path: str="/root/autodl-tmp/BAAI/bge-m3") -> tuple[M3_LlamaForCausalLM, AutoTokenizer, MemoryKVCache]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, device_map="auto")
    # tokenizer.add_special_tokens({'pad_token': '<|finetune_right_pad_id|>'})
    tokenizer.pad_token_id = 128001
    modeling_llama.LlamaConfig = M3_LlamaConfig
    try:
        model = M3_LlamaForCausalLM.from_pretrained(model_path, device_map="auto").half()
    except Exception as e:
        print(e)
        model = M3_LlamaForCausalLM.from_pretrained(model_path, ignore_mismatched_sizes=True, device_map="auto").half()
    retrieval_model = Retriever(retrieval_model_path)
    return model, tokenizer, retrieval_model

def pre_process(query: str, memory_token_length: int, tokenizer: AutoTokenizer):
    special_tokens = tokenizer.special_tokens_map
    bos_token = special_tokens['bos_token']
    pad_token = special_tokens['pad_token']
    query = "Reference: " + pad_token*memory_token_length + bos_token + query 
    return query

def get_response(inputs,outputs,tokenizer,num_return):
    responses_list=[]
    batch_return=[]
    for i, output in enumerate(outputs):
        input_len = len(inputs[0])
        generated_output = output[input_len:]
        batch_return.append(tokenizer.decode(generated_output, skip_special_tokens=True))
        if i%num_return==num_return-1:
            responses_list.append(batch_return)
            batch_return=[]
    return responses_list

def generate(prompt: str, model: M3_LlamaForCausalLM, tokenizer: AutoTokenizer):
    # accelerator = Accelerator()
    # gen_kwargs = {'num_return_sequences': 1, 'min_new_tokens': 10 ,'max_length':2048, 'num_beams':1,
    #         'do_sample':True, 'top_p':0.7, 'temperature':0.9, 'repetition_penalty':1.2}
    gen_kwargs = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.9,
    "_from_model_config": True,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "transformers_version": "4.43.0.dev0"
    }
    model = model.eval()
    # prompt = pre_process(prompt, memory_token_length, tokenizer)
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    input_ids = input_ids.to(model.device)
    # input_ids = input_ids.to('cuda')
    outputs = model.generate(input_ids, **gen_kwargs)
    response = get_response(input_ids,outputs,tokenizer,1)
    return response[0][0]

    # model,input_ids = accelerator.prepare(model,input_ids)
    # # input_ids = input_ids.to('cuda')
    # # outputs = accelerator.unwrap_model(model).generate(input_ids,**gen_kwargs)
    # outputs = accelerator.unwrap_model(model).generate(input_ids, previous_key_values=cache)
    # response = get_response(input_ids,outputs,tokenizer,1)
    # return response[0][0]

def generate_with_cache(prompt: list[torch.Tensor], model: M3_LlamaForCausalLM, tokenizer: AutoTokenizer, cache: MemoryKVCache, cache_position: torch.Tensor):
    # accelerator = Accelerator()
    # gen_kwargs = {'num_return_sequences': 1, 'min_new_tokens': 10 ,'max_length':2048, 'num_beams':1,
    #         'do_sample':True, 'top_p':0.7, 'temperature':0.9, 'repetition_penalty':1.2}
    gen_kwargs = {
    "do_sample": True,
    "temperature": 0.6,
    "top_p": 0.9,
    "_from_model_config": True,
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "transformers_version": "4.43.0.dev0",
    "cache_position": cache_position,
    "is_causal": False
    }
    model = model.eval()
    memory_token_length = model.config.memory_token_length
    # prompt = pre_process(prompt, memory_token_length, tokenizer)
    # input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
    input_ids = prompt.to(model.device)
    # input_ids = input_ids.to('cuda')
    outputs = model.generate(input_ids, past_key_values=cache, **gen_kwargs)
    response = get_response(input_ids,outputs,tokenizer,1)
    return response[0][0]

    # model,input_ids = accelerator.prepare(model,input_ids)
    # # input_ids = input_ids.to('cuda')
    # # outputs = accelerator.unwrap_model(model).generate(input_ids,**gen_kwargs)
    # outputs = accelerator.unwrap_model(model).generate(input_ids, previous_key_values=cache)
    # response = get_response(input_ids,outputs,tokenizer,1)
    # return response[0][0]
