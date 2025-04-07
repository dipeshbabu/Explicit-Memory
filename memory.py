# memory class
from transformers import DynamicCache, PreTrainedTokenizer, PreTrainedModel
import torch
import faiss
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import os
from retriever import Retriever
from config import M3_LlamaConfig
from collections import OrderedDict
from copy import deepcopy
# This class implements an explicit memory database that can encode knowledge, store memory to disk,
# load memory from disk, and retrieve memory to return a MemoryCache that can be directly used in inference
# Stores a vector database for querying, computed from plain text
# Stores a memory pt database for storing memory
# When retrieving memory, first query the vector database to get memory ids, then load memory from pt database
# Finally returns a MemoryCache containing memory key-value pairs and all model-generated kv cache
# In attention's forward pass, it first checks if it is a memory head
# If yes, use the complete memory cache, otherwise remove the memory portion from memory cache
# Memory heads see <s>Reference: memory<s>...
# While other heads see <s>...
# Attention will also do token sparsification using attention weights at each position during forward pass
def expand_kv_cache(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=0, repeats=n_rep). The hidden states go from (batch,
    layer_num, num_key_value_heads, seqlen, head_dim) to (batch*n_rep, layer_num, num_key_value_heads, seqlen, head_dim)
    """
    batch, layer_num, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, None, :, :, :, :].expand(batch, n_rep, layer_num, num_key_value_heads, slen, head_dim)
    return hidden_states.reshape(batch*n_rep, layer_num, num_key_value_heads, slen, head_dim)

class LRUCache:
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1
        # 把访问的 key 移到队尾（表示最近使用）
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            # 更新已有键值
            self.cache[key] = value
            self.cache.move_to_end(key)
        else:
            # 添加新键值
            self.cache[key] = value
            # 如果超出容量，删除最久未使用的元素（第一个元素）
            if len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

@dataclass
class MemoryChunk:
    """Data structure for storing a single memory chunk"""
    text: str  # Original text
    key_states: torch.Tensor  # attention key states (batch_size, num_heads, seq_len, head_dim)
    value_states: torch.Tensor  # attention value states (batch_size, num_heads, seq_len, head_dim)
    full_text: str # full text
    token_ids: torch.Tensor # token ids

@dataclass
class Memory:
    """Data structure for storing memories feed into the model"""
    key_states: torch.Tensor  # attention key states (batch_size, num_heads, seq_len, head_dim)
    value_states: torch.Tensor  # attention value states (batch_size, num_heads, seq_len, head_dim)

class Base_Memory_3():
    def __init__(
        self,
        model: PreTrainedModel,  # Model
        tokenizer: PreTrainedTokenizer,  # tokenizer
        retrieval_model: Retriever,  # Model for text embedding
        config: M3_LlamaConfig,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.retrieval_model = retrieval_model
        self.memory_length = config.memory_token_length
        self.num_memory_chunks = config.num_memory_chunks
        self.memory_update_interval = config.memory_update_interval
        self.memory_layers = config.memory_layers
        
        # Initialize vector database
        self.vector_db = faiss.IndexFlatIP(self.retrieval_model.config.hidden_size)
        self.memory_chunks: List[MemoryChunk] = []
        self.memory_cache: Dict[int, MemoryChunk] = {}
        self.memory_cache_size = 3*self.num_memory_chunks
        self.lru_cache = LRUCache(self.memory_cache_size)
        # Track generated tokens separately for each batch
        self.last_tokens = {}  # batch_idx -> tokens list
        self.load_path = "/root/autodl-tmp/Explicit-Memory/memory"
        if not os.path.exists(self.load_path):
            os.makedirs(self.load_path, exist_ok=True)

    # Chunck the knowledgebase and store them to the disk without model involving
    def process_knowledge_base(self, knowledge_base: List[str], save_path: str):
        """Process knowledge base and store as memory chunks"""
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        chunk_texts = []
        chunk_embeddings = []
        
        idx = 0
        for text in knowledge_base:
            # Split text into chunks and left padding
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, add_special_tokens=False)
            remainder = tokens.input_ids.size(1) % self.memory_length
            if remainder > 0:
                # Pad the last incomplete chunk
                padding = torch.full(
                    (1, self.memory_length - remainder), 
                    self.tokenizer.pad_token_id,
                    dtype=tokens.input_ids.dtype
                )
                tokens.input_ids = torch.cat([padding, tokens.input_ids], dim=1)
            # with torch.no_grad():
            #     outputs = self.model(
            #         tokens.input_ids.to(self.model.device),
            #         output_hidden_states=True,
            #         use_cache=True,
            #         is_causal=False
            #     )
            #     # Get key and value states from first layer
            #     past_key_values = outputs.past_key_values
            #     key_states = [past_key_values.key_cache[i][:,:,self.prefix_len:,:].detach() for i in range(self.model.config.num_hidden_layers)]  # (layer_num, batch_size, num_kv_heads, seq_len, head_dim)
            #     value_states = [past_key_values.value_cache[i][:,:,self.prefix_len:,:].detach() for i in range(self.model.config.num_hidden_layers)]
            #     # self.prefix_key_states = [past_key_values.key_cache[i][:,:,:self.prefix_len,:].detach() for i in range(self.model.config.num_hidden_layers)]
            #     # self.prefix_value_states = [past_key_values.value_cache[i][:,:,:self.prefix_len,:].detach() for i in range(self.model.config.num_hidden_layers)]
            #     # chunk_keys.append(key_states)
            #     # chunk_values.append(value_states)
            for i in range(0, tokens.input_ids.size(1), self.memory_length):
                chunk_tokens = tokens.input_ids[:, i:i+self.memory_length]
                chunk_text = self.tokenizer.decode(chunk_tokens[0], skip_special_tokens=True)
                chunk_texts.append(chunk_text)
                # chunk_key_states = [key_states[j][:, :, i:i+self.memory_length, :] for j in range(self.model.config.num_hidden_layers)]
                # chunk_value_states = [value_states[j][:, :, i:i+self.memory_length, :] for j in range(self.model.config.num_hidden_layers)]
                with open(os.path.join(save_path, f"memory_chunks_{idx}.pkl"), "wb") as f:
                    pickle.dump(MemoryChunk(chunk_text, None, None, text, chunk_tokens), f)
                idx += 1
                with torch.no_grad():
                    embedding = self.retrieval_model.encode(chunk_text)
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)[0]
                    chunk_embeddings.append(embedding.cpu().numpy())

        # Build and store vector database
        chunk_embeddings = np.vstack(chunk_embeddings)
        self.vector_db.add(chunk_embeddings)
        # Save to disk
        self._save_to_disk(save_path)
    
    # first check if the memory is in the cache
    # if not check if the memory is in the disk
    # if yes, load the memory from the cache or disk
    def retrieve_memory(self, query: str, top_k: int):
        """Retrieve memory from disk"""
        with torch.no_grad():
            query_embedding = self.retrieval_model.encode(query)
            query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)[0]
            query_embedding = query_embedding.cpu().numpy()
            distances, indices = self.vector_db.search(query_embedding.reshape(1, -1), top_k)
        return distances, indices
    
    def preprocess(self, query: torch.Tensor, update_disk=False, use_cache=False):
        start = True
        for i in range(0, query.size(1), self.memory_update_interval):
            if i+self.memory_update_interval <= query.size(1):
                memory = self.update(query[:, i:i+self.memory_update_interval], update_disk, use_cache)
            else:
                memory = self.update(query[:, i:], update_disk, use_cache)
            if start:
                memory_keys = memory.key_states
                memory_values = memory.value_states
                start = False
            else:
                memory_keys = [torch.cat([memory_keys[i], memory.key_states[i]], dim=2) for i in range(self.model.config.num_hidden_layers)]
                memory_values = [torch.cat([memory_values[i], memory.value_states[i]], dim=2) for i in range(self.model.config.num_hidden_layers)]
        return Memory(memory_keys, memory_values)

    def update(self, query: torch.Tensor, update_disk: bool=True, use_cache: bool=True):
        bsz, _ = query.shape
        start = True
        for i in range(bsz):
            query_text = self.tokenizer.decode(query[i], skip_special_tokens=True)
            distances, indices = self.retrieve_memory(query_text, self.num_memory_chunks)
            self._load_memory_chunk_from_disk(self.load_path, indices[0], update_disk, use_cache)
            if start:
                memory_keys = [torch.cat([self.memory_chunks[j].key_states[i] for j in range(len(self.memory_chunks))], dim=2) for i in range(self.model.config.num_hidden_layers)]  # shape: (1, num_heads, mem_seq_len, head_dim)
                memory_values = [torch.cat([self.memory_chunks[j].value_states[i] for j in range(len(self.memory_chunks))], dim=2) for i in range(self.model.config.num_hidden_layers)]
                start = False
            else:
                keys = [torch.cat([self.memory_chunks[j].key_states[i] for j in range(len(self.memory_chunks))], dim=2) for i in range(self.model.config.num_hidden_layers)]  # shape: (1, num_heads, mem_seq_len, head_dim)
                values = [torch.cat([self.memory_chunks[j].value_states[i] for j in range(len(self.memory_chunks))], dim=2) for i in range(self.model.config.num_hidden_layers)]
                memory_keys = [torch.cat([memory_keys[i], keys[i]], dim=0) for i in range(self.model.config.num_hidden_layers)]
                memory_values = [torch.cat([memory_values[i], values[i]], dim=0) for i in range(self.model.config.num_hidden_layers)]
        return Memory(memory_keys, memory_values)

    def _save_to_disk(self, save_path: str):
        """Save database to disk"""
        faiss.write_index(self.vector_db, os.path.join(save_path, "vector_db.index"))
        # self._save_prefix_to_disk(save_path)

    # def _load_prefix_from_disk(self, load_path: str):
    #     with open(os.path.join(load_path, "prefix_key_states.pkl"), "rb") as f:
    #         self.prefix_key_states = pickle.load(f)
    #     with open(os.path.join(load_path, "prefix_value_states.pkl"), "rb") as f:
    #         self.prefix_value_states = pickle.load(f)
    
    def encode_memory(self, memory_chunk: MemoryChunk, use_full_text: bool=False, token_sparsification: bool=False):
        if use_full_text:
            pass
        else:
            if token_sparsification:
                pass
            else:
                tokens = memory_chunk.token_ids.to(self.model.device)
                outputs = self.model(tokens, use_cache=True, is_encoding_memory=True)
        past_key_values = outputs.past_key_values
        key_states = [past_key_values.key_cache[i][:, :, 1:, :].detach() for i in range(self.model.config.num_hidden_layers)]  # (layer_num, batch_size, num_kv_heads, seq_len, head_dim)
        value_states = [past_key_values.value_cache[i][:, :, 1:, :].detach() for i in range(self.model.config.num_hidden_layers)]
        return key_states, value_states

    def _load_memory_chunk_from_disk(self, load_path: str, indices: List[int], update_disk: bool=False, use_cache: bool=True):
        self.memory_chunks = []
        for idx in indices:
            if use_cache:
                chunk = self.lru_cache.get(idx)
            else:
                chunk = -1
            if chunk == -1:
                with open(os.path.join(load_path, f"memory_chunks_{idx}.pkl"), "rb") as f:
                    chunk = pickle.load(f)
                if chunk.key_states is None and chunk.value_states is None:
                    chunk.key_states, chunk.value_states = self.encode_memory(chunk)
                    if update_disk:
                        with open(os.path.join(load_path, f"memory_chunks_{idx}.pkl"), "wb") as f:
                            pickle.dump(chunk, f)
                if use_cache:
                    self.lru_cache.put(idx, chunk)
            self.memory_chunks.append(chunk)

    def load_from_disk(self, load_path: str):
        self.vector_db = faiss.read_index(os.path.join(load_path, "vector_db.index"))
        # self._load_prefix_from_disk(load_path)
        # prefix_len = self.prefix_key_states[0].size(2)
        # self.prefix_len = prefix_len
        # pad_token = self.tokenizer.pad_token_id
        # pad_ids = torch.tensor([[pad_token]], device=self.model.device)
        # with torch.no_grad():
        #     outputs = self.model(pad_ids, use_cache=True)
        # self.pad_key_states = outputs.past_key_values.key_cache
        # self.pad_value_states = outputs.past_key_values.value_cache
        

class MemoryKVCache(Base_Memory_3):
    def __init__(
        self,
        model: PreTrainedModel,  # Model
        tokenizer: PreTrainedTokenizer,
        retrieval_model,
        config: M3_LlamaConfig,
    ):
        super().__init__(
            model,
            tokenizer,
            retrieval_model,
            config
        )
        # self.memory_cache = {} # Cache to store frequently accessed memory chunks
        # special_tokens = self.tokenizer.special_tokens_map
        # bos_token = special_tokens['bos_token']
        # pad_token = special_tokens['pad_token']
    

    def init_memory_cache(self, prompt: List[str]|str):
        """Initialize memory cache for a new batch"""
        prompt_tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, add_special_tokens=True).to(self.model.device)
        _, head_num, seq_len, head_dim = self.prefix_key_states[0].shape
        if isinstance(prompt, str):
            # memory_key_padding = [torch.zeros((1, head_num, self.memory_length*self.num_memory_chunks, head_dim), device=self.model.device, dtype=self.model.dtype) for _ in range(self.model.config.num_hidden_layers)]
            # memory_value_padding = [torch.zeros((1, head_num, self.memory_length*self.num_memory_chunks, head_dim), device=self.model.device, dtype=self.model.dtype) for _ in range(self.model.config.num_hidden_layers)]
            memory_key_padding = [torch.cat([self.pad_key_states[i]]*self.memory_length*self.num_memory_chunks, dim=-2) for i in range(self.model.config.num_hidden_layers)]
            memory_value_padding = [torch.cat([self.pad_value_states[i]]*self.memory_length*self.num_memory_chunks, dim=-2) for i in range(self.model.config.num_hidden_layers)]
            self.key_cache = [torch.cat([self.prefix_key_states[i], memory_key_padding[i]], dim=-2) for i in range(self.model.config.num_hidden_layers)]
            self.value_cache = [torch.cat([self.prefix_value_states[i], memory_value_padding[i]], dim=-2) for i in range(self.model.config.num_hidden_layers)]
            if prompt_tokens.input_ids.size(1) <= self.memory_update_interval:
                self.update_memory(prompt, 0)
            else:
                init_prompt = self.tokenizer.decode(prompt_tokens.input_ids[0, :self.memory_length], skip_special_tokens=True)
                self.update_memory(init_prompt, 0)
        else:
            bsz = len(prompt)
            memory_key_padding = [torch.zeros((1, head_num, self.memory_length*self.num_memory_chunks, head_dim), device=self.model.device, dtype=self.model.dtype) for _ in range(self.model.config.num_hidden_layers)]
            memory_value_padding = [torch.zeros((1, head_num, self.memory_length*self.num_memory_chunks, head_dim), device=self.model.device, dtype=self.model.dtype) for _ in range(self.model.config.num_hidden_layers)]
            self.key_cache = [torch.cat([self.prefix_key_states[i], memory_key_padding[i]], dim=-2) for i in range(self.model.config.num_hidden_layers)]
            self.value_cache = [torch.cat([self.prefix_value_states[i], memory_value_padding[i]], dim=-2) for i in range(self.model.config.num_hidden_layers)]
            self.key_cache = [expand_kv_cache(self.key_cache[i], bsz) for i in range(self.model.config.num_hidden_layers)]
            self.value_cache = [expand_kv_cache(self.value_cache[i], bsz) for i in range(self.model.config.num_hidden_layers)]
            # for batch_idx, p in enumerate(prompt):
            #     if p.size(1) <= self.memory_length:
            #         self.update_memory(p, batch_idx)
            #     else:
            #         init_prompt = self.tokenizer.decode(p[0, :self.memory_length], skip_special_tokens=True)
            #         self.update_memory(init_prompt, batch_idx)
        #@todo: 逻辑有问题，需要修改以及验证，晚饭后继续
        
        for i in range(0, prompt_tokens.input_ids.size(1)-1, 1):
            outputs = self.model(
                prompt_tokens.input_ids[:, i:i+1],
                use_cache=True,
                is_causal=False,
                past_key_values=self
            )
            # self.key_cache = outputs.past_key_values.key_cache
            # self.value_cache = outputs.past_key_values.value_cache
        return prompt_tokens.input_ids[:, prompt_tokens.input_ids.size(1)-1:]

    def update_memory(
        self, query: str, batch_idx: int
    ):
        """Only update KV cache for specified layer and corresponding memory heads"""

        distances, indices = self.retrieve_memory(query, self.num_memory_chunks)

        
        # # token sparsification here
        # # Token sparsification: Only attend to top-k tokens
        # top_k = 8  # Number of tokens to attend to
        # _, top_k_indices = torch.topk(distances, k=top_k, dim=-1)
        
        # # Mask out other tokens
        # sparse_indices = torch.zeros_like(indices)
        # sparse_indices.scatter_(-1, top_k_indices, 1.0)
        
        # # Select only top-k indices
        # selected_indices = indices[sparse_indices == 1]

        # # Select specific parts of the memory chunks
        # selected_memory_keys = []
        # selected_memory_values = []
        # for idx in selected_indices:
        #     chunk = self.memory_chunks[int(idx)]

        #     # Selected only specific parts of the memory chunk
        #     selected_keys = chunk.key_states[:, :, :8, :] # First 8 tokens
        #     selected_values = chunk.value_states[:, :, :8, :]
        #     selected_memory_keys.append(selected_keys)
        #     selected_memory_values.append(selected_values)

        # # Update memory cache with selected parts
        # selected_memory_keys = torch.cat(selected_memory_keys, dim=2)
        # selected_memory_values = torch.cat(selected_memory_values, dim=2)
        # # token sparsification end
        
        # memory_keys = []
        # memory_values = []
        self._load_memory_chunk_from_disk(self.load_path, indices[0])
        # for chunk in self.memory_chunks:
        #     if chunk is not None:
        #         memory_keys.append(chunk.key_states)
        #         memory_values.append(chunk.value_states)
        # for idx in indices[0]:
        #     # @todo: why tuple?
        #     # @todo: when to update memory cache?
        #     cache_key = tuple[idx]
        #     if cache_key in self.memory_cache:
        #         memory_keys, memory_values = self.memory_cache[cache_key]
        #     else:
        #         chunk = self.memory_chunks[int(idx)]
        #         memory_keys.append(chunk.key_states)
        #         memory_values.append(chunk.value_states)
        memory_keys = [torch.cat([self.memory_chunks[j].key_states[i] for j in range(len(self.memory_chunks))], dim=2) for i in range(self.model.config.num_hidden_layers)]  # shape: (1, num_heads, mem_seq_len, head_dim)
        memory_values = [torch.cat([self.memory_chunks[j].value_states[i] for j in range(len(self.memory_chunks))], dim=2) for i in range(self.model.config.num_hidden_layers)]
        mem_seq_len = self.memory_length * self.num_memory_chunks  # memory_token_length * num_memory_chunks
        prefix_len = self.prefix_len
        # Only update heads configured as memory heads in this layer
        for layer_idx in self.memory_layers.keys():
            designated_heads = self.memory_layers[layer_idx]  # list of head indices
            layer_idx = int(layer_idx)
            for head_idx in designated_heads:
                self.key_cache[layer_idx][batch_idx:batch_idx+1, head_idx:head_idx+1, prefix_len:prefix_len+mem_seq_len, :] = \
                    memory_keys[layer_idx][:, head_idx:head_idx+1, :mem_seq_len, :]
                self.value_cache[layer_idx][batch_idx:batch_idx+1, head_idx:head_idx+1, prefix_len:prefix_len+mem_seq_len, :] = \
                    memory_values[layer_idx][:, head_idx:head_idx+1, :mem_seq_len, :]
    
    def update(
        self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        layer_idx: int, 
        cache_kwargs: Optional[Dict[str, torch.Tensor]] = None
    ):
        # Call parent class's basic update method
        key_states, value_states = super().update(key_states, value_states, layer_idx, cache_kwargs)
        
        # Check if in generation mode
        if 'input_ids' in cache_kwargs and layer_idx == 0:  # If cache has been initialized
            memory_head_indices = cache_kwargs.get("memory_head_indices", None)
            input_ids = cache_kwargs.get("input_ids", None)
            batch_size = input_ids.size(0)
            
            for batch_idx in range(batch_size):
                # Get generation count from model
                if batch_idx not in self.last_tokens:
                    self.last_tokens[batch_idx] = []
                self.last_tokens[batch_idx].append(input_ids[batch_idx, :])
                tokens_count = sum([self.last_tokens[batch_idx][i].size(0) for i in range(len(self.last_tokens[batch_idx]))])
                
                # Update memory every self.memory_update_interval tokens
                if tokens_count > 0 and tokens_count % self.memory_update_interval == 0:
                    recent_tokens = self.last_tokens[batch_idx]
                    if recent_tokens:
                        recent_text = self.tokenizer.batch_decode(recent_tokens)
                        self.update_memory(recent_text, batch_idx)
                    self.last_tokens[batch_idx] = []

        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    # @todo: If memory is too large, need to change encoding part to only encode memory head portion

class M3_cache(DynamicCache):
    def __init__(self):
        super().__init__()
        self.generated_tokens = None
            
