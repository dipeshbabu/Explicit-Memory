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

@dataclass
class MemoryChunk:
    """Data structure for storing a single memory chunk"""
    text: str  # Original text
    key_states: torch.Tensor  # attention key states (batch_size, num_heads, seq_len, head_dim)
    value_states: torch.Tensor  # attention value states (batch_size, num_heads, seq_len, head_dim)

class Base_Memory_3(DynamicCache):
    def __init__(
        self,
        model: PreTrainedModel,  # Model
        tokenizer: PreTrainedTokenizer,  # tokenizer
        retrieval_model: Retriever,  # Model for text embedding
        memory_length: int = 128,  # Length of each memory chunk
        num_memory_chunks: int = 5,  # Number of chunks to keep in memory
        memory_update_interval: int = 64,  # Update memory every N generated tokens
        device: str = "cuda"
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.retrieval_model = retrieval_model
        self.memory_length = memory_length
        self.num_memory_chunks = num_memory_chunks
        self.memory_update_interval = memory_update_interval
        self.device = device
        
        # Initialize vector database
        self.vector_db = faiss.IndexFlatIP(self.retrieval_model.config.hidden_size)
        self.memory_chunks: Dict[int, MemoryChunk] = {}
        
        # Track generated tokens separately for each batch
        self.last_tokens = {}  # batch_idx -> tokens list
        
    def process_knowledge_base(self, knowledge_base: List[str], save_path: str):
        """Process knowledge base and store as memory chunks"""
        chunk_texts = []
        chunk_embeddings = []
        chunk_keys = []
        chunk_values = []
        
        # Ensure save path exists
        os.makedirs(save_path, exist_ok=True)
        
        for text in knowledge_base:
            # Split text into chunks
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, add_special_tokens=False).to(self.device)
            
            # Split into chunks of memory_length
            for i in range(0, tokens.input_ids.size(1), self.memory_length):
                chunk_tokens = tokens.input_ids[:, i:i+self.memory_length]
                if chunk_tokens.size(1) < self.memory_length:
                    # Pad the last incomplete chunk
                    padding = torch.zeros(
                        1, 
                        self.memory_length - chunk_tokens.size(1), 
                        dtype=torch.long,
                        device=self.device
                    )
                    chunk_tokens = torch.cat([chunk_tokens, padding], dim=1)
                
                # Get text representation of chunk
                chunk_text = self.tokenizer.decode(chunk_tokens[0], skip_special_tokens=True)
                chunk_texts.append(chunk_text)
                
                # Calculate chunk embedding
                with torch.no_grad():
                    embedding = self.retrieval_model.encode(chunk_text, convert_to_tensor=True)
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                    chunk_embeddings.append(embedding.cpu().numpy())
                
                # Calculate chunk key-value states, without attention mask to allow global attention
                with torch.no_grad():
                    # @todo: attention sparsification here
                    # Create all-ones attention mask to allow all positions to attend to each other
                    attention_mask = torch.ones(
                        (1, chunk_tokens.size(1)), 
                        dtype=torch.bool, 
                        device=self.device
                    )
                    outputs = self.model(
                        chunk_tokens,
                        output_hidden_states=True,
                        use_cache=True,
                        attention_mask=attention_mask  # Don't use attention mask
                    )
                    # Get key and value states from first layer
                    key_states = outputs.past_key_values[0][0].detach()  # (batch_size, num_heads, seq_len, head_dim)
                    value_states = outputs.past_key_values[0][1].detach()
                    chunk_keys.append(key_states)
                    chunk_values.append(value_states)
        
        # Build and store vector database
        chunk_embeddings = np.vstack(chunk_embeddings)
        self.vector_db.add(chunk_embeddings)
        
        # Store memory chunks
        for i, (text, key, value) in enumerate(zip(chunk_texts, chunk_keys, chunk_values)):
            self.memory_chunks[i] = MemoryChunk(text, key, value)
        
        # Save to disk
        self._save_to_disk(save_path)
    
    def _save_to_disk(self, save_path: str):
        """Save database to disk"""
        faiss.write_index(self.vector_db, os.path.join(save_path, "vector_db.index"))
        
        # Convert tensors to CPU version before saving
        cpu_memory_chunks = {}
        for idx, chunk in self.memory_chunks.items():
            cpu_memory_chunks[idx] = MemoryChunk(
                text=chunk.text,
                key_states=chunk.key_states.cpu(),
                value_states=chunk.value_states.cpu()
            )
            
        with open(os.path.join(save_path, "memory_chunks.pkl"), "wb") as f:
            pickle.dump(cpu_memory_chunks, f)
    
    def load_from_disk(self, load_path: str):
        """Load database from disk"""
        self.vector_db = faiss.read_index(os.path.join(load_path, "vector_db.index"))
        with open(os.path.join(load_path, "memory_chunks.pkl"), "rb") as f:
            cpu_memory_chunks = pickle.load(f)
            
        # Move tensors to correct device
        self.memory_chunks = {}
        for idx, chunk in cpu_memory_chunks.items():
            self.memory_chunks[idx] = MemoryChunk(
                text=chunk.text,
                key_states=chunk.key_states.to(self.device),
                value_states=chunk.value_states.to(self.device)
            )

class MemoryKVCache(Base_Memory_3):
    def __init__(
        self,
        model: PreTrainedModel,  # Model
        tokenizer: PreTrainedTokenizer,
        retrieval_model,
        memory_token_length: int = 16,
        num_memory_chunks: int = 5,
        memory_update_interval: int = 64,
        device: str = "cuda"
    ):
        super().__init__(
            model,
            tokenizer,
            retrieval_model,
            memory_length=memory_token_length,
            num_memory_chunks=num_memory_chunks,
            memory_update_interval=memory_update_interval,
            device=device
        )
    
    def update_memory(
        self, query: str, batch_idx: int, layer_idx: int, memory_head_indices: List[int]
    ):
        """Only update KV cache for specified layer and corresponding memory heads"""
        with torch.no_grad():
            query_embedding = self.retrieval_model.encode(query, convert_to_tensor=True)
            query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
            query_embedding = query_embedding.cpu().numpy()
        distances, indices = self.vector_db.search(
            query_embedding.reshape(1, -1), self.num_memory_chunks
        )
        memory_keys = []
        memory_values = []
        for idx in indices[0]:
            chunk = self.memory_chunks[int(idx)]
            memory_keys.append(chunk.key_states)
            memory_values.append(chunk.value_states)
        memory_keys = torch.cat(memory_keys, dim=2)  # shape: (1, num_heads, mem_seq_len, head_dim)
        memory_values = torch.cat(memory_values, dim=2)
        mem_seq_len = memory_keys.size(2)  # memory_token_length * num_memory_chunks
        
        # Only update heads configured as memory heads in this layer
        if layer_idx in memory_head_indices:
            designated_heads = memory_head_indices  # list of head indices
            for head_idx in designated_heads:
                self.key_cache[layer_idx][batch_idx:batch_idx+1, head_idx:head_idx+1, :mem_seq_len] = \
                    memory_keys[:, head_idx:head_idx+1, :mem_seq_len, :]
                self.value_cache[layer_idx][batch_idx:batch_idx+1, head_idx:head_idx+1, :mem_seq_len] = \
                    memory_values[:, head_idx:head_idx+1, :mem_seq_len, :]
    
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
        if hasattr(cache_kwargs, 'input_ids'):  # If cache has been initialized
            memory_head_indices = cache_kwargs.get("memory_head_indices", None)
            input_ids = cache_kwargs.get("input_ids", None)
            batch_size = input_ids.size(0)
            for batch_idx in range(batch_size):
                # Get generation count from model
                if batch_idx not in self.last_tokens:
                    self.last_tokens[batch_idx] = []
                self.last_tokens[batch_idx].append(input_ids[batch_idx, :])
                tokens_count = len(self.last_tokens[batch_idx])
                
                # Update memory every self.memory_update_interval tokens
                if tokens_count > 0 and tokens_count % self.memory_update_interval == 0:
                    recent_tokens = self.last_tokens[batch_idx]
                    if recent_tokens:
                        recent_text = self.tokenizer.decode(recent_tokens)
                        if memory_head_indices is not None:
                            self.update_memory(recent_text, batch_idx, layer_idx, memory_head_indices)
                    self.last_tokens[batch_idx] = []
    
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
    
    # @todo: If memory is too large, need to change encoding part to only encode memory head portion