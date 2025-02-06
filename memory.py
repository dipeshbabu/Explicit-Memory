# memory class
from transformers import DynamicCache, PreTrainedTokenizer, PreTrainedModel
import torch
import faiss
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import os
# 这个类用来实现explicit memory数据库，它可以encode knowledge，存储memory到disk上，load memory from disk, 
# 并且可以retrieve memory返回需inference中能直接使用的MemoryCache
# 存储一个用来query的向量数据库，由plain text计算而来
# 存储一个memory的pt库，存储memory
# retrieve memory时，需要先query向量数据库，得到memory的id，然后从pt库中load memory
# 最后返回一个MemoryCache，里面存储了memory的key value pair以及所有模型生成的kv cache
# 在attention的forward中，首先它会判断自己是不是memory head
# 如果是，则使用完整的memory cache，否则去除掉memory cache中的memory部分使用
# memory head看到<s>Reference：memory<s>...
# 而其他head看到<s>...
# attention在forward的时候还会先使使用每个位置的attention weight做token sparsification

@dataclass
class MemoryChunk:
    """存储单个memory chunk的数据结构"""
    text: str  # 原始文本
    key_states: torch.Tensor  # attention key states (batch_size, num_heads, seq_len, head_dim)
    value_states: torch.Tensor  # attention value states (batch_size, num_heads, seq_len, head_dim)

class Base_Memory_3(DynamicCache):
    def __init__(
        self,
        model: PreTrainedModel,  # 模型
        tokenizer: PreTrainedTokenizer,  # tokenizer
        retrieval_model,  # 用于文本嵌入的模型
        memory_length: int = 128,  # 每个memory chunk的长度
        num_memory_chunks: int = 5,  # 内存中保存的chunk数量
        memory_update_interval: int = 64,  # 每生成多少个token更新一次memory
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
        
        # 初始化向量数据库
        self.vector_db = faiss.IndexFlatIP(self.retrieval_model.config.hidden_size)
        self.memory_chunks: Dict[int, MemoryChunk] = {}
        
        # 用于追踪生成的tokens，每个batch单独追踪
        self.generated_tokens = {}  # batch_idx -> count
        self.last_tokens = {}  # batch_idx -> tokens list
        
    def process_knowledge_base(self, knowledge_base: List[str], save_path: str):
        """处理知识库并存储为memory chunks"""
        chunk_texts = []
        chunk_embeddings = []
        chunk_keys = []
        chunk_values = []
        
        # 确保保存路径存在
        os.makedirs(save_path, exist_ok=True)
        
        for text in knowledge_base:
            # 对文本进行分块
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, add_special_tokens=False).to(self.device)
            
            # 按memory_length进行分块
            for i in range(0, tokens.input_ids.size(1), self.memory_length):
                chunk_tokens = tokens.input_ids[:, i:i+self.memory_length]
                if chunk_tokens.size(1) < self.memory_length:
                    # 对最后一个不完整的chunk进行padding
                    padding = torch.zeros(
                        1, 
                        self.memory_length - chunk_tokens.size(1), 
                        dtype=torch.long,
                        device=self.device
                    )
                    chunk_tokens = torch.cat([chunk_tokens, padding], dim=1)
                
                # 获取chunk的文本表示
                chunk_text = self.tokenizer.decode(chunk_tokens[0], skip_special_tokens=True)
                chunk_texts.append(chunk_text)
                
                # 计算chunk的embedding
                with torch.no_grad():
                    embedding = self.retrieval_model.encode(chunk_text, convert_to_tensor=True)
                    embedding = torch.nn.functional.normalize(embedding, p=2, dim=0)
                    chunk_embeddings.append(embedding.cpu().numpy())
                
                # 计算chunk的key-value states，不使用attention mask以允许全局注意力
                with torch.no_grad():
                    # @todo: attention sparsification here
                    # 创建一个全1的attention mask，允许所有位置互相访问
                    attention_mask = torch.ones(
                        (1, chunk_tokens.size(1)), 
                        dtype=torch.bool, 
                        device=self.device
                    )
                    outputs = self.model(
                        chunk_tokens,
                        output_hidden_states=True,
                        use_cache=True,
                        attention_mask=attention_mask  # 不使用attention mask
                    )
                    # 获取第一层的key和value states
                    key_states = outputs.past_key_values[0][0].detach()  # (batch_size, num_heads, seq_len, head_dim)
                    value_states = outputs.past_key_values[0][1].detach()
                    chunk_keys.append(key_states)
                    chunk_values.append(value_states)
        
        # 构建并存储向量数据库
        chunk_embeddings = np.vstack(chunk_embeddings)
        self.vector_db.add(chunk_embeddings)
        
        # 存储memory chunks
        for i, (text, key, value) in enumerate(zip(chunk_texts, chunk_keys, chunk_values)):
            self.memory_chunks[i] = MemoryChunk(text, key, value)
        
        # 保存到磁盘
        self._save_to_disk(save_path)
    
    def _save_to_disk(self, save_path: str):
        """将数据库保存到磁盘"""
        faiss.write_index(self.vector_db, os.path.join(save_path, "vector_db.index"))
        
        # 将tensors转换为CPU版本再保存
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
        """从磁盘加载数据库"""
        self.vector_db = faiss.read_index(os.path.join(load_path, "vector_db.index"))
        with open(os.path.join(load_path, "memory_chunks.pkl"), "rb") as f:
            cpu_memory_chunks = pickle.load(f)
            
        # 将tensors移动到正确的设备上
        self.memory_chunks = {}
        for idx, chunk in cpu_memory_chunks.items():
            self.memory_chunks[idx] = MemoryChunk(
                text=chunk.text,
                key_states=chunk.key_states.to(self.device),
                value_states=chunk.value_states.to(self.device)
            )
    
    def update_memory(self, query: str, batch_idx: int):
        raise NotImplementedError
    
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs: Optional[Dict[str, torch.Tensor]] = None):
        raise NotImplementedError


class ExplicitMemory(Base_Memory_3):

    def update_memory(self, query: str, batch_idx: int):
        """根据query更新memory部分的key-value pairs"""
        # 获取query的embedding
        with torch.no_grad():
            query_embedding = self.retrieval_model.encode(query, convert_to_tensor=True)
            query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=0)
            query_embedding = query_embedding.cpu().numpy()
        
        # 检索最相关的chunks
        distances, indices = self.vector_db.search(
            query_embedding.reshape(1, -1), 
            self.num_memory_chunks
        )
        
        # 更新cache中的memory部分
        memory_keys = []
        memory_values = []
        for idx in indices[0]:
            chunk = self.memory_chunks[int(idx)]
            memory_keys.append(chunk.key_states)
            memory_values.append(chunk.value_states)
        
        # 合并检索到的key-value pairs
        memory_keys = torch.cat(memory_keys, dim=2)  # 在序列长度维度上拼接
        memory_values = torch.cat(memory_values, dim=2)
        
        # 更新cache中的memory部分
        if hasattr(self, 'key_states') and len(self.key_states) > 0:
            memory_length = self.memory_length * self.num_memory_chunks
            for layer_idx in range(len(self.key_states)):
                # 只更新对应batch的memory
                self.key_states[layer_idx][batch_idx:batch_idx+1, :, :memory_length] = memory_keys
                self.value_states[layer_idx][batch_idx:batch_idx+1, :, :memory_length] = memory_values
    
    def update(
        self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        layer_idx: int, 
        cache_kwargs: Optional[Dict[str, torch.Tensor]] = None
    ):
        # 调用父类基本的 update 方法
        key_states, value_states = super().update(key_states, value_states, layer_idx, cache_kwargs)
        
        # 检查是否在生成模式
        if hasattr(self, 'key_states'):  # 如果已经初始化了cache
            batch_size = key_states.size(0)
            memory_head_indices = cache_kwargs.get("memory_head_indices", None)
            
            for batch_idx in range(batch_size):
                # 从模型获取生成计数
                tokens_count = self.model.generated_tokens.get(batch_idx, 0)
                
                # 每生成self.memory_update_interval个token更新一次memory
                if tokens_count > 0 and tokens_count % self.memory_update_interval == 0:
                    recent_tokens = self.model.last_tokens.get(batch_idx, [])
                    if recent_tokens:
                        recent_text = self.tokenizer.decode(recent_tokens)
                        if memory_head_indices is not None:
                            self.update_memory(recent_text, batch_idx)
    
        return key_states, value_states

class MemoryKVCache(ExplicitMemory):
    def __init__(
        self,
        model: PreTrainedModel,  # 模型
        tokenizer: PreTrainedTokenizer,
        retrieval_model,
        memory_token_length: int = 16,
        num_memory_chunks: int = 5,
        memory_update_interval: int = 64,
        memory_layers: Optional[dict] = None,  # dict: {layer_idx: [head_idx, ...]}
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
        """只更新指定层和对应memory head的KV缓存"""
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
        
        # 只更新本层中配置为 memory head 的那几路
        if layer_idx in memory_head_indices:
            designated_heads = memory_head_indices  # list of head indices
            for head_idx in designated_heads:
                self.key_states[layer_idx][batch_idx:batch_idx+1, head_idx:head_idx+1, :mem_seq_len] = \
                    memory_keys[:, head_idx:head_idx+1, :mem_seq_len, :]
                self.value_states[layer_idx][batch_idx:batch_idx+1, head_idx:head_idx+1, :mem_seq_len] = \
                    memory_values[:, head_idx:head_idx+1, :mem_seq_len, :]
    
    def update(
        self, 
        key_states: torch.Tensor, 
        value_states: torch.Tensor, 
        layer_idx: int, 
        cache_kwargs: Optional[Dict[str, torch.Tensor]] = None
    ):
        # 调用父类基本的 update 方法
        key_states, value_states = super().update(key_states, value_states, layer_idx, cache_kwargs)
        
        # 检查是否在生成模式
        if hasattr(self, 'key_states'):  # 如果已经初始化了cache
            batch_size = key_states.size(0)
            memory_head_indices = cache_kwargs.get("memory_head_indices", None)
            
            for batch_idx in range(batch_size):
                # 从模型获取生成计数
                tokens_count = self.model.generated_tokens.get(batch_idx, 0)
                
                # 每生成self.memory_update_interval个token更新一次memory
                if tokens_count > 0 and tokens_count % self.memory_update_interval == 0:
                    recent_tokens = self.model.last_tokens.get(batch_idx, [])
                    if recent_tokens:
                        recent_text = self.tokenizer.decode(recent_tokens)
                        if memory_head_indices is not None:
                            self.update_memory(recent_text, batch_idx, layer_idx, memory_head_indices)
    
        return key_states, value_states
    
    # @todo: 如果memory太大要改变encoding部分，只encode memory head的部分的memory