import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import (
    get_linear_schedule_with_warmup,
    default_data_collator,
    AutoTokenizer
)
from config import M3_LlamaConfig
from m3_model import M3_LlamaForCausalLM
from memory import MemoryKVCache
from retriever import Retriever
from tqdm import tqdm
from datasets import load_dataset

class MemoryEnhancedTrainer:
    def __init__(self, config: M3_LlamaConfig, train_args: dict):
        self.config = config
        self.train_args = train_args
        self.accelerator = Accelerator(
            mixed_precision=self.train_args.get("mixed_precision", "fp16"),
            gradient_accumulation_steps=self.train_args.get("gradient_accumulation_steps", 1),
        )
        
        # Initialize components
        self.model = M3_LlamaForCausalLM(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.retriever = Retriever()
        self.memory_cache = MemoryKVCache(
            self.model,
            self.tokenizer,
            self.retriever,
            config,
            device=self.accelerator.device
        )

        # Initialize optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.train_args.get("learning_rate", 2e-5),
            weight_decay=self.train_args.get("weight_decay", 0.01)
        )
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=self.optimizer,
            num_warmup_steps=self.train_args.get("warmup_steps", 100),
            num_training_steps=self.train_args["total_steps"],
        )

        # Prepare components with Accelerator
        (
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.memory_cache
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.memory_cache
        )

    def _dataset_formatter(self, example):
        """Format dataset-specific examples into text sequences"""
        if self.train_args["dataset_format"] == "MetaMathQA":
            return {"text": f"Question: {example['query']}\nAnswer: {example['response']}"}
        elif self.train_args["dataset_format"] == "Capybara":
            conversation = "\n".join(
                [f"{turn['input']}\n{turn['output']}" 
                 for turn in example["conversation"]]
            )
            return {"text": conversation}
        elif self.train_args["dataset_format"] == "PythonCode":
            return {"text": f"Instruction: {example['instruction']}\nInput: {example['input']}\nOutput: {example['output']}"}
        else:
            return {"text": example["text"]}

    def _prepare_dataset(self):
        """Load and preprocess dataset with proper formatting"""
        dataset = load_dataset(
            self.train_args["dataset_path"], 
            split=f"train[:{self.train_args.get('max_samples', '100%')}]"
        )
        
        # Format dataset entries
        formatted_dataset = dataset.map(
            self._dataset_formatter,
            remove_columns=dataset.column_names
        )
        
        # Tokenize dataset
        tokenized_dataset = formatted_dataset.map(
            lambda x: self.tokenizer(
                x["text"],
                max_length=self.config.max_position_embeddings,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ),
            batched=True,
            remove_columns=["text"]
        )
        
        tokenized_dataset.set_format(type="torch")
        return tokenized_dataset

    def _initialize_memory(self, knowledge_base: list):
        """Initialize memory with domain knowledge"""
        self.memory_cache.process_knowledge_base(
            knowledge_base,
            save_path=self.train_args.get("memory_save_path", "./memory_db")
        )
        self.accelerator.wait_for_everyone()

    def train(self):
        """Main training loop with memory integration"""
        # Prepare data
        dataset = self._prepare_dataset()
        dataloader = DataLoader(
            dataset,
            batch_size=self.train_args.get("batch_size", 2),
            collate_fn=default_data_collator,
            shuffle=True
        )
        dataloader = self.accelerator.prepare(dataloader)

        # Initialize memory with knowledge base
        if self.train_args.get("knowledge_base"):
            self._initialize_memory(self.train_args["knowledge_base"])

        # Training loop
        progress_bar = tqdm(
            range(self.train_args["total_steps"]),
            disable=not self.accelerator.is_local_main_process
        )
        
        current_step = 0
        self.model.train()
        
        while current_step < self.train_args["total_steps"]:
            for batch in dataloader:
                if current_step >= self.train_args["total_steps"]:
                    break
                
                with self.accelerator.accumulate(self.model):
                    # Forward pass with memory integration
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        past_key_values=self.memory_cache,
                        use_cache=True
                    )
                    
                    # Backward pass
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    
                    # Gradient management
                    if self.accelerator.sync_gradients:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.train_args.get("max_grad_norm", 1.0)
                        )
                    
                    # Parameter update
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                    
                    # Memory update
                    if current_step % self.config.memory_update_interval == 0:
                        context = self.tokenizer.decode(
                            batch["input_ids"][0][-self.config.memory_update_interval:],
                            skip_special_tokens=True
                        )
                        self.memory_cache.update_memory(
                            query=context,
                            layer_idx=list(self.config.memory_layers.keys())[0],
                            batch_idx=0,
                            memory_head_indices=list(self.config.memory_layers.values())[0]
                        )
                    
                    # Logging and checkpointing
                    if current_step % self.train_args.get("logging_steps", 50) == 0:
                        self.accelerator.log({
                            "loss": loss.item(),
                            "learning_rate": self.lr_scheduler.get_last_lr()[0],
                            "step": current_step
                        })
                        
                    if current_step % self.train_args.get("save_steps", 1000) == 0:
                        self.accelerator.save_state(
                            f"{self.train_args['output_dir']}/checkpoint-{current_step}"
                        )
                    
                    progress_bar.update(1)
                    current_step += 1

        # Final save
        self.accelerator.wait_for_everyone()
        self.accelerator.save_model(
            self.accelerator.unwrap_model(self.model),
            self.train_args["output_dir"],
            safe_serialization=True
        )

if __name__ == "__main__":
    # Example configuration for MetaMathQA
    config = M3_LlamaConfig(
        vocab_size=32000,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        memory_layers={3: [0, 2], 5: [1]},
        memory_token_length=128,
        num_memory_chunks=5,
        memory_update_interval=64,
        tokenizer_name="meta-math/MetaMathQA"
    )
    
    train_args = {
        "dataset_path": "meta-math/MetaMathQA",
        "dataset_format": "MetaMathQA",
        "knowledge_base": [
            "Mathematical problem-solving techniques",
            "Algebraic manipulation rules",
            "Complex number operations"
        ],
        "batch_size": 4,
        "total_steps": 10000,
        "learning_rate": 3e-5,
        "output_dir": "./math_model_checkpoints",
        "logging_steps": 50,
        "save_steps": 1000,
        "max_grad_norm": 1.0,
        "max_samples": 5000  # For quick testing
    }
    
    trainer = MemoryEnhancedTrainer(config, train_args)
    trainer.train()