# Explicit-Memory
This project aims to explore methods to transform existing pretrained LLMs into M3 models through supervised fine-tuning.

<img width="1330" alt="截屏2025-04-11 23 01 02" src="https://github.com/user-attachments/assets/837f7392-39fe-4a11-a628-0ae234359622" />

## ✨ What's New
+  [2025.04.11] We release our inference, training code and training data.

## 🗓 Coming Soon
- [x] Inference and training code are released
- [x] Training Data are released
- [ ] M3 model is under training...

## Data
We use a series of data filtering techniques to get our training data. Details can be seen at this [repo](https://github.com/Ezrill-Lin/DataPipeline-for-ExplicitMemory), including the summary of the data pipeline building work, usage of the pipeline, and intro of each section of the pipeline. Training data is available [here](https://drive.google.com/file/d/1DiB-rhDx1w0Ze5UGqhu4o6gT2bADrnk5/view?usp=sharing)

## Model
Based on huggingface transformers, we implement inference and training code for llama model at [m3_model.py](https://github.com/szjiozi/Explicit-Memory/blob/main/m3llama/m3_model.py). We also implement a memory class to build the vector database, retrieve knowledge and update explicit memory that can be used by the model at [memory.py](https://github.com/szjiozi/Explicit-Memory/blob/main/m3llama/memory.py).
