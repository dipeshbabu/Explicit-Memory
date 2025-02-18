""" m3_knowledge_retriever for bge-m3"""
import torch
from transformers import AutoModel, AutoTokenizer
from typing import List

class Retriever:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.config = self.model.config
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model.to(self.device)

    def encode(self, text: List[str]|str):
        with torch.no_grad():
            encoded_input = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
            model_output = self.model(**encoded_input)
            embeddings = model_output[0][:, 0]
            return embeddings

# sentences_1 = ["样例数据-1", "样例数据-2"]
# sentences_2 = ["样例数据-3", "样例数据-4"]
# with torch.no_grad():
#     encoded_input_1 = tokenizer(sentences_1, padding=True, truncation=True, return_tensors='pt')
#     encoded_input_2 = tokenizer(sentences_2, padding=True, truncation=True, return_tensors='pt')
#     model_output_1 = model(**encoded_input_1)
#     model_output_2 = model(**encoded_input_2)
#     embeddings_1 = model_output_1[0][:, 0]
#     embeddings_2 = model_output_2[0][:, 0]
#     similarity = embeddings_1 @ embeddings_2.T
#     print(similarity)