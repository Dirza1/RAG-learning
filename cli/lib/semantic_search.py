from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticSearch:
    def __init__(self) -> None:
        self.model:SentenceTransformer = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embedding(self,text:str):
        if text.replace(" ","") == "":
            raise ValueError("Text nly contains white space")
        embedding = self.model.encode([text])[0]
        return embedding


def verify_model()->None:
    ss:SemanticSearch = SemanticSearch()
    print(f"Model loaded: {ss.model}")
    print(f"Max sequence length: {ss.model.max_seq_length}")

def embed_text(text:str) -> None:
    ss:SemanticSearch = SemanticSearch()
    embedding = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")