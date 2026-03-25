# Embedding - convert text to vector
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingManager():
    def __init__(self, model_name: str = "multi-qa-MiniLM-L6-cos-v1"):
        self.model_name = model_name
        self.load_model()
        self.model = None
        
    def load_model(self):
        try:
            print(f"Loading model {self.model_name} ...")
            self.model = SentenceTransformer(self.model_name)
            print(f"\nModel embedding: {self.model.get_sentence_embedding_dimension()}")
            print(f"\nModel loaded successfully.")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise
        
    def generate_embedding(self, texts: list[str]) -> np.array:
        if not self.model:
            raise ValueError("Model not loaded.")
        print(f"Genrating embedding for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, batch_size=32, show_progress_bar=True)
        print(f"Embedding generated, shape: {embeddings.shape}")
        return embeddings