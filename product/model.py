import os
# Disable TensorFlow warnings and avoid TF conflicts
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Define the MedDRA data path
MEDDRA_PATH = "data/meddra_terms.csv"

class MedDRAMatcher:
    def __init__(self, meddra_csv_path):
        self.meddra_df = pd.read_csv(meddra_csv_path)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Pre-compute embeddings (VERY IMPORTANT)
        self.pt_embeddings = self.model.encode(
            self.meddra_df["pt_name"].tolist(),
            show_progress_bar=True
        )

    def predict(self, adr_text, top_k=5):
        adr_text = adr_text.lower()

        adr_emb = self.model.encode([adr_text])
        sims = cosine_similarity(adr_emb, self.pt_embeddings)[0]

        top_idx = np.argsort(sims)[-top_k:][::-1]

        results = []
        for idx in top_idx:
            results.append({
                "pt_code": self.meddra_df.iloc[idx]["pt_code"],
                "pt_name": self.meddra_df.iloc[idx]["pt_name"],
                "score": float(sims[idx])
            })

        return results

# Create module-level matcher instance for compatibility
matcher = MedDRAMatcher(MEDDRA_PATH)
