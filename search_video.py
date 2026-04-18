import os
import json
import numpy as np
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

# Setup Proxy
os.environ['http_proxy'] = 'http://127.0.0.1:1080'
os.environ['https_proxy'] = 'http://127.0.0.1:1080'

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

class VideoSearcher:
    def __init__(self, embeddings_path):
        self.embeddings_path = embeddings_path
        self.video_data = {}
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        self.load_embeddings()

    def load_embeddings(self):
        if not os.path.exists(self.embeddings_path):
            print(f"Error: {self.embeddings_path} not found. Please run generate_embeddings.py first.")
            return
        with open(self.embeddings_path, 'r', encoding='utf-8') as f:
            self.video_data = json.load(f)
        print(f"Loaded {len(self.video_data)} video embeddings.")

    def search(self, query_text, top_k=3):
        print(f"\nQuery: '{query_text}'")
        
        # 1. Generate embedding for the text query
        res = self.client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=query_text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        query_vector = res.embeddings[0].values

        # 2. Calculate similarities
        results = []
        for file_name, data in self.video_data.items():
            sim = cosine_similarity(query_vector, data['embedding'])
            results.append((file_name, sim))

        # 3. Sort and return top K
        results.sort(key=lambda x: x[1], reverse=True)
        
        print(f"{'Rank':<5} | {'Video File':<30} | {'Similarity':<10}")
        print("-" * 50)
        for i, (file_name, score) in enumerate(results[:top_k]):
            print(f"{i+1:<5} | {file_name:<30} | {score:.4f}")
        
        return results[:top_k]

    def check_video_similarity(self, video_a, video_b):
        if video_a not in self.video_data or video_b not in self.video_data:
            print("One or both videos not found in embeddings.")
            return
        
        sim = cosine_similarity(self.video_data[video_a]['embedding'], 
                                self.video_data[video_b]['embedding'])
        print(f"\nSimilarity between '{video_a}' and '{video_b}': {sim:.4f}")

if __name__ == "__main__":
    searcher = VideoSearcher("embeddings.json")
    
    queries = [
        "一个小女孩在公园面吹泡泡",
        "一个小女孩在捉迷藏",
        "一个直播演员正在直播，演员是男性，正面面对镜头",
        "一个拿着笔记本电脑的女性在办公室内行走",
        "一只手把硬盘推入桌面上的NAS",
        "公园的板凳上放着无人机、无人机遥控器、充电宝"
    ]
    
    for q in queries:
        searcher.search(q, top_k=3)
