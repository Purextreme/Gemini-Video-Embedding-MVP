import os
import json
import shutil
import numpy as np
import sys
import time
from pathlib import Path
from dotenv import load_dotenv
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import normalize
from google import genai
from google.genai import types

# Force UTF-8 Mode
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Ensure we load the LATEST key
load_dotenv(override=True)

# Proxy
os.environ['http_proxy'] = 'http://127.0.0.1:1080'
os.environ['https_proxy'] = 'http://127.0.0.1:1080'

def get_semantic_name(client, video_path, cluster_filenames):
    """
    Get multi-tag semantic name (e.g., Tag1_Tag2_Tag3).
    Provides cluster filenames as context to help Gemini find commonalities.
    """
    model_name = "gemini-3.1-flash-lite-preview"
    try:
        with open(video_path, "rb") as f:
            video_bytes = f.read()
            
        # Context from filenames in this cluster (first 10 files)
        context_files = ", ".join(cluster_filenames[:10])
        
        # PROMPT FOR PRECISION TAGS WITH FILENAME FILTERING
        prompt = f"""
        Analyze this video and its cluster context.
        Cluster filenames for reference: {context_files}
        
        Task: Provide 3-5 highly descriptive keywords (Subject_Environment_Object_Vibe).
        
        Rules:
        1. If filenames are generic codes (like 'IMG_1284', 'DCR_123', '01', '03_1'), IGNORE them and rely ONLY on visual analysis.
        2. If filenames contain descriptive words (like 'Wireless', 'Office', 'Product'), use them to enhance your visual analysis.
        3. Format: Chinese language, separated by '_'.
        4. Target: [Subject]_[Environment]_[Object]_[Context].
        
        Return ONLY the tags. Example: '女生_办公室_手机_职场精英'.
        """
        
        generate_content_config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
            tools=[types.Tool(google_search=types.GoogleSearch())]
        )

        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=video_bytes, mime_type="video/mp4"),
                        types.Part.from_text(text=prompt)
                    ]
                )
            ],
            config=generate_content_config
        )
        
        name = response.text.strip().replace('"', '').replace("'", "").replace("\n", "").replace(" ", "")
        # Sanitize for filesystem
        for char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|', '.', '，', '。']:
            name = name.replace(char, '_')
        return name if name else "Uncategorized_Group"
    except Exception as e:
        print(f"  API Call failed for {video_path.name}")
        return f"Group_{int(time.time())}"

def cluster_videos(embeddings_json, video_dir, output_root):
    print(f"Loading embeddings from {embeddings_json}...")
    with open(embeddings_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    filenames = list(data.keys())
    embeddings = np.array([data[f]['embedding'] for f in filenames])
    
    print(f"Clustering {len(filenames)} videos...")
    embeddings_norm = normalize(embeddings)
    
    # HDBSCAN parameters
    clusterer = HDBSCAN(min_cluster_size=2, cluster_selection_epsilon=0.18)
    labels = clusterer.fit_predict(embeddings_norm)
    
    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    print(f"Found {n_clusters} clusters and {list(labels).count(-1)} noise points.")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    client = genai.Client(api_key=api_key)
    
    output_path = Path(output_root)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(exist_ok=True)
    
    processed_count = 0
    video_path_root = Path(video_dir)

    for i in unique_labels:
        cluster_indices = np.where(labels == i)[0]
        cluster_files = [filenames[idx] for idx in cluster_indices]
        
        if i == -1:
            semantic_name = "Others"
        else:
            centroid = np.mean(embeddings_norm[cluster_indices], axis=0)
            distances = np.linalg.norm(embeddings_norm[cluster_indices] - centroid, axis=1)
            rep_file = filenames[cluster_indices[np.argmin(distances)]]
            
            print(f"Cluster {i}: {len(cluster_files)} videos. Representative: {rep_file}...")
            # Use multi-tag naming
            semantic_name = get_semantic_name(client, video_path_root / rep_file, cluster_files)
            
            try:
                print(f"  -> Label: {semantic_name}")
            except:
                print(f"  -> Label received (non-ASCII)")
        
        cluster_dir = output_path / semantic_name
        cluster_dir.mkdir(exist_ok=True, parents=True)
        
        for f_name in cluster_files:
            src = video_path_root / f_name
            if src.exists():
                shutil.copy2(src, cluster_dir / f_name)
                processed_count += 1
            else:
                print(f"  Warning: {f_name} not found.")

    print(f"\nCompleted! Processed {processed_count}/{len(filenames)} files.")

if __name__ == "__main__":
    cluster_videos(
        embeddings_json="embeddings.json",
        video_dir="test_video_720",
        output_root="clustered_videos"
    )
