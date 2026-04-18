import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

# Setup Proxy
os.environ['http_proxy'] = 'http://127.0.0.1:1080'
os.environ['https_proxy'] = 'http://127.0.0.1:1080'

def generate_video_embeddings_inline(video_dir, output_json):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY not found.")
        return

    client = genai.Client(api_key=api_key)
    video_path = Path(video_dir)
    
    # Load existing embeddings if any
    if Path(output_json).exists():
        try:
            with open(output_json, 'r', encoding='utf-8') as f:
                all_embeddings = json.load(f)
        except Exception:
            all_embeddings = {}
    else:
        all_embeddings = {}

    video_files = list(video_path.glob("*.mp4"))
    print(f"Found {len(video_files)} videos in {video_dir}")

    for vid_file in video_files:
        if vid_file.name in all_embeddings:
            print(f"Skipping {vid_file.name}, embedding already exists.")
            continue

        print(f"Processing {vid_file.name} (Inline Mode)...", end="", flush=True)
        try:
            # 1. Read local file as bytes
            with open(vid_file, "rb") as f:
                video_bytes = f.read()
            
            # 2. Generate embedding directly (Inline Data)
            res = client.models.embed_content(
                model="gemini-embedding-2-preview",
                contents=types.Part.from_bytes(
                    data=video_bytes,
                    mime_type="video/mp4"
                ),
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            
            embedding_vector = res.embeddings[0].values
            
            all_embeddings[vid_file.name] = {
                "embedding": embedding_vector,
                "file_path": str(vid_file)
            }
            
            # Save progress incrementally
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(all_embeddings, f, ensure_ascii=False, indent=2)
            
            print(" Success.")
            
            # 稍微停顿一下，避免触发 API 频率限制
            time.sleep(1)

        except Exception as e:
            print(f"\nError processing {vid_file.name}: {e}")
            # 如果遇到频率限制，多等一会儿
            if "429" in str(e):
                print("Rate limit hit, sleeping for 10s...")
                time.sleep(10)

    print(f"\nAll done! Embeddings saved to {output_json}")

if __name__ == "__main__":
    generate_video_embeddings_inline("test_video_720", "embeddings.json")
