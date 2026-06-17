import numpy as np

from app.config import load_settings


class GeminiEmbedder:
    def __init__(self) -> None:
        self.settings = load_settings()
        self.model_name = self.settings["model_name"]
        self._client = None
        self._types = None

    def _ensure_client(self):
        if not self.settings["api_key"]:
            raise RuntimeError("缺少 GEMINI_API_KEY 或 GOOGLE_API_KEY")
        if self._client is None:
            from google import genai
            from google.genai import types

            self._client = genai.Client(api_key=self.settings["api_key"])
            self._types = types
        return self._client, self._types

    def embed_media_bytes(self, data: bytes, mime_type: str) -> np.ndarray:
        client, types = self._ensure_client()
        response = client.models.embed_content(
            model=self.model_name,
            contents=types.Part.from_bytes(data=data, mime_type=mime_type),
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        return np.array(response.embeddings[0].values, dtype=np.float32)

    def embed_text(self, query: str) -> np.ndarray:
        client, types = self._ensure_client()
        response = client.models.embed_content(
            model=self.model_name,
            contents=query,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        return np.array(response.embeddings[0].values, dtype=np.float32)

