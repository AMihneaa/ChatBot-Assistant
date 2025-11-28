from pathlib import Path
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MD_PATH = Path("frontend_knowledge_base.md")
INDEX_PATH = Path("faiss_index.bin")
META_PATH = Path("faiss_metadata.json")

CHUNK_SIZE = 800       
CHUNK_OVERLAP = 200    


def split_text(text: str, chunk_size: int, overlap: int):
  chunks = []
  start = 0
  n = len(text)

  while start < n:
    end = min(start + chunk_size, n)
    chunk = text[start:end]
    chunks.append(chunk)
    start += chunk_size - overlap

  return chunks


def main():
  text = MD_PATH.read_text(encoding="utf-8")
  chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

  print(f"[i] Avem {len(chunks)} chunk-uri de text.")

  model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
  embeddings = model.encode(chunks, show_progress_bar=True)
  embeddings = np.asarray(embeddings, dtype="float32")

  dim = embeddings.shape[1]
  index = faiss.IndexFlatL2(dim)
  index.add(embeddings)

  faiss.write_index(index, str(INDEX_PATH))

  META_PATH.write_text(
    json.dumps({"chunks": chunks}, ensure_ascii=False, indent=2),
    encoding="utf-8",
  )

  print(f"[i] Index salvat în {INDEX_PATH}")
  print(f"[i] Metadate salvate în {META_PATH}")


if __name__ == "__main__":
  main()
