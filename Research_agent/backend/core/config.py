from dataclasses import dataclass
from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parents[2]  # Research_agent/
PROJECT_ROOT = BASE_DIR.parent  # repo root, where .env lives
CHECKPOINT_DB_PATH = BASE_DIR / "checkpoints.sqlite"


def load_environment() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")

        if key and key not in os.environ:
            os.environ[key] = value


load_environment()

@dataclass
class qdrant:
    PORT = 6333
    HOST = "localhost"
    IMAGE_COLLECTION_NAME = "research_paper_image_v6"
    TEXT_COLLECTION_NAME = "research_papers_text_v5"
    FUSION_METHOD = "Weighted_Fusion"
    ALPHA_VALUE = 0.4
    SEARCH_LIMIT = 10
    FINAL_DOC_LIMIT = 5

@dataclass
class vlm_model:
    VLM_MODEl = "openai/gpt-oss-120b"

@dataclass
class embedding_model:
    dense_text_model = "mixedbread-ai/mxbai-embed-large-v1"
    sparse_text_model = "Qdrant/bm25"
    image_model = "google/siglip-base-patch16-512"


@dataclass
class small_llm_config:
    MODEL_NAME = "openai/gpt-oss-20b"


@dataclass
class app_config:
    APP_NAME = "Research Paper RAG Agent"
    SUMMARY_TOKEN_LIMIT = 50000


    
