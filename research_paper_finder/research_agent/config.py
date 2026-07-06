import os
from pathlib import Path


def _load_dotenv() -> None:

    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        env_path = Path.cwd() / ".env"

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


_load_dotenv()

SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1"

OPENALEX_API = "https://api.openalex.org"

ARXIV_API = "http://export.arxiv.org/api/query"

SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

REQUEST_TIMEOUT = 30
