from typing import Iterable

from paper_qa_pipeline.models import TextChunk


class QdrantChunkStore:

    def upsert_chunks(self, chunks: Iterable[TextChunk]) -> None:

        try:
            import qdrant_client  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Qdrant storage needs qdrant-client. Install it with "
                "`pip install qdrant-client`."
            ) from exc

        raise NotImplementedError(
            "Qdrant collection creation and vector upsert will be implemented "
            "after choosing the embedding model."
        )
