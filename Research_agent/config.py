from dataclasses import dataclass

@dataclass
class qdrant:
    PORT = 6333
    HOST = "localhost"
    IMAGE_COLLECTION_NAME = "research_paper_rag_hybrid_v6"
    TEXT_COLLECTION_NAME = "research_paper_rag_text_v5"
    FUSION_METHOD= "Weigthed_Fusion"
    ALPHA_VALUE = 0.4

@dataclass
class vlm_model:
    VLM_MODEl = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"

@dataclass
class embedding_model:
    dense_text_model = "mixedbread-ai/mxbai-embed-large-v1"
    sparse_text_model = "Qdrant/bm25"
    image_model = "google/siglip-base-patch16-512"




    