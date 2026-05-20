"""
Pinecone client for vector storage - FREE tier configuration.

Pinecone Serverless FREE tier (as of Dec 2025):
- Storage: 2 GB (~300K vectors with 1536 dim + metadata)
- Region: AWS us-east-1 only
- Indexes: 5 max
- Namespaces: 100 per index
- Write Units: 2M/month (~300K writes)
- Read Units: 1M/month (~100K reads)
- Embeddings: 5M tokens/month (via Pinecone Inference)
- Cost: $0.00/month forever ✅

Our usage with 5,000 vectors:
- Storage: ~36 MB (1.83% of FREE tier)
- Growth margin: 55x before hitting limit
- Sustainable: 5-10+ years on FREE tier
"""
import os
from typing import Final

from pinecone import Pinecone, ServerlessSpec

from polars_runner.core.constants import RAG_CONFIG


class PineconeClient:
    """
    Client for Pinecone vector database (FREE tier).
    
    Features:
    - Auto-creates index if missing
    - Uses Serverless (FREE tier)
    - AWS us-east-1 (FREE region)
    - Cosine similarity metric
    """
    
    # Configuration from constants (DRY principle)
    INDEX_NAME: Final[str] = "ai-data-transformer"
    DIMENSION: Final[int] = RAG_CONFIG.embedding_dimension
    CLOUD: Final[str] = "aws"
    REGION: Final[str] = "us-east-1"  # FREE tier region
    METRIC: Final[str] = "cosine"
    
    def __init__(self):
        """Initialize Pinecone client with FREE tier config."""
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError(
                "PINECONE_API_KEY environment variable not set. "
                "Get your FREE API key at: https://app.pinecone.io"
            )
        
        self._pc = Pinecone(api_key=api_key)
        self._ensure_index_exists()
    
    def _ensure_index_exists(self) -> None:
        """Create index if it doesn't exist (idempotent)."""
        existing_indexes = self._pc.list_indexes().names()
        
        if self.INDEX_NAME not in existing_indexes:
            self._pc.create_index(
                name=self.INDEX_NAME,
                dimension=self.DIMENSION,
                metric=self.METRIC,
                spec=ServerlessSpec(
                    cloud=self.CLOUD,
                    region=self.REGION,
                ),
            )
    
    def get_index(self):
        """
        Get index instance for operations.
        
        Returns:
            Pinecone Index instance
        """
        return self._pc.Index(self.INDEX_NAME)
    
    def get_inference(self):
        """
        Get inference client for FREE embeddings.
        
        Pinecone FREE tier includes 5M tokens/month for embeddings.
        This is more than enough for our needs (we use ~50K/month).
        
        Returns:
            Pinecone Inference client
        """
        return self._pc.inference
