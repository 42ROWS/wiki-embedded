"""
RAG (Retrieval-Augmented Generation) system for code reuse.

This module implements a vector-based code memory system using Pinecone FREE tier:
- Stores successful/failed transformations
- Searches for similar past solutions (similarity > 85%)
- Enables intelligent code reuse
- Uses value-based cleanup (quality score) instead of TTL

Architecture:
- Pinecone Serverless (AWS us-east-1, FREE tier)
- Embeddings via Pinecone Inference (5M tokens/month included)
- Storage: 2GB (~300K vectors, we use 5K = 1.83%)
- Cost: $0.00/month forever ✅
"""
from .storage import TransformationStorage
from .pinecone_client import PineconeClient

__all__ = [
    "TransformationStorage",
    "PineconeClient",
]
