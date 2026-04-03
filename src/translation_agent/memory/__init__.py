"""Memory package."""

from translation_agent.search_index import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL_ID,
    cosine_similarity,
    deserialize_embedding,
    embed_text,
    embedding_metadata_for_entry,
    embedding_metadata_for_query,
    search_document_for_entry,
    search_document_for_query,
    serialize_embedding,
)

from .consolidation import (
    DeterministicMemoryConsolidationBackend,
    MemoryConsolidationBackend,
)
from .prompt_evolution import (
    DeterministicPromptEvolutionBackend,
    PromptEvolutionBackend,
)
from .prompt_resolution import PromptResolver, ProposalBackedPromptResolver
from .recall import (
    BlobBackedLongTermMemoryStore,
    InMemoryLongTermMemoryStore,
    LongTermMemoryRecallBackend,
    MemoryRecallBackend,
    OperationalStoreLongTermMemoryStore,
)
from .staging import DeterministicMemoryStagingBackend, MemoryStagingBackend

__all__ = [
    "BlobBackedLongTermMemoryStore",
    "DeterministicMemoryConsolidationBackend",
    "DeterministicMemoryStagingBackend",
    "DeterministicPromptEvolutionBackend",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_MODEL_ID",
    "InMemoryLongTermMemoryStore",
    "LongTermMemoryRecallBackend",
    "MemoryConsolidationBackend",
    "MemoryRecallBackend",
    "MemoryStagingBackend",
    "OperationalStoreLongTermMemoryStore",
    "PromptResolver",
    "PromptEvolutionBackend",
    "ProposalBackedPromptResolver",
    "cosine_similarity",
    "deserialize_embedding",
    "embed_text",
    "embedding_metadata_for_entry",
    "embedding_metadata_for_query",
    "search_document_for_entry",
    "search_document_for_query",
    "serialize_embedding",
]
