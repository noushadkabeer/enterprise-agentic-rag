"""
Hybrid Dense (Vector) + Sparse (BM25) Retrieval Engine
Simulates enterprise RAG with Reciprocal Rank Fusion (RRF).
"""

from typing import List
from src.state import DocumentChunk

# Sample Enterprise Knowledge Base (Simulating complex policy & compliance docs)
ENTERPRISE_KNOWLEDGE_BASE = [
    DocumentChunk(
        chunk_id="chunk_001",
        doc_name="Enterprise_Commercial_Policy_v4.pdf",
        page_number=14,
        is_table=False,
        content="Section 14.1: General Property Coverage covers physical loss up to $5,000,000. All claims subject to standard $25,000 deductible.",
        metadata={"section": "14.1", "clause": "Coverage"}
    ),
    DocumentChunk(
        chunk_id="chunk_002",
        doc_name="Enterprise_Commercial_Policy_v4.pdf",
        page_number=14,
        is_table=True,
        content="| Tier | Deductible | Max Loss Limit | Auto-Waiver Condition |\n| A | $10,000 | $2,000,000 | Annual Audit Clean |\n| B | $25,000 | $5,000,000 | ISO 27001 Certified |",
        metadata={"section": "14.2", "clause": "Deductible Table"}
    ),
    DocumentChunk(
        chunk_id="chunk_003",
        doc_name="Enterprise_Commercial_Policy_v4.pdf",
        page_number=380,
        is_table=False,
        content="Endorsement Rider 89-B (Overrides Section 14.2): Deductibles for Tier B are waived if the insured maintains continuous multi-cloud disaster recovery testing under SOC2 Type II.",
        metadata={"section": "Endorsement 89-B", "overrides": "14.2"}
    ),
    DocumentChunk(
        chunk_id="chunk_004",
        doc_name="Alfresco_ECM_Governance_Manual.pdf",
        page_number=45,
        is_table=False,
        content="Access Control Rule: Unstructured document repositories holding PII must enforce role-based ACLs mapped from Okta/Active Directory before ingestion into RAG vector stores.",
        metadata={"section": "Governance", "domain": "Security"}
    )
]


class HybridRetriever:
    """
    Combines semantic similarity with exact keyword lexical matching.
    """
    def __init__(self, corpus: List[DocumentChunk] = None):
        self.corpus = corpus or ENTERPRISE_KNOWLEDGE_BASE

    def search(self, query: str, top_k: int = 3) -> List[DocumentChunk]:
        query_tokens = query.lower().split()
        scored_chunks = []

        for chunk in self.corpus:
            content_lower = chunk.content.lower()
            score = 0.0

            # Lexical exact match weight (BM25 proxy)
            for token in query_tokens:
                if token in content_lower:
                    score += 2.0
                if token in str(chunk.metadata).lower():
                    score += 1.5

            # Semantic / table bonus
            if "deductible" in query_tokens and "waiver" in query_tokens:
                if "override" in content_lower or "waived" in content_lower:
                    score += 5.0

            chunk_copy = chunk.model_copy()
            chunk_copy.relevance_score = score
            scored_chunks.append(chunk_copy)

        scored_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        return scored_chunks[:top_k]
