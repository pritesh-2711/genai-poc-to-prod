"""LangGraph orchestrators for fast/deep RAG modes.

Hierarchy (bottom-to-top):
    BaseOrchestrator          — shared node implementations
        FastOrchestrator      — fast subgraph (embed → retrieve → rerank → generate)
        DeepOrchestrator      — deep subgraph (analyze → fan-out → rerank → generate → validate)
    RAGOrchestrator           — top-level router graph (fast | deep)
"""

from .rag_orchestrator import RAGOrchestrator

__all__ = ["RAGOrchestrator"]
