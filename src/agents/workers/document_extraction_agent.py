"""Worker agent specialized in document entity extraction via RaV-IDP."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent

from ...chat_service import ChatService
from ...core.models import ChatRecord
from .._shared import AgentRunResult, extract_agent_run_result

_DOCUMENT_EXTRACTION_PROMPT = """You are the Document Extraction Worker.

Role:
- Extract structured entities and assess quality/fidelity from documents using
  the RaV-IDP tools provided to you.
- Use `rav_idp_process_and_ingest` to run the full extraction pipeline and
  persist the entity records to JSON for downstream use.
- Use `rav_idp_get_document_fidelity` when only quality metrics are needed
  (faster — does not persist output).
- Return a clear, structured summary of what was extracted: entity types,
  counts, average fidelity, and any low-confidence findings.
- If the document path is not accessible or extraction fails, report the error
  clearly so the supervisor can inform the user.

Working style:
- Use `rav_idp_get_document_fidelity` first when the user only asks about
  document quality or confidence — it is faster.
- Use `rav_idp_process_and_ingest` when the user needs the full entity set,
  or when entities need to be saved for downstream analysis.
- Do not fabricate entity data — only report what the tool returned.
- Quote fidelity scores and entity type breakdowns from the tool output.
- Note the `output_path` when `rav_idp_process_and_ingest` is used, so the
  supervisor can reference it if needed.
"""


class DocumentExtractionWorkerAgent:
    def __init__(
        self,
        chat_service: ChatService,
        tools: list[Any],
        short_term_history: list[ChatRecord] | None = None,
        long_term_history: list[dict] | None = None,
    ) -> None:
        self._chat_service = chat_service
        self._tools = tools
        self._short_term_history = short_term_history or []
        self._long_term_history = long_term_history or []

    def _build_system_prompt(self) -> str:
        base = self._chat_service._build_system_prompt(  # noqa: SLF001
            short_term_history=self._short_term_history,
            long_term_history=self._long_term_history,
            rag_context=None,
        )
        return f"{base}\n\n{_DOCUMENT_EXTRACTION_PROMPT}"

    async def arun(self, task: str) -> AgentRunResult:
        graph = create_agent(
            model=self._chat_service.llm_provider.llm,
            tools=self._tools,
            system_prompt=self._build_system_prompt(),
            name="document_extraction_worker",
        )
        result = await graph.ainvoke({"messages": [{"role": "user", "content": task}]})
        return extract_agent_run_result(
            result,
            "The document extraction worker could not produce a final response.",
        )
