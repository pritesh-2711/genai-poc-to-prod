# Research Paper Chat Application

A production-oriented chat application built step by step — from a bare LLM call to a multi-user API with memory, guardrails, and safety evaluations.

## Quick Start

```bash
python main.py          # CLI mode
python api_server.py    # REST API (FastAPI + Uvicorn)
```

----

## Learning path for beginners

Each branch adds one capability on top of the previous. Follow them in order.

### Branch: `explore`

- Read `notebooks/explore_langchain.ipynb`
- Covers LangChain basics: prompt templates, chains, LLM providers (Ollama / OpenAI)

### Branch: `feature/beginners-app`

- Minimalistic project structure: logging, configs, Pydantic models, custom exceptions
- Chat capability with swappable LLM providers (Ollama / OpenAI)
- Simple UI via Chainlit and a CLI chat system

### Branch: `feature/memory`

- Read `notebooks/explore_memory.ipynb` — validates the DB schema and the full conversation round-trip before any application code
- Read `docs/design_intuition.md` — explains why a three-table schema (users → sessions → chats) is the right design for multi-user isolation
- Run `sql/init.sql` against a local PostgreSQL instance (pgAdmin or psql)
- Key additions in `src/memory/repository.py`: user auth (bcrypt), session management, conversation history retrieval
- Conversation history is injected into the LLM system prompt on every call via `ChatService._build_system_prompt`

### Branch: `feature/compliance`

- Read `notebooks/explore_safety_evaluations.ipynb` — explores both NeMo Guardrails (Colang flows, topical rails, `self check` rails) and DeepEval safety metrics before wiring either into the app
- The notebook covers:
  - How Colang pattern-matching blocks harmful inputs without an extra LLM call
  - Why `self check input`/`self check output` rails require a tuned prompt and when they cause false positives
  - How DeepEval's `ToxicityMetric`, `BiasMetric`, and `GEval` can be used as pre-LLM input guards
- Application changes:
  - `src/guardrails/input_guard.py` — runs three DeepEval metrics concurrently (asyncio.gather) before every LLM call
  - `src/core/exceptions.py` — `InputBlockedError` distinguishes a blocked message from an LLM failure
  - `src/core/models.py` + `configs/config.yaml` — `GuardrailsConfig` controls which guards are active and which evaluator model to use
  - `src/api/chat.py` — blocked messages are saved as a polite assistant reply (HTTP 201) instead of surfacing as an error

----

**What initial application version (branch: `beginners-app`) was lacking:**

- Query is sent directly to the LLM without understanding intent or complexity
- No conversation history — only the current message is used as context
- Application intelligence does not improve over time
- Responses are limited to the LLM's training knowledge
- Harmful content and jailbreaking are not handled
- No way to evaluate response quality
- No additional tools integrated with the LLM
- Cannot support long, multi-step workflows
- Not built for multiple users

**These are the checklists we are solving throughout this repo:**

- Query Analysis
- Memory : short-term, long-term, intersession, user feedbacks & preferences 
- Feedback Learning
- RAG
- Guardrails
- Evaluations
- Tool calling
- Workflows & Agents
- A good system design

----

**What has been covered from the checklist:**

- [x] **Memory** — PostgreSQL-backed conversation history per user per session. History is injected as context into every LLM call. CURRENTLY ONLY SUPPORTS SHORT TERM MEMORY.
- [x] **Multi-user support** — JWT-authenticated REST API. Each user sees only their own sessions and messages.
- [x] **Guardrails (input safety)** — DeepEval metrics (`ToxicityMetric`, `BiasMetric`, `GEval`) run concurrently before every LLM call. Configurable per guard, configurable evaluator model. Blocked messages return a friendly assistant reply, not an error.

**Still to address:**

- [ ] Query Analysis / Intent detection
- [ ] Feedback Learning
- [ ] RAG
- [ ] Post-LLM Evaluations (response quality, hallucination, relevance)
- [ ] Tool calling
- [ ] Workflows
- [ ] Agents

----

## License

See LICENSE file for details.