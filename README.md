# Cost-Optimized RAG Documentation Assistant

A configurable Retrieval-Augmented Generation (RAG) assistant that answers
technical questions from the official FastAPI documentation, cites its sources,
and minimizes LLM spending with exact and semantic caching.

## Project goal

Build a documentation assistant that:

- Retrieves relevant FastAPI documentation with hybrid search.
- Generates answers that are grounded in retrieved documentation.
- Includes links to the source pages used for each answer.
- Reuses verified answers for repeated or semantically similar questions.
- Invalidates cached answers when their source documentation changes.
- Lets users choose between economy, balanced, and accuracy-focused settings.
- Measures retrieval quality, answer quality, latency, and estimated cost savings.

The initial knowledge base will contain the official FastAPI tutorial and guides.
The ingestion and retrieval components should remain reusable so that other
documentation sets can be added later.

## Why this project exists

A normal RAG application can call an embedding model, a reranker, and an LLM for
every question. That approach becomes expensive when users repeatedly ask the
same question in different words. This project adds a version-aware semantic
cache so a previously verified answer can be reused when it is safe to do so.

A cache hit avoids another answer-generation call. It is not necessarily
completely free because the application may still incur embedding, database,
and hosting costs. The dashboard will report avoided LLM cost separately from
the actual infrastructure cost.

## Planned request flow

1. Normalize the user's question.
2. Check for an exact cached match.
3. Create a query embedding and check for a high-confidence semantic match.
4. Confirm that the cached answer is unexpired and its cited documents are
   still current.
5. Return the cached answer when every cache validation succeeds.
6. Otherwise, run keyword and vector searches over the active documentation.
7. Combine and optionally rerank the retrieved chunks.
8. Generate a grounded answer with citations, or decline to answer when the
   retrieved evidence is insufficient.
9. Record quality, latency, token, and cost information.
10. Cache the answer only if it passes the configured quality requirements.

## Proposed technology stack

- **API:** Python and FastAPI
- **Application and vector database:** PostgreSQL with pgvector
- **Database access and migrations:** SQLAlchemy and Alembic
- **Keyword retrieval:** PostgreSQL full-text search
- **Embeddings:** Sentence Transformers by default, with an optional hosted model
- **Document extraction:** HTTPX and Beautiful Soup
- **LLM provider:** Provider-agnostic adapter with configurable models
- **Optional reranking:** Local cross-encoder or hosted reranking API
- **Dashboard:** Streamlit for the first version
- **Local development:** Docker Compose
- **Testing:** pytest

## Document ingestion

Install the project and development dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,local]'
```

Scrape and normalize every approved FastAPI documentation page:

```bash
python -m scripts.ingest_fastapi_docs
```

Useful development options:

```bash
# Process one page while developing the parser.
python -m scripts.ingest_fastapi_docs --document-id request-body

# Reparse saved HTML without sending network requests.
python -m scripts.ingest_fastapi_docs --use-local-html

# Validate fetching and parsing without writing output.
python -m scripts.ingest_fastapi_docs --dry-run
```

The source manifest is stored in `data/sources/fastapi_urls.json`. Original HTML
is written to `data/raw`, normalized documents to `data/documents`, and previous
document versions to `data/versions`. Generated data is ignored by Git; only the
directory placeholders and source manifest are committed.

The command exits with a nonzero status when any page fails. A single failed page
is recorded and reported without preventing the remaining sources from being
processed.

### Document chunking

After ingestion creates normalized JSON documents, split them into retrieval
chunks:

```bash
python -m scripts.chunk_documents
```

The default configuration uses a maximum of 600 `cl100k_base` tokens and 75
tokens of overlap. Override these values or process one document while tuning:

```bash
python -m scripts.chunk_documents \
  --document-id request-body \
  --max-tokens 500 \
  --overlap-tokens 75
```

Use `--dry-run` to validate and chunk inputs without writing output. Chunk files
are stored in `data/chunks/<document-id>.json` and include stable IDs, document
hashes, section metadata, citation URLs, token counts, code blocks, and content
hashes. Code blocks remain intact even when an unusually large block causes a
chunk to exceed the configured maximum.

### Vector indexing

Create local embeddings and synchronize the persistent vector index after
chunking:

```bash
python -m scripts.index_documents
```

The first run downloads the configured FastEmbed ONNX model. Embeddings are then
generated locally, so indexing does not require a paid embedding API. The
default `BAAI/bge-small-en-v1.5` model and index location can be changed:

```bash
python -m scripts.index_documents \
  --model BAAI/bge-small-en-v1.5 \
  --index-path data/index/vectors.sqlite3 \
  --batch-size 32
```

Use `--dry-run` to validate chunk files and preview the number of embeddings
needed without loading the model or modifying the index. Use `--document-id` for
an incremental single-document update, or `--rebuild` to recreate the complete
index. Generated indexes are ignored by Git.

The SQLite index stores chunk metadata separately from vectors and reuses one
embedding when multiple chunks have the same content hash. Re-running the
command embeds only new or changed content, updates metadata, removes stale
chunks, and cleans up unreferenced embeddings. Embedding input includes both
prose and fenced code, so code-only documentation chunks remain searchable.

### Vector retrieval

Search the indexed FastAPI documentation without calling an LLM. Hybrid search
is the default and combines semantic vector rankings with BM25-style keyword
rankings using Reciprocal Rank Fusion:

```bash
python -m scripts.search_documents \
  "How do I create a request body in FastAPI?"
```

Control the number of results, filter by source category, or return structured
JSON for an API or evaluation script:

```bash
python -m scripts.search_documents \
  "How does dependency injection work?" \
  --top-k 3 \
  --category tutorial

python -m scripts.search_documents \
  "How do I test a FastAPI application?" \
  --json
```

Search must use the same `--model` and `--index-path` used for indexing. Results
include their score, page and section title, prose, code blocks, and canonical
FastAPI source URL.

Compare the retrieval modes or tune fusion behavior:

```bash
python -m scripts.search_documents "request body" --mode vector
python -m scripts.search_documents "request body" --mode keyword
python -m scripts.search_documents "request body" \
  --mode hybrid \
  --vector-weight 1.0 \
  --keyword-weight 1.0 \
  --rrf-k 60
```

Keyword retrieval gives extra weight to page titles and section headings. Hybrid
retrieval also applies a small exact title-phrase boost so focused pages such as
`Request Body` are preferred over broadly related advanced pages.

### Search API

Start the FastAPI service after creating the local vector index:

```bash
uvicorn app.api.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation, or
check service and index availability directly:

```bash
curl http://127.0.0.1:8000/health
```

Search uses hybrid retrieval by default:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I create a request body?",
    "top_k": 5,
    "category": "tutorial",
    "mode": "hybrid"
  }'
```

The response includes combined and component scores, vector and keyword ranks,
chunk text, code blocks, and canonical source URLs. Set `RAG_INDEX_PATH` or
`RAG_EMBEDDING_MODEL` to override the default index and embedding model.

Run the automated checks with:

```bash
ruff check .
pytest
```

## Target data model

| Entity | Purpose |
| --- | --- |
| `documents` | One logical FastAPI documentation page |
| `document_versions` | Content hash and metadata for each observed page version |
| `document_chunks` | Searchable text, metadata, full-text vector, and embedding |
| `questions` | Normalized user questions and their embeddings |
| `retrieval_results` | Retrieved chunks, scores, and rank for each request |
| `answers` | Generated response, citations, model, tokens, and latency |
| `cache_entries` | Reusable answers with thresholds, expiry, and validity state |
| `usage_events` | Actual cost, avoided cost, cache outcome, and timing |
| `evaluations` | Retrieval, faithfulness, citation, and user-feedback scores |

## Target API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Confirm the API and database are available |
| `POST` | `/search` | Return retrieved documentation without generation |
| `POST` | `/ask` | Retrieve evidence and return a cited answer |
| `POST` | `/feedback` | Store helpful or unhelpful user feedback |
| `POST` | `/admin/ingest` | Start an authorized documentation ingestion job |
| `GET` | `/admin/metrics` | Return evaluation and cost metrics |

## Implementation roadmap

Complete the following sections in order. A phase is complete only when its
acceptance criteria pass.

### Phase 0: Establish scope and measurements

- [ ] Limit the first release to selected pages from the official FastAPI
  tutorial, security, testing, and deployment documentation.
- [ ] Create a manifest containing each approved documentation URL and a stable
  document identifier.
- [ ] Define the initial quality targets for retrieval hit rate, citation
  correctness, cache precision, latency, and cost per uncached question.
- [ ] Define what counts as an unsupported answer and when the application must
  respond that it lacks enough evidence.
- [ ] Create at least 30 evaluation questions and record the expected source page
  or section for each one.

**Acceptance criteria:** The source manifest, evaluation dataset, and measurable
quality targets are committed and can be reviewed without running the service.

### Phase 1: Create the application skeleton

- [ ] Create `app/api`, `app/ingestion`, `app/retrieval`, `app/generation`,
  `app/cache`, `app/evaluation`, and `app/models` packages.
- [ ] Add dependency management and separate runtime, development, and test
  dependencies.
- [ ] Add typed application settings loaded from environment variables.
- [ ] Add structured logging and a request ID for every API request.
- [ ] Implement `GET /health`.
- [ ] Configure formatting, linting, static type checking, and pytest.
- [ ] Add a continuous-integration workflow that runs the same checks used
  locally.

**Acceptance criteria:** A clean environment can install the project, start the
API, receive a successful health response, and pass all automated checks.

### Phase 2: Start PostgreSQL and define the schema

- [ ] Add Docker Compose configuration for PostgreSQL with pgvector enabled.
- [ ] Configure SQLAlchemy database sessions.
- [ ] Configure Alembic migrations.
- [ ] Create tables for documents, versions, chunks, questions, retrieval
  results, answers, cache entries, usage events, and evaluations.
- [ ] Add indexes for document identity, active versions, normalized questions,
  full-text search, and vector similarity.
- [ ] Add a migration test that builds an empty database from the first migration
  through the latest migration.

**Acceptance criteria:** The database can be created from migrations, pgvector is
available, and the migration test passes on a new database.

### Phase 3: Ingest and version FastAPI documentation

- [ ] Download only URLs listed in the approved source manifest.
- [ ] Identify the crawler with a clear user agent and configure timeouts,
  retries, and polite request limits.
- [ ] Extract the page title, headings, prose, code blocks, canonical URL, and
  section anchors.
- [ ] Remove navigation, footer, search, and other repeated page elements.
- [ ] Normalize extracted content without changing the meaning of code examples.
- [ ] Calculate a SHA-256 hash for the normalized page content.
- [ ] Skip unchanged pages and create a new document version when the hash
  changes.
- [ ] Mark the prior version inactive without deleting its history.
- [ ] Record ingestion success, failure, and timestamps for every source.
- [ ] Save extraction fixtures and write parser tests for headings, prose, code,
  links, and malformed pages.

**Acceptance criteria:** Running ingestion twice creates no duplicate versions;
changing a fixture creates one new active version and preserves the old version.

### Phase 4: Chunk and embed active documents

- [ ] Split content by page and section headings before applying token limits.
- [ ] Keep each code example with the prose that explains it.
- [ ] Start with chunks of approximately 400-700 tokens and 50-100 tokens of
  overlap, then tune these values through evaluation.
- [ ] Store the page title, section title, anchor URL, document version, token
  count, and chunk order with every chunk.
- [ ] Generate embeddings in batches using the configured embedding provider.
- [ ] Store embeddings in pgvector and create PostgreSQL full-text-search values.
- [ ] Deduplicate identical chunks by content hash.
- [ ] Add a repeatable command that reprocesses only new or changed versions.

**Acceptance criteria:** Every active source is represented by searchable chunks,
unchanged chunks are not embedded twice, and all stored citations resolve to the
correct page or section.

### Phase 5: Implement and evaluate retrieval

- [ ] Implement vector similarity search filtered to active document versions.
- [ ] Implement PostgreSQL full-text search for exact technical terms, function
  names, and error messages.
- [ ] Combine keyword and vector rankings with Reciprocal Rank Fusion or another
  documented scoring method.
- [ ] Make the retrieval count and hybrid-search weights configurable.
- [ ] Add an optional reranking interface that can be disabled.
- [ ] Implement `POST /search` so retrieval can be inspected without calling an
  LLM.
- [ ] Run the evaluation dataset and calculate Hit Rate at 3, Hit Rate at 5,
  Mean Reciprocal Rank, and latency.
- [ ] Tune chunking and retrieval settings using evaluation results rather than
  individual demonstration questions.

**Acceptance criteria:** The correct source appears within the agreed top-k target
for the required percentage of the evaluation dataset, and `/search` returns
scores plus complete source metadata.

### Phase 6: Generate grounded answers with citations

- [ ] Define a provider-independent LLM interface.
- [ ] Build a prompt containing the user question, retrieved evidence, source
  identifiers, and instructions not to use unsupported information.
- [ ] Limit context by token count rather than only by chunk count.
- [ ] Require the generated answer to identify its supporting sources.
- [ ] Convert source identifiers into clickable FastAPI documentation links.
- [ ] Refuse to generate a definitive answer when retrieval confidence or
  evidence coverage is below the configured threshold.
- [ ] Implement `POST /ask`.
- [ ] Record the selected model, prompt version, input tokens, output tokens,
  latency, citations, and estimated generation cost.
- [ ] Add tests that reject missing, invalid, and unsupported citations.

**Acceptance criteria:** Evaluation answers are supported by retrieved text, every
factual answer has valid citations, and low-evidence questions fail safely.

### Phase 7: Add exact and semantic caching

- [ ] Normalize questions while preserving meaningful technical identifiers.
- [ ] Check for an exact normalized-question match before creating a query
  embedding.
- [ ] Store question embeddings for semantic-cache lookup.
- [ ] Initially require a strict semantic-similarity threshold, such as 0.95,
  and tune it with measured cache precision.
- [ ] Require matching documentation scope and version before reusing an answer.
- [ ] Confirm that every source version cited by a cached answer is still active.
- [ ] Add configurable expiry, minimum quality score, and maximum answer age.
- [ ] Store cache misses, exact hits, semantic hits, rejected matches, and the
  reason each candidate was rejected.
- [ ] Return cache status and answer creation time in API response metadata.
- [ ] Add adversarial tests for similar-looking questions that require different
  answers.

**Acceptance criteria:** Exact and safe paraphrased questions avoid a second LLM
call, while meaningfully different questions do not reuse the cached answer.

### Phase 8: Invalidate stale cached answers

- [ ] Link each cache entry to the document versions and chunks supporting it.
- [ ] When ingestion detects a changed page, mark cache entries citing its old
  version invalid.
- [ ] Preserve cache entries whose supporting documents did not change.
- [ ] Revalidate an expired entry through retrieval and generation rather than
  silently extending it.
- [ ] Add an administrative command to invalidate entries by document, version,
  model, prompt version, or date.
- [ ] Test that a changed source invalidates only the affected cached answers.

**Acceptance criteria:** No answer based on an inactive source version can be
served from the cache, and unrelated cache entries remain reusable.

### Phase 9: Add configurable cost controls

- [ ] Add `economy`, `balanced`, and `accuracy` presets.
- [ ] Make the cache threshold, cache lifetime, retrieval count, reranking,
  embedding provider, generation model, context limit, and answer limit
  configurable.
- [ ] Use local embeddings by default so repeated query lookup does not require a
  hosted embedding request.
- [ ] Route high-confidence, straightforward questions to a lower-cost model.
- [ ] Use a stronger model only for configured low-confidence or multi-source
  questions.
- [ ] Add per-request, daily, and monthly budget guards.
- [ ] Return a clear response when a budget guard prevents generation.
- [ ] Never log secrets, provider keys, or complete sensitive request headers.

**Acceptance criteria:** Each preset has version-controlled settings, budget
limits are enforced, and the chosen settings are recorded with every request.

### Phase 10: Measure quality, latency, and savings

- [ ] Store actual embedding, reranking, and generation usage separately.
- [ ] Calculate actual request cost from versioned provider pricing configured by
  the application.
- [ ] For cache hits, calculate the estimated generation cost avoided using a
  documented baseline.
- [ ] Keep actual cost and estimated avoided cost as separate metrics.
- [ ] Measure exact-cache hit rate, semantic-cache hit rate, retrieval quality,
  answer faithfulness, citation correctness, latency, and user feedback.
- [ ] Track incorrect semantic-cache matches as a primary safety metric.
- [ ] Compare economy, balanced, and accuracy presets on the same evaluation set.
- [ ] Export evaluation results in a machine-readable format.

**Acceptance criteria:** A repeatable evaluation run produces comparable quality,
latency, and cost results for all presets without presenting estimated savings as
actual money spent.

### Phase 11: Build the dashboard and user interface

- [ ] Build a chat interface showing the answer, clickable citations,
  documentation version, cache status, and answer age.
- [ ] Allow users to select economy, balanced, or accuracy mode.
- [ ] Add helpful and unhelpful feedback controls.
- [ ] Build an evaluation dashboard showing total questions, cache-hit rates,
  LLM calls avoided, actual cost, estimated avoided cost, latency, retrieval
  scores, citation results, and incorrect cache matches.
- [ ] Add filters for date, model, configuration preset, cache outcome, and
  question category.
- [ ] Add a failure-analysis view containing the question, retrieved evidence,
  answer, citations, scores, and feedback.

**Acceptance criteria:** A user can ask a question and inspect its sources, while
an evaluator can trace every displayed metric back to stored request data.

### Phase 12: Secure, test, document, and deploy

- [ ] Require authorization for ingestion, invalidation, settings, and metrics
  endpoints.
- [ ] Validate all request sizes and configuration values.
- [ ] Add unit tests for normalization, parsing, chunking, ranking, cost
  calculations, and cache decisions.
- [ ] Add integration tests for ingestion, retrieval, generation, citations,
  cache reuse, and source-aware invalidation.
- [ ] Add end-to-end tests for the primary user flows.
- [ ] Test concurrent requests for the same uncached question to prevent duplicate
  generation and cache writes.
- [ ] Add Dockerfiles and a local Docker Compose environment.
- [ ] Document local setup, configuration, database migrations, ingestion,
  evaluation, and deployment.
- [ ] Add application health, readiness, structured logs, and basic metrics.
- [ ] Deploy a reproducible demonstration environment.

**Acceptance criteria:** A new contributor can follow the documentation to start
the application, ingest the approved FastAPI sources, run the evaluation suite,
and reproduce the deployed behavior.

## Planned configuration

Configuration names may change during implementation, but the project should
support the following categories without hard-coding provider credentials:

```text
DATABASE_URL
DOCUMENTATION_MANIFEST_PATH
EMBEDDING_PROVIDER
EMBEDDING_MODEL
GENERATION_PROVIDER
GENERATION_MODEL
RERANKER_PROVIDER
RERANKER_MODEL
RETRIEVAL_TOP_K
HYBRID_VECTOR_WEIGHT
HYBRID_KEYWORD_WEIGHT
SEMANTIC_CACHE_THRESHOLD
CACHE_TTL_SECONDS
MAX_CONTEXT_TOKENS
MAX_ANSWER_TOKENS
DAILY_BUDGET_USD
MONTHLY_BUDGET_USD
```

Secrets will be supplied through environment variables or a secret manager and
will never be committed to the repository.

## Testing strategy

Tests will be organized into four layers:

1. **Unit tests:** Deterministic functions such as hashing, question
   normalization, chunking, ranking, cost calculations, and cache eligibility.
2. **Integration tests:** PostgreSQL, pgvector, ingestion, retrieval, cache
   invalidation, and provider adapters using controlled fixtures or fakes.
3. **RAG evaluation:** A versioned question set measuring retrieval, grounding,
   citation correctness, semantic-cache precision, latency, and cost.
4. **End-to-end tests:** Complete uncached, exact-cache, semantic-cache,
   low-evidence, stale-document, and budget-exceeded request flows.

External LLM calls should be replaced with deterministic fakes in the normal
test suite. A separate, explicitly invoked evaluation can use real providers.

## Definition of done

The first release is complete when:

- [ ] Official FastAPI documentation can be ingested and versioned repeatedly.
- [ ] Hybrid retrieval meets the documented evaluation target.
- [ ] Answers are grounded and include valid, clickable citations.
- [ ] Low-evidence questions fail safely.
- [ ] Exact and semantic caching reduce duplicate generation calls.
- [ ] Documentation changes invalidate only affected cache entries.
- [ ] Economy, balanced, and accuracy presets are available.
- [ ] The dashboard reports quality, latency, actual cost, and estimated avoided
  cost.
- [ ] Automated tests and continuous integration pass.
- [ ] A new contributor can run the complete system from the README.

## Future extensions

- Add more documentation collections through separate source manifests.
- Add document-level access control for private company documentation.
- Add conversational follow-up questions with bounded history summarization.
- Add multilingual retrieval and generation.
- Compare local and hosted embedding, reranking, and generation models.
- Deploy the service to Kubernetes after the single-service version is stable.
