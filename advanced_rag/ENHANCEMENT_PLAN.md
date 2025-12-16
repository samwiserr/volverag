# Enhance RAG System with Advanced Features

## Overview

This plan upgrades all LLM models to `gpt-4o` and implements 5 advanced enhancements while ensuring zero regression through comprehensive testing.

## Phase 1: Update All LLM Models to gpt-4o

### Files to Modify

1. **[advanced_rag/src/tools/retriever_tool.py](advanced_rag/src/tools/retriever_tool.py)**
   - Line 82: Change default `RAG_RERANK_MODEL` from `"gpt-4o-mini"` to `"gpt-4o"`

2. **[advanced_rag/src/graph/nodes.py](advanced_rag/src/graph/nodes.py)**
   - Line 30: Change default `OPENAI_MODEL` from `"gpt-4o"` (already correct, but verify)
   - Line 38: Change `OPENAI_GRADE_MODEL` default to `"gpt-4o"` (currently falls back to OPENAI_MODEL)

3. **[advanced_rag/src/normalize/entity_resolver.py](advanced_rag/src/normalize/entity_resolver.py)**
   - Line 270: Change default from `"gpt-4o-mini"` to `"gpt-4o"`

4. **[advanced_rag/src/normalize/agent_disambiguator.py](advanced_rag/src/normalize/agent_disambiguator.py)**
   - Line 35: Change default from `"gpt-4o-mini"` to `"gpt-4o"`

### Verification
- Run existing queries to ensure responses are still accurate
- Check that all LLM calls use gpt-4o via environment variable inspection

---

## Phase 1.5: Incomplete Query Handling & Enhanced Evidence Display

### Problem
Handle incomplete queries like "Wellbore 15/9-F-15 C was sidetracked..." by:
1. Detecting and completing incomplete statements
2. Understanding narrative queries
3. Showing exact page evidence where answers were extracted

### Implementation

**New File: [advanced_rag/src/query/incomplete_query_handler.py](advanced_rag/src/query/incomplete_query_handler.py)**
- Detect incomplete queries (ending with "...", "was", "is", etc.)
- Use gpt-4o to complete/expand the query into answerable questions
- Convert narrative statements into questions
- Example: "Wellbore 15/9-F-15 C was sidetracked..." → "What happened to wellbore 15/9-F-15 C? Was it sidetracked? What are the details?"

**New File: [advanced_rag/src/query/query_completer.py](advanced_rag/src/query/query_completer.py)**
- Implement `complete_incomplete_query(query: str) -> List[str]` function
- Detect patterns: "...", "was", "is", "has", "did", "will" at end
- Detect patterns: "Wellbore X was", "Well X was" without completion
- Use gpt-4o to generate multiple question variations for better retrieval

**Modify: [advanced_rag/src/graph/nodes.py](advanced_rag/src/graph/nodes.py)**
- Add `detect_and_complete_query()` function before `generate_query_or_respond`
- For incomplete queries:
  1. Use LLM to expand into complete questions
  2. Retrieve documents for expanded queries
  3. Extract answer with source pages
  4. Format response showing what was completed
- Enhance `generate_answer()` to always extract page numbers from retrieved documents
- Ensure all ToolMessage content includes page metadata
- Format citations as: "Source: [file path] (page X)" or "Source: [file path] (pages X-Y)"

**Modify: [advanced_rag/src/tools/retriever_tool.py](advanced_rag/src/tools/retriever_tool.py)**
- Ensure `retrieve()` returns documents with page metadata preserved
- Add `_extract_page_from_document()` helper to extract page numbers from document metadata
- Enhance document metadata extraction to preserve page_start and page_end

**Modify: [advanced_rag/web_app.py](advanced_rag/web_app.py)**
- Enhance `_parse_citations()` to handle single page citations: "Source: path (page X)"
- Improve PDF viewer to show exact page where answer was found
- Add visual indicator (if possible) showing which part of the page contains the answer
- Display page preview image with answer context

### Detection Patterns
- Ends with "..." (ellipsis)
- Ends with incomplete verb: "was", "is", "has", "did", "will"
- Ends with "and", "or", "but"
- Contains "Wellbore X was" or "Well X was" without completion
- Pattern: `r".*\b(was|is|has|did|will|were|are|have)\s*\.\.\.?$"`

### Example Flow
```
User: "Wellbore 15/9-F-15 C was sidetracked..."
  ↓
System detects incomplete query
  ↓
LLM expands to: "What happened to wellbore 15/9-F-15 C? Was it sidetracked? What are the details about the sidetrack?"
  ↓
Retrieve documents for expanded queries
  ↓
Extract answer: "Wellbore 15/9-F-15 C was sidetracked from the main wellbore at depth X to Y..."
  ↓
Show answer with: "Source: [document.pdf] (page 12)"
  ↓
UI displays PDF page 12 with the relevant section visible
```

### Configuration
- Add environment variable `RAG_ENABLE_QUERY_COMPLETION=true` (default: `true`)
- Add environment variable `RAG_QUERY_COMPLETION_MODEL=gpt-4o`

### Verification
- Test with incomplete queries: "Wellbore X was...", "Formation Y is...", etc.
- Verify page numbers are always included in citations
- Verify PDF viewer opens to correct page
- Test that completed queries retrieve better results than incomplete ones
- Ensure no false positives (complete queries shouldn't be modified)

---

## Phase 2: Cross-Encoder Reranking

### Implementation

**New File: [advanced_rag/src/tools/cross_encoder_reranker.py](advanced_rag/src/tools/cross_encoder_reranker.py)**
- Implement cross-encoder reranking using `sentence-transformers` library
- Use model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (lightweight, fast)
- Alternative: `cross-encoder/ms-marco-MiniLM-L-12-v2` (more accurate, slower)
- Add caching for query-document pairs to avoid redundant computations
- Integrate as optional enhancement (can be enabled/disabled via env var)

**Modify: [advanced_rag/src/tools/retriever_tool.py](advanced_rag/src/tools/retriever_tool.py)**
- Add `_cross_encoder_rerank()` method
- Update `_llm_rerank()` to optionally use cross-encoder first, then LLM rerank
- Add environment variable `RAG_USE_CROSS_ENCODER` (default: `true`)
- Add environment variable `RAG_CROSS_ENCODER_MODEL` (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)

**Architecture:**
```
Hybrid Retrieval → MMR Diversification → Cross-Encoder Rerank (top 24) → LLM Rerank (top 12) → Final Results
```

### Dependencies
- Add to `requirements.txt`: `sentence-transformers>=2.2.0`

### Verification
- Compare reranking quality: run same queries with/without cross-encoder
- Measure latency impact (should be <100ms for 24 docs)
- Verify no regression in answer quality

---

## Phase 3: Query Decomposition/Rewriting

### Implementation

**New File: [advanced_rag/src/query/query_decomposer.py](advanced_rag/src/query/query_decomposer.py)**
- Implement LLM-based query decomposition for complex multi-part queries
- Decompose queries like "What is the porosity and permeability for Hugin in 15/9-F-5?" into:
  - "What is the porosity for Hugin formation in well 15/9-F-5?"
  - "What is the permeability for Hugin formation in well 15/9-F-5?"
- Use gpt-4o for decomposition
- Add query rewriting for better retrieval (synonym expansion, domain term enhancement)

**Modify: [advanced_rag/src/graph/nodes.py](advanced_rag/src/graph/nodes.py)**
- Add `decompose_query()` function before `generate_query_or_respond`
- For complex queries (detected by LLM), decompose into sub-queries
- Retrieve for each sub-query, then synthesize final answer
- Integrate with existing `rewrite_question` node and Phase 1.5 query completion

**Modify: [advanced_rag/src/tools/retriever_tool.py](advanced_rag/src/tools/retriever_tool.py)**
- Enhance `_expand_query()` with LLM-based synonym expansion
- Add domain-specific term expansion (e.g., "poro" → "porosity", "perm" → "permeability")

### Configuration
- Add environment variable `RAG_ENABLE_QUERY_DECOMPOSITION` (default: `true`)
- Add environment variable `RAG_DECOMPOSITION_MODEL` (default: `gpt-4o`)

### Verification
- Test with complex multi-part queries
- Verify decomposed queries retrieve better results
- Ensure no performance degradation for simple queries

---

## Phase 4: Evaluation Framework

### Implementation

**New File: [advanced_rag/src/evaluation/evaluator.py](advanced_rag/src/evaluation/evaluator.py)**
- Implement evaluation metrics:
  - **Precision@K**: Fraction of retrieved docs that are relevant
  - **Recall@K**: Fraction of relevant docs that were retrieved
  - **MRR (Mean Reciprocal Rank)**: Average of 1/rank of first relevant doc
  - **NDCG@K**: Normalized Discounted Cumulative Gain
- Support ground truth dataset format (JSON with query, expected answers, relevant doc IDs)

**New File: [advanced_rag/src/evaluation/test_suite.py](advanced_rag/src/evaluation/test_suite.py)**
- Create test suite with sample queries and expected results
- Include queries for:
  - Well-specific queries (e.g., "formations in 15/9-F-5")
  - Formation-specific queries (e.g., "porosity for Hugin")
  - Parameter queries (e.g., "fluid density for Hugin in 15/9-F-5")
  - Complex queries (e.g., "all formations and their properties")
  - Incomplete queries (e.g., "Wellbore X was...")

**New File: [advanced_rag/src/evaluation/benchmark.py](advanced_rag/src/evaluation/benchmark.py)**
- Run evaluation suite and generate reports
- Compare metrics before/after enhancements
- Export results to JSON/CSV for tracking

**New Script: [advanced_rag/scripts/run_evaluation.py](advanced_rag/scripts/run_evaluation.py)**
- CLI script to run evaluation: `python scripts/run_evaluation.py`
- Options: `--baseline`, `--compare`, `--export`

### Test Data
- Create `advanced_rag/data/evaluation/test_queries.json` with ground truth
- Include 20-30 representative queries covering all query types

### Verification
- Run evaluation suite before and after changes
- Ensure metrics don't degrade
- Document baseline metrics for future comparison

---

## Phase 5: Performance Monitoring

### Implementation

**New File: [advanced_rag/src/monitoring/performance_monitor.py](advanced_rag/src/monitoring/performance_monitor.py)**
- Track metrics:
  - Query latency (retrieval time, LLM time, total time)
  - Token usage (input tokens, output tokens, cost estimation)
  - Cache hit rates (embedding cache, lexical cache)
  - Retrieval quality (number of relevant docs retrieved)
- Log to structured format (JSON lines)
- Support real-time metrics export

**New File: [advanced_rag/src/monitoring/metrics_collector.py](advanced_rag/src/monitoring/metrics_collector.py)**
- Decorator-based metrics collection
- Automatic instrumentation of key functions:
  - `RetrieverTool.retrieve()`
  - `generate_answer()`
  - `grade_documents()`
  - Tool invocations
  - Query completion (Phase 1.5)

**Modify: [advanced_rag/src/tools/retriever_tool.py](advanced_rag/src/tools/retriever_tool.py)**
- Add timing decorators to `retrieve()`, `_hybrid_retrieve()`, `_llm_rerank()`
- Log metrics after each retrieval

**Modify: [advanced_rag/src/graph/nodes.py](advanced_rag/src/graph/nodes.py)**
- Add timing to `generate_answer()`, `grade_documents()`, `rewrite_question()`
- Track token usage from LLM responses
- Track query completion time (Phase 1.5)

**New File: [advanced_rag/src/monitoring/dashboard.py](advanced_rag/src/monitoring/dashboard.py)**
- Simple Streamlit dashboard for metrics visualization
- Show: latency trends, token usage, cache hit rates, query success rates
- Optional: Run via `streamlit run src/monitoring/dashboard.py`

**New File: [advanced_rag/data/monitoring/metrics.jsonl](advanced_rag/data/monitoring/metrics.jsonl)**
- Persistent metrics storage (JSON lines format)
- Rotate logs (keep last 30 days)

### Configuration
- Add environment variable `RAG_ENABLE_MONITORING` (default: `true`)
- Add environment variable `RAG_METRICS_LOG_PATH` (default: `./data/monitoring/metrics.jsonl`)

### Verification
- Verify metrics are collected without impacting performance (<5ms overhead)
- Test dashboard loads and displays data correctly
- Ensure metrics logging doesn't cause disk space issues

---

## Phase 6: Comprehensive Verification

### Test Plan

1. **Functional Tests**
   - Run existing test queries and verify answers are identical or better
   - Test all tool invocations (well picks, petro params, eval params, etc.)
   - Verify stateful chat still works
   - Test incomplete query handling (Phase 1.5)

2. **Performance Tests**
   - Measure query latency before/after (should be <2x slower)
   - Check memory usage (should not increase significantly)
   - Verify caching works correctly

3. **Quality Tests**
   - Run evaluation suite and compare metrics
   - Manual review of 10-15 complex queries
   - Verify no hallucinations or incorrect answers
   - Test incomplete queries complete correctly

4. **Integration Tests**
   - Test CLI: `python -m src.main --query "test query"`
   - Test Streamlit UI: `streamlit run web_app.py`
   - Verify all environment variables work correctly
   - Test PDF viewer opens to correct pages (Phase 1.5)

### Rollback Plan
- Keep git commits atomic per phase
- Document baseline metrics before changes
- Ability to disable new features via environment variables

---

## Dependencies

Add to `requirements.txt`:
```
sentence-transformers>=2.2.0
numpy>=1.24.0  # For cross-encoder
```

---

## Environment Variables Summary

New variables to document:
- `RAG_ENABLE_QUERY_COMPLETION=true` (enable incomplete query handling)
- `RAG_QUERY_COMPLETION_MODEL=gpt-4o`
- `RAG_USE_CROSS_ENCODER=true` (enable cross-encoder reranking)
- `RAG_CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2`
- `RAG_ENABLE_QUERY_DECOMPOSITION=true`
- `RAG_DECOMPOSITION_MODEL=gpt-4o`
- `RAG_ENABLE_MONITORING=true`
- `RAG_METRICS_LOG_PATH=./data/monitoring/metrics.jsonl`

Updated variables:
- `RAG_RERANK_MODEL=gpt-4o` (was gpt-4o-mini)
- `OPENAI_MODEL=gpt-4o` (verify)
- `OPENAI_GRADE_MODEL=gpt-4o` (was gpt-4o-mini)
- `RAG_ENTITY_RESOLVER_MODEL=gpt-4o` (was gpt-4o-mini)
- `RAG_AGENT_DISAMBIGUATE_MODEL=gpt-4o` (was gpt-4o-mini)

---

## Success Criteria

1. All LLM calls use gpt-4o
2. Incomplete queries are detected and completed correctly (Phase 1.5)
3. All answers show page numbers in citations (Phase 1.5)
4. Cross-encoder reranking improves retrieval quality (MRR improvement >5%)
5. Query decomposition handles complex queries better
6. Evaluation framework provides measurable metrics
7. Performance monitoring shows system health
8. Zero regression: all existing queries work correctly
9. Performance impact <2x latency increase

---

## Implementation Order

1. Phase 1: Update models to gpt-4o (low risk, quick)
2. Phase 6: Run baseline verification
3. **Phase 1.5: Incomplete Query Handling** (adds immediate value, builds on Phase 1)
4. Phase 2: Cross-encoder reranking (isolated, testable)
5. Phase 3: Query decomposition (builds on existing rewrite and Phase 1.5)
6. Phase 4: Evaluation framework (enables measurement)
7. Phase 5: Performance monitoring (low risk, high value)
8. Phase 6: Final verification

Each phase should be committed separately with verification before proceeding.

---

## Todos

- [ ] Update all LLM model defaults to gpt-4o
- [ ] Run baseline tests to establish current system performance
- [ ] Implement incomplete query detection and completion module
- [ ] Integrate query completion into nodes.py workflow
- [ ] Enhance source extraction to always include page numbers
- [ ] Improve PDF viewer to show exact page with answer highlighted
- [ ] Implement cross-encoder reranking module
- [ ] Integrate cross-encoder reranking into retrieval pipeline
- [ ] Implement query decomposition/rewriting module
- [ ] Integrate query decomposition into nodes.py workflow
- [ ] Implement evaluation framework with metrics
- [ ] Create test suite with ground truth queries
- [ ] Implement performance monitoring
- [ ] Create metrics dashboard
- [ ] Run comprehensive verification tests

