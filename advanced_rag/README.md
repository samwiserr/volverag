# VolveRAG

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/streamlit-ready-orange.svg)
![CI](https://img.shields.io/badge/tests-passing-brightgreen.svg)

VolveRAG is a Streamlit-ready RAG application for querying Volve petrophysical reports with natural language. It uses LangGraph for orchestration, Groq for LLM calls, local Hugging Face embeddings for vector search, and deterministic structured tools for exact petrophysical values.

## 🚀 Features

### Core Capabilities
- **Natural Language Querying**: Ask questions about wells, formations, petrophysical parameters, and more
- **Deterministic Structured Lookups**: Direct access to parsed tables and data (100% accurate for structured queries)
- **Hybrid Retrieval**: Combines semantic search (vector) and keyword search (BM25) for better results
- **Cross-Encoder Reranking**: Advanced reranking for improved relevance
- **Query Completion**: Handles incomplete queries intelligently
- **Query Decomposition**: Breaks complex queries into simpler sub-queries
- **Document-Level Retrieval**: Retrieves all chunks from relevant documents for comprehensive context

### Specialized Tools
- **Well Picks Tool**: Direct lookup of formation depths (MD, TVD, TVDSS)
- **Petrophysical Parameters Tool**: Exact values for Net/Gross, PHIF, SW, KLOGH (Klinkenberg-corrected horizontal permeability)
- **Evaluation Parameters Tool**: Archie parameters, matrix/fluid density, GR min/max
- **Structured Facts Tool**: General numeric facts from narrative text
- **Section Lookup Tool**: Direct access to document sections
- **Formation Properties Tool**: One-shot queries for formations and their properties

### Advanced Features
- **Stateful Chat**: Maintains conversation context across turns
- **Entity Disambiguation**: Handles typos and ambiguous queries
- **Source Citations**: Every answer includes exact page numbers
- **PDF Viewer**: Click sources to view exact pages in-app
- **Performance Monitoring**: Built-in metrics and evaluation framework
- **Incomplete Query Handling**: Automatically completes partial queries
- **Streamlit Release Assets**: Downloads a prebuilt vectorstore and PDF bundle on first startup

## ⚡ Quick Start (Golden Path)

**The supported way to run VolveRAG:**

```bash
git clone https://github.com/samwiserr/volverag.git
cd volverag/advanced_rag

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add GROQ_API_KEY

# Place it at: ../spwla_volve-main/ (or configure your path)
# See DATA_POLICY.md for details

# Build a Streamlit-ready local index
python scripts/build_sota.py --documents-path ../spwla_volve-main --no-contextual --no-raptor

streamlit run web_app/app.py
```

`--no-contextual --no-raptor` is the Groq-free-tier friendly build path. Contextual chunking and RAPTOR are implemented, but full builds can hit Groq request/token limits unless the account has higher throughput.

## 📋 Prerequisites

- Python 3.10+
- Groq API key
- Volve dataset (petrophysical reports) - **download separately, do not commit to repo**
- (Optional) antiword for `.doc` file support - see [EXTERNAL_TOOLS.md](../EXTERNAL_TOOLS.md)

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/samwiserr/volverag.git
cd volverag/advanced_rag
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**

Create a `.env` file in the `advanced_rag/` directory:
```bash
GROQ_API_KEY=gsk-your-api-key-here
LLM_PROVIDER=groq
EMBEDDING_PROVIDER=huggingface
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_FAST_MODEL=llama-3.1-8b-instant
LOCAL_EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5
```

Or export them:
```bash
export GROQ_API_KEY="gsk-your-api-key-here"
export LLM_PROVIDER="groq"
export EMBEDDING_PROVIDER="huggingface"
```

4. **Download and prepare the Volve dataset:**

**Important**: The Volve dataset should NOT be in the repository. Download it separately from the official Equinor source and place it outside the repository (e.g., `../spwla_volve-main/`).

The system will automatically discover and process:
- PDF files (petrophysical reports)
- DOC/DOCX files (LFP reports)
- DAT files (Well_picks_Volve_v1.dat)

See [DATA_POLICY.md](../DATA_POLICY.md) for details on why data files are excluded from the repository.

## 🚀 Quick Start

### 1. Build the Index

First, process all documents and build the vector store:

```bash
python scripts/build_sota.py --documents-path ../spwla_volve-main --no-contextual --no-raptor
```

This will:
- Extract text from all PDFs, DOCX, and DOC files
- Parse structured data (well picks, petrophysical parameters, evaluation parameters)
- Create embeddings and build the vector store
- Generate caches for fast deterministic lookups

**Note:** The validated local build for the full Volve folder processed 511 documents into 2,606 chunks and completed in about 9 minutes without contextual chunking/RAPTOR.

### 2. Run the Web UI

```bash
streamlit run web_app/app.py
```

Or use the provided scripts:
```bash
# Windows
run_web_app.bat

# Linux/Mac
chmod +x run_web_app.sh
./run_web_app.sh
```

The app will open at `http://localhost:8501`

## Streamlit Cloud Deployment

Streamlit Community Cloud should not build the vectorstore at runtime. Use the prebuilt GitHub Release assets:

```toml
VECTORSTORE_URL = "https://github.com/samwiserr/volverag/releases/download/v2.0.1-sota/vectorstore.zip"
PDFS_URL = "https://github.com/samwiserr/volverag/releases/download/v2.0.1-sota/pdfs.zip"
GROQ_API_KEY = "gsk_..."
LLM_PROVIDER = "groq"
EMBEDDING_PROVIDER = "huggingface"
LOCAL_EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
```

The `vectorstore.zip` asset contains:

- Chroma vectorstore
- BM25 lexical store
- `petro_params_cache.json` with 64 rows
- `well_picks_cache.json` with 409 rows
- `eval_params_cache.json` with 16 tables
- `facts_cache.json` with 293 rows
- `section_index.json`

The `pdfs.zip` asset contains uniquely named PDFs used by the in-app PDF viewer.

### 3. Query via CLI

```bash
python -m src.main --query "What formations are in well 15/9-F-5?"
python -m src.main --query "What is the porosity for Hugin formation in 15/9-F-5?"
python -m src.main --query "What is the depth of Sleipner formation in 15/9-19A?"
```

### 4. Interactive Chat Mode

```bash
python -m src.main --chat
```

## 📖 Example Queries

### Formation Queries
- "What formations are in well 15/9-F-5?"
- "List all formations in 15/9-F-15 A"
- "all formations and their properties"

### Depth Queries
- "What is the depth of Sleipner formation in 15/9-19A?"
- "TVDSS for Hugin in 15/9-F-5"

### Petrophysical Parameters
- "What is the porosity for Hugin in 15/9-F-5?"
- "What is the water saturation value of Hugin formation in 15/9-F-5?"
- "What is the permeability for Sleipner in 15/9-F-5?"
- "What is KLOGH for Hugin in 15/9-F-5?" (KLOGH = Klinkenberg-corrected horizontal permeability)

### Evaluation Parameters
- "What is the Archie n for Hugin in 15/9-F-5?"
- "What is the matrix density for Hugin in 15/9-F-5?"
- "What is the fluid density for Hugin in 15/9-F-5?"

### Comprehensive Queries
- "list all well formations and their properties"
- "every formations and their properties"

## 🔍 Retrieval Strategy

VolveRAG uses a **dual retrieval approach** for optimal accuracy:

### 1. **Deterministic Fact Retrieval** (Structured Tools)
For precise numeric queries, the system uses direct lookups from parsed tables:
- **Well Picks Tool**: Formation depths (MD, TVD, TVDSS)
- **Petrophysical Parameters Tool**: Net/Gross, PHIF, SW, KLOGH (Klinkenberg-corrected horizontal permeability)
- **Evaluation Parameters Tool**: Archie parameters, matrix/fluid density, GR min/max
- **Structured Facts Tool**: General numeric facts from notes and narrative text

**When used**: Queries asking for specific values (e.g., "What is the porosity for Hugin in 15/9-F-5?")

**Benefits**: 100% accurate, no LLM interpretation needed, deterministic results

### 2. **Narrative RAG** (Hybrid Retrieval)
For exploratory or contextual queries:
- Semantic search (vector embeddings via ChromaDB)
- Keyword search (BM25)
- Hybrid fusion with Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- LLM-based reranking and answer synthesis with document context

**When used**: Queries asking for explanations, summaries, or complex analysis (e.g., "What happened to wellbore 15/9-F-15 C?")

**Benefits**: Handles natural language, provides context, explains relationships

The system automatically routes queries to the appropriate retrieval method based on query intent.

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                    User Query                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │  Query Normalization  │
         │  (Well/Formation/Prop) │
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │  Deterministic Routing │
         │  (Structured Tools)    │
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │  Hybrid Retrieval      │
         │  (Vector + BM25)       │
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │  Cross-Encoder Rerank  │
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │  LLM Rerank (Groq)    │
         └───────────┬────────────┘
                     │
         ┌───────────▼────────────┐
         │  Answer Generation     │
         │  (with citations)       │
         └────────────────────────┘
```

### Key Technologies

- **LangGraph**: Agentic workflow orchestration
- **ChromaDB**: Vector database with HNSW indexing
- **Groq**: LLM provider for routing, query expansion, reranking, and answer generation
- **Hugging Face / Sentence Transformers**: local embedding model (`nomic-ai/nomic-embed-text-v1.5`)
- **BM25**: Keyword-based retrieval
- **Sentence Transformers**: Cross-encoder reranking
- **RapidFuzz**: Fuzzy matching for entity resolution

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | ✅ Yes | - | Groq API key for LLM calls |
| `LLM_PROVIDER` | No | `groq` | LLM provider |
| `EMBEDDING_PROVIDER` | No | `huggingface` | Embedding provider |
| `GROQ_MODEL` | No | `llama-3.3-70b-versatile` | Main chat model |
| `GROQ_FAST_MODEL` | No | `llama-3.1-8b-instant` | Fast model for routing/expansion |
| `LOCAL_EMBEDDING_MODEL` | No | `nomic-ai/nomic-embed-text-v1.5` | Local embedding model |
| `VECTORSTORE_URL` | Streamlit | - | GitHub Release URL for `vectorstore.zip` |
| `PDFS_URL` | Streamlit | - | GitHub Release URL for `pdfs.zip` |
| `RAG_USE_CROSS_ENCODER` | No | `true` | Enable cross-encoder reranking |
| `RAG_HYBRID_FUSION` | No | `rrf` | Hybrid fusion method |
| `RAG_RRF_K` | No | `60` | RRF constant \(k\) for rank fusion (higher = less rank impact) |
| `RAG_HYDE` | No | `true` | Enable Hypothetical Document Embeddings at query time |
| `RAG_ENABLE_QUERY_COMPLETION` | No | `true` | Enable incomplete query handling |
| `RAG_ENABLE_QUERY_DECOMPOSITION` | No | `true` | Enable query decomposition |
| `RAG_ENABLE_MONITORING` | No | `true` | Enable performance monitoring |

See `.env.example` for a complete list.

## 📁 Project Structure

```
VolveRAG/
├── advanced_rag/          # ✅ CURRENT SYSTEM (use this)
│   ├── src/
│   │   ├── graph/              # LangGraph workflow
│   │   │   ├── nodes.py        # Core workflow nodes
│   │   │   └── rag_graph.py    # Graph definition
│   │   ├── tools/              # Specialized lookup tools
│   │   │   ├── well_picks_tool.py
│   │   │   ├── petro_params_tool.py
│   │   │   ├── eval_params_tool.py
│   │   │   └── retriever_tool.py
│   │   ├── normalize/          # Query normalization
│   │   ├── query/              # Query processing
│   │   ├── loaders/            # Document loaders
│   │   ├── processors/         # Text processing
│   │   ├── evaluation/         # Evaluation framework
│   │   └── monitoring/         # Performance monitoring
│   ├── data/                    # Generated (not in repo)
│   │   ├── vectorstore/        # ChromaDB storage
│   │   └── indices/            # Cached indices
│   ├── web_app/                # Streamlit UI
│   ├── requirements.txt        # Dependencies
│   └── README.md              # Main documentation
├── src/                  # ⚠️ LEGACY (deprecated, may be removed)
├── spwla_volve-main/     # ❌ NOT IN REPO (data - download separately)
├── DATA_POLICY.md        # Data handling policy
├── EXTERNAL_TOOLS.md     # External dependencies
├── LICENSE               # MIT License
└── .gitignore           # Excludes data files
```

**Important Notes**:
- ✅ **Use `advanced_rag/`** for all new development and usage
- ⚠️ **`src/` is legacy** and may be removed in future versions
- ❌ **`spwla_volve-main/` should NOT be in the repository** - download separately
- 📁 **`data/` directories** are generated during indexing and excluded from Git

## 🔍 How It Works

### 1. Document Processing
- Extracts text from PDFs, DOCX, DOC files
- Parses structured tables (petrophysical parameters, evaluation parameters)
- Creates intelligent chunks with semantic boundaries
- Generates embeddings and stores in ChromaDB

### 2. Query Processing
- **Normalization**: Extracts well, formation, property from query
- **Routing**: Determines if query needs structured lookup or RAG retrieval
- **Retrieval**: Hybrid search (semantic + keyword) finds relevant documents
- **Reranking**: Cross-encoder and LLM reranking improve relevance
- **Answer Generation**: LLM synthesizes answer from retrieved context

### 3. Deterministic Tools
For structured queries (depths, parameters), the system uses direct lookups:
- Parses data files during indexing
- Stores in JSON caches
- Returns exact values (no LLM interpretation)

## 🧪 Evaluation

Run the evaluation framework:

```bash
python scripts/run_evaluation.py --baseline
```

Compare with baseline:

```bash
python scripts/run_evaluation.py --compare
```

## 📊 Performance Monitoring

View metrics dashboard:

```bash
streamlit run src/monitoring/dashboard.py
```

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 License

See [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- Volve dataset provided by Equinor
- Built with [LangChain](https://www.langchain.com/) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Uses [Groq](https://groq.com/) for LLM calls and Hugging Face-compatible local embeddings

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🧪 Testing

VolveRAG includes comprehensive testing:

### Running Tests

```bash
# All tests
pytest

# Unit tests only (fast)
pytest tests/unit -m unit

# Integration tests
pytest tests/integration -m integration

# Property-based tests (Hypothesis)
pytest tests/property -m property

# Performance tests
pytest tests/performance -m performance

# With coverage
pytest --cov=src --cov-report=html
```

### Test Markers

- `@pytest.mark.unit`: Fast unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.property`: Property-based tests
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.requires_api`: Needs API keys
- `@pytest.mark.requires_data`: Needs dataset
- `@pytest.mark.requires_vectorstore`: Needs built vectorstore
- `@pytest.mark.slow`: Slow-running tests

## 📚 Documentation

- **[Architecture Guide](docs/ARCHITECTURE.md)** - System architecture and design
- **[Migration Guide](docs/MIGRATION.md)** - Migrating to new architecture
- **[Setup Guide](SETUP.md)** - Detailed installation instructions
- **[Enhancement Plan](ENHANCEMENT_PLAN.md)** - Roadmap and features
- **[Verification](VERIFICATION.md)** - System verification report

## 🔗 Links

- [Setup Guide](SETUP.md) - Detailed installation instructions
- [Architecture Guide](docs/ARCHITECTURE.md) - System architecture
- [Migration Guide](docs/MIGRATION.md) - Migration from old code
- [Enhancement Plan](ENHANCEMENT_PLAN.md) - Roadmap and features
- [Verification](VERIFICATION.md) - System verification report
