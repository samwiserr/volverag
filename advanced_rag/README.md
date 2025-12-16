# VolveRAG

A state-of-the-art Retrieval-Augmented Generation (RAG) system for querying Volve petrophysical reports using natural language. Built with LangGraph, OpenAI GPT-4o, and advanced retrieval techniques.

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

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- Volve dataset (petrophysical reports)

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/volve-rag.git
cd volve-rag/advanced_rag
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**

Create a `.env` file in the `advanced_rag/` directory:
```bash
OPENAI_API_KEY=sk-your-api-key-here
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Or export them:
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

4. **Prepare your documents:**

Place the Volve petrophysical reports in a directory (e.g., `../spwla_volve-main/`). The system will automatically discover and process:
- PDF files (petrophysical reports)
- DOC/DOCX files (LFP reports)
- DAT files (Well_picks_Volve_v1.dat)

## 🚀 Quick Start

### 1. Build the Index

First, process all documents and build the vector store:

```bash
python -m src.main --build-index --documents-path ../spwla_volve-main
```

This will:
- Extract text from all PDFs, DOCX, and DOC files
- Parse structured data (well picks, petrophysical parameters, evaluation parameters)
- Create embeddings and build the vector store
- Generate caches for fast deterministic lookups

**Note:** First-time indexing may take 10-30 minutes depending on document count.

### 2. Run the Web UI

```bash
streamlit run web_app.py
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
         │  LLM Rerank (gpt-4o)  │
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
- **OpenAI GPT-4o**: LLM for query understanding and answer generation
- **OpenAI Embeddings**: text-embedding-3-small for semantic search
- **BM25**: Keyword-based retrieval
- **Sentence Transformers**: Cross-encoder reranking
- **RapidFuzz**: Fuzzy matching for entity resolution

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | ✅ Yes | - | Your OpenAI API key |
| `OPENAI_MODEL` | No | `gpt-4o` | Chat model for LLM calls |
| `OPENAI_EMBEDDING_MODEL` | No | `text-embedding-3-small` | Embedding model |
| `RAG_USE_CROSS_ENCODER` | No | `true` | Enable cross-encoder reranking |
| `RAG_ENABLE_QUERY_COMPLETION` | No | `true` | Enable incomplete query handling |
| `RAG_ENABLE_QUERY_DECOMPOSITION` | No | `true` | Enable query decomposition |
| `RAG_ENABLE_MONITORING` | No | `true` | Enable performance monitoring |

See `.env.example` for a complete list.

## 📁 Project Structure

```
advanced_rag/
├── src/
│   ├── graph/              # LangGraph workflow
│   │   ├── nodes.py        # Core workflow nodes
│   │   └── rag_graph.py    # Graph definition
│   ├── tools/              # Specialized lookup tools
│   │   ├── well_picks_tool.py
│   │   ├── petro_params_tool.py
│   │   ├── eval_params_tool.py
│   │   └── retriever_tool.py
│   ├── normalize/          # Query normalization
│   ├── query/              # Query processing
│   ├── loaders/            # Document loaders
│   ├── processors/         # Text processing
│   ├── evaluation/         # Evaluation framework
│   └── monitoring/         # Performance monitoring
├── data/
│   ├── vectorstore/        # ChromaDB storage
│   └── indices/            # Cached indices
├── web_app.py              # Streamlit UI
├── requirements.txt        # Dependencies
└── README.md              # This file
```

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
- Uses [OpenAI](https://openai.com/) for embeddings and LLM

## 📧 Contact

For questions or issues, please open an issue on GitHub.

## 🔗 Links

- [Setup Guide](SETUP.md) - Detailed installation instructions
- [Enhancement Plan](ENHANCEMENT_PLAN.md) - Roadmap and features
- [Verification](VERIFICATION.md) - System verification report
