# RAG System for Volve Wells

A Retrieval-Augmented Generation (RAG) system for querying and searching well log data from the Volve field dataset. This system enables natural language querying and semantic search across well log data, incorporating patterns from the Geolog-Python-Loglan repository.

## Features

- **Natural Language Querying**: Ask questions like "Which wells have porosity > 0.2?" or "Show me wells in the Hugin formation"
- **Semantic Search**: Find similar wells based on log patterns and characteristics
- **LAS File Processing**: Reads and processes LAS (Log ASCII Standard) files using the `lasio` library
- **Formation Tops Integration**: Incorporates formation top data from well picks
- **Interactive Visualizations**: Optional Altair-based visualizations for log data

## Project Structure

```
.
├── src/
│   ├── data_ingestion.py      # LAS file reading and parsing
│   ├── data_processor.py       # Data structuring and summarization
│   ├── embeddings.py           # Embedding generation
│   ├── vector_store.py         # Vector database operations
│   ├── rag_query.py            # Query processing and RAG pipeline
│   ├── visualization.py        # Altair-based visualizations (optional)
│   └── main.py                 # Main application entry point
├── data/
│   └── processed/              # Processed data cache
├── vector_db/                  # Vector database storage
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the Volve data is in the `spwla_volve-main/` directory

## Usage

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the Volve data is in the `spwla_volve-main/` directory

### Web Frontend (Recommended)

The easiest way to use the system is through the modern web interface:

```bash
# Windows
run_frontend.bat

# Linux/Mac
chmod +x run_frontend.sh
./run_frontend.sh

# Or directly with Streamlit
streamlit run src/frontend.py
```

The frontend will open in your browser at `http://localhost:8501`

**Features:**
- 🎨 Modern, sleek interface
- 🔍 Natural language querying
- 📊 Interactive results display
- ⚙️ Easy configuration
- 💡 Example queries
- 📈 Results visualization

### Command Line Interface

#### Build the index (first time or to rebuild):
```bash
python src/main.py --rebuild
```

#### Run a single query:
```bash
python src/main.py --query "Which wells have high porosity?"
python src/main.py --query "Find wells with permeability > 100 mD"
python src/main.py --query "Show me wells in the Hugin formation"
```

#### Interactive mode:
```bash
python src/main.py --interactive
```

#### Query options:
- `--query, -q`: Query string to execute
- `--results, -n`: Number of results to return (default: 10)
- `--search-type, -s`: Type of search - "structured", "semantic", or "hybrid" (default: hybrid)
- `--rebuild, -r`: Rebuild the vector index from scratch
- `--interactive, -i`: Run in interactive mode

### Python API Usage

```python
from src.main import VolveRAGSystem

# Initialize the system (will use existing index if available)
rag = VolveRAGSystem()

# Or rebuild the index
rag = VolveRAGSystem(rebuild_index=True)

# Query the system
results = rag.query("Which wells have high porosity?", n_results=10)
rag.print_results(results, "Which wells have high porosity?")
```

## Data Sources

- **LAS Files**: Well log data in LAS format from various well folders
- **Formation Tops**: Extracted from `Well_picks_Volve_v1.dat`
- **Metadata**: Well information, depth ranges, locations, etc.

## Key Log Curves

The system processes common petrophysical curves:
- **PHIF**: Porosity (V/V)
- **KLOGH**: Klinkenberg-corrected horizontal permeability (mD)
- **SW**: Water saturation (V/V)
- **VSH**: Shale volume (V/V)
- **BVW**: Bound volume water (V/V)

## Notes

- The system handles missing data gracefully
- Processed data is cached to avoid re-processing LAS files
- Vector database is stored locally in `vector_db/` directory

