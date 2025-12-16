"""
Main entry point for LangGraph-based RAG system.
"""
import os
import argparse
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from .loaders.document_loader import DocumentLoader
from .tools.retriever_tool import RetrieverTool
from .tools.well_picks_tool import WellPicksTool
from .tools.section_lookup_tool import SectionLookupTool
from .tools.petro_params_tool import PetroParamsTool
from .tools.formation_properties_tool import FormationPropertiesTool
from .tools.eval_params_tool import EvalParamsTool
from .tools.structured_facts_tool import StructuredFactsTool
from .graph.rag_graph import build_rag_graph

# Load environment variables
load_dotenv()

# Fallback: load OPENAI_API_KEY from Streamlit secrets if present.
def _load_openai_key_from_streamlit_secrets() -> None:
    if os.getenv("OPENAI_API_KEY"):
        return
    try:
        import tomllib  # py311+
    except Exception:
        return
    # Prefer project-relative path (works even if CWD is not advanced_rag/)
    candidates = [
        Path(".streamlit") / "secrets.toml",
        Path(__file__).resolve().parent.parent / ".streamlit" / "secrets.toml",
    ]
    for p in candidates:
        try:
            if not p.exists():
                continue
            data = tomllib.loads(p.read_text(encoding="utf-8"))
            key = data.get("OPENAI_API_KEY")
            if isinstance(key, str) and key.strip():
                os.environ["OPENAI_API_KEY"] = key.strip()
                return
        except Exception:
            continue


_load_openai_key_from_streamlit_secrets()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_index(
    documents_path: str,
    persist_directory: str = "./data/vectorstore",
    embedding_model: str = "text-embedding-3-small"
):
    """
    Build vector store index from documents.
    
    Args:
        documents_path: Path to directory containing documents
        persist_directory: Directory to persist ChromaDB
        embedding_model: OpenAI embedding model name
    """
    logger.info("=" * 60)
    logger.info("Building RAG Index")
    logger.info("=" * 60)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Load documents
    logger.info(f"Loading documents from: {documents_path}")
    loader = DocumentLoader()
    documents = loader.load_documents(Path(documents_path))
    
    if not documents:
        raise ValueError("No documents found to process")
    
    # Build vector store with finer indexing (smaller chunks, better overlap)
    logger.info("Building vector store with IntelligentChunker...")
    retriever_tool = RetrieverTool(
        persist_directory=persist_directory,
        embedding_model=embedding_model,
        chunk_size=500,  # Finer indexing: smaller chunks
        chunk_overlap=150  # 30% overlap for better context
    )

    # Build deterministic section index (heading -> section text)
    try:
        section_index_path = str(Path(persist_directory) / "section_index.json")
        SectionLookupTool.build_index(documents, section_index_path)
    except Exception as e:
        logger.warning(f"[SECTION] Failed to build section index: {e}")

    # Build deterministic petrophysical parameters cache from section index
    try:
        section_index_path = str(Path(persist_directory) / "section_index.json")
        petro_cache_path = str(Path(persist_directory) / "petro_params_cache.json")
        PetroParamsTool.build_index(section_index_path, petro_cache_path)
    except Exception as e:
        logger.warning(f"[PETRO_PARAMS] Failed to build petro params cache: {e}")

    # Build deterministic evaluation parameters cache (tables like "Evaluation Parameters 15/9-F5")
    try:
        eval_params_cache_path = str(Path(persist_directory) / "eval_params_cache.json")
        EvalParamsTool.build_index(documents_path, eval_params_cache_path)
    except Exception as e:
        logger.warning(f"[EVAL_PARAMS] Failed to build evaluation params cache: {e}")

    # Build deterministic "facts" cache from section index (notes/narrative numeric statements)
    try:
        facts_cache_path = str(Path(persist_directory) / "facts_cache.json")
        section_index_path = str(Path(persist_directory) / "section_index.json")
        StructuredFactsTool.build_index(documents_path, facts_cache_path, section_index_path=section_index_path)
    except Exception as e:
        logger.warning(f"[FACTS] Failed to build facts cache: {e}")

    retriever_tool.build_vectorstore(documents)
    
    logger.info("=" * 60)
    logger.info(f"[OK] Index built successfully!")
    logger.info(f"  - Documents processed: {len(documents)}")
    logger.info(f"  - Vector store location: {persist_directory}")
    logger.info("=" * 60)


def query(
    question: str,
    persist_directory: str = "./data/vectorstore",
    embedding_model: str = "text-embedding-3-small",
    stream: bool = False
):
    """
    Query the RAG system.
    
    Args:
        question: User question
        persist_directory: Directory where ChromaDB is persisted
        embedding_model: OpenAI embedding model name
        stream: Whether to stream results
    """
    logger.info("=" * 60)
    logger.info(f"Query: {question}")
    logger.info("=" * 60)
    
    # Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    
    # Load vector store
    retriever_tool = RetrieverTool(
        persist_directory=persist_directory,
        embedding_model=embedding_model
    )
    
    if not retriever_tool.load_vectorstore():
        raise RuntimeError("Vector store not found. Run --build-index first.")
    
    # Get retriever tool
    retrieve_tool = retriever_tool.get_retriever_tool()

    # Structured well picks tool (deterministic lookup; no embeddings)
    # Prefer the real .dat in the documents folder if available; otherwise try default location.
    well_picks_path = None
    # Try to locate in the adjacent spwla_volve-main folder relative to CWD
    candidate_paths = [
        Path("spwla_volve-main") / "Well_picks_Volve_v1.dat",
        Path("../spwla_volve-main") / "Well_picks_Volve_v1.dat",
        Path("./") / "Well_picks_Volve_v1.dat",
    ]
    for p in candidate_paths:
        if p.exists():
            well_picks_path = str(p)
            break
    if well_picks_path is None:
        # Fall back: search within the documents_path logic used by build-index defaults
        # This is best-effort; if not found, structured tool will error on init.
        well_picks_path = str(Path.cwd().parent / "spwla_volve-main" / "Well_picks_Volve_v1.dat")

    well_picks_tool = WellPicksTool(dat_path=well_picks_path).get_tool()

    # Section lookup tool
    section_tool = None
    try:
        section_index_path = str(Path(persist_directory) / "section_index.json")
        section_tool = SectionLookupTool(section_index_path).get_tool()
    except Exception as e:
        logger.warning(f"[SECTION] Section lookup tool unavailable: {e}")

    # Petrophysical parameter lookup tool
    petro_tool = None
    try:
        petro_cache_path = str(Path(persist_directory) / "petro_params_cache.json")
        petro_tool = PetroParamsTool(petro_cache_path).get_tool()
    except Exception as e:
        logger.warning(f"[PETRO_PARAMS] Petro params tool unavailable: {e}")

    # One-shot formation+properties tool (requires both well picks .dat and petro params cache)
    formation_props_tool = None
    try:
        petro_cache_path = str(Path(persist_directory) / "petro_params_cache.json")
        if Path(petro_cache_path).exists():
            formation_props_tool = FormationPropertiesTool(
                well_picks_dat_path=well_picks_path,
                petro_params_cache_path=petro_cache_path,
            ).get_tool()
    except Exception as e:
        logger.warning(f"[WELL_FORMATION_PROPERTIES] Tool unavailable: {e}")

    # Evaluation parameters tool
    eval_params_tool = None
    try:
        eval_cache_path = str(Path(persist_directory) / "eval_params_cache.json")
        if Path(eval_cache_path).exists():
            eval_params_tool = EvalParamsTool(eval_cache_path).get_tool()
    except Exception as e:
        logger.warning(f"[EVAL_PARAMS] Tool unavailable: {e}")

    # Structured facts tool
    facts_tool = None
    try:
        facts_cache_path = str(Path(persist_directory) / "facts_cache.json")
        if Path(facts_cache_path).exists():
            facts_tool = StructuredFactsTool(facts_cache_path).get_tool()
    except Exception as e:
        logger.warning(f"[FACTS] Tool unavailable: {e}")
    
    # Build graph
    tools = [well_picks_tool]
    if section_tool:
        tools.append(section_tool)
    if petro_tool:
        tools.append(petro_tool)
    if formation_props_tool:
        tools.append(formation_props_tool)
    if eval_params_tool:
        tools.append(eval_params_tool)
    if facts_tool:
        tools.append(facts_tool)
    tools.append(retrieve_tool)
    graph = build_rag_graph(tools)
    
    # Run query
    initial_state = {"messages": [{"role": "user", "content": question}]}
    
    if stream:
        logger.info("Streaming results...")
        print("\n" + "=" * 60)
        for chunk in graph.stream(initial_state):
            for node, update in chunk.items():
                if node == "generate_query_or_respond":
                    msg = update["messages"][-1]
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        print(f"\n[AGENT] Decided to retrieve documents...")
                    else:
                        print(f"\n[AGENT] {msg.content}")
                elif node == "retrieve":
                    print(f"\n[RETRIEVE] Retrieved context from documents")
                elif node == "generate_answer":
                    print(f"\n[ANSWER] {update['messages'][-1].content}")
        print("=" * 60 + "\n")
    else:
        result = graph.invoke(initial_state)
        answer = result["messages"][-1].content
        
        print("\n" + "=" * 60)
        print("ANSWER:")
        print("=" * 60)
        print(answer)
        print("=" * 60 + "\n")


def chat(
    persist_directory: str = "./data/vectorstore",
    embedding_model: str = "text-embedding-3-small",
):
    """
    Interactive multi-turn chat mode.

    This preserves message history so clarification workflows work in the terminal:
    - User: "mtrx densiy hugin 15/9-f-5"
    - Assistant: "matrix density or fluid density?"
    - User: "matrix density"
    -> system can still resolve well/formation from history.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    # Load vector store (once)
    retriever_tool = RetrieverTool(persist_directory=persist_directory, embedding_model=embedding_model)
    if not retriever_tool.load_vectorstore():
        raise RuntimeError("Vector store not found. Run --build-index first.")
    retrieve_tool = retriever_tool.get_retriever_tool()

    # Locate well picks .dat (best effort)
    well_picks_path = None
    candidate_paths = [
        Path("spwla_volve-main") / "Well_picks_Volve_v1.dat",
        Path("../spwla_volve-main") / "Well_picks_Volve_v1.dat",
        Path("./") / "Well_picks_Volve_v1.dat",
    ]
    for p in candidate_paths:
        if p.exists():
            well_picks_path = str(p)
            break
    if well_picks_path is None:
        well_picks_path = str(Path.cwd().parent / "spwla_volve-main" / "Well_picks_Volve_v1.dat")

    tools = [WellPicksTool(dat_path=well_picks_path).get_tool()]

    # Optional deterministic tools
    try:
        section_index_path = str(Path(persist_directory) / "section_index.json")
        tools.append(SectionLookupTool(section_index_path).get_tool())
    except Exception as e:
        logger.warning(f"[SECTION] Section lookup tool unavailable: {e}")

    try:
        petro_cache_path = str(Path(persist_directory) / "petro_params_cache.json")
        tools.append(PetroParamsTool(petro_cache_path).get_tool())
        if Path(petro_cache_path).exists():
            tools.append(
                FormationPropertiesTool(
                    well_picks_dat_path=well_picks_path,
                    petro_params_cache_path=petro_cache_path,
                ).get_tool()
            )
    except Exception as e:
        logger.warning(f"[PETRO_PARAMS] Petro params tool unavailable: {e}")

    try:
        eval_cache_path = str(Path(persist_directory) / "eval_params_cache.json")
        if Path(eval_cache_path).exists():
            tools.append(EvalParamsTool(eval_cache_path).get_tool())
    except Exception as e:
        logger.warning(f"[EVAL_PARAMS] Tool unavailable: {e}")

    try:
        facts_cache_path = str(Path(persist_directory) / "facts_cache.json")
        if Path(facts_cache_path).exists():
            tools.append(StructuredFactsTool(facts_cache_path).get_tool())
    except Exception as e:
        logger.warning(f"[FACTS] Tool unavailable: {e}")

    tools.append(retrieve_tool)
    graph = build_rag_graph(tools)

    print("\n" + "=" * 60)
    print("CHAT MODE (type 'exit' to quit)")
    print("=" * 60)

    messages = []
    while True:
        try:
            user = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not user:
            continue
        if user.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break

        messages.append({"role": "user", "content": user})
        result = graph.invoke({"messages": messages})
        answer = result["messages"][-1].content
        print("\nAssistant:\n")
        print(answer)
        messages.append({"role": "assistant", "content": answer})


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="LangGraph-based RAG System")
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build vector store index from documents"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query the RAG system"
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Interactive multi-turn chat mode (preserves conversation history)"
    )
    parser.add_argument(
        "--documents-path",
        type=str,
        default=None,
        help="Path to documents directory (defaults to ../spwla_volve-main)"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./data/vectorstore",
        help="Directory to persist vector store (default: ./data/vectorstore)"
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model (default: text-embedding-3-small)"
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream results (for queries)"
    )
    
    args = parser.parse_args()
    
    # Set default documents path
    if args.documents_path is None:
        current_dir = Path.cwd()
        possible_paths = [
            current_dir / "spwla_volve-main",
            current_dir.parent / "spwla_volve-main",
            Path("../spwla_volve-main")
        ]
        for path in possible_paths:
            if path.exists():
                args.documents_path = str(path)
                break
        
        if args.documents_path is None:
            raise FileNotFoundError(
                "Could not find spwla_volve-main directory. "
                "Please specify --documents-path"
            )
    
    try:
        if args.build_index:
            build_index(
                documents_path=args.documents_path,
                persist_directory=args.persist_dir,
                embedding_model=args.embedding_model
            )
        elif args.chat:
            chat(
                persist_directory=args.persist_dir,
                embedding_model=args.embedding_model,
            )
        elif args.query:
            query(
                question=args.query,
                persist_directory=args.persist_dir,
                embedding_model=args.embedding_model,
                stream=args.stream
            )
        else:
            parser.print_help()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

