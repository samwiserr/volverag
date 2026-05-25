"""
build_sota.py - Offline SOTA index builder for VolveRAG.

This script is the single entry point to build a production-ready vectorstore
that includes all three SOTA indexing techniques:

  1. Contextual Chunking  (Anthropic, Sept 2024)
  2. RAPTOR tree          (Sarthi et al., ICLR 2024)
  3. Hybrid BM25 + local dense embeddings (RRF fusion)

Run this LOCALLY before deploying to Streamlit Community Cloud.
After the build, zip the vectorstore and upload it to GitHub Releases.

Usage:
    cd advanced_rag
    pip install -r requirements.txt -r requirements-build.txt
    python scripts/build_sota.py --documents-path ../spwla_volve-main

    # Optional: skip RAPTOR (faster, lower cost)
    python scripts/build_sota.py --documents-path ../spwla_volve-main --no-raptor

    # Optional: skip contextual enrichment (fastest, no extra LLM calls)
    python scripts/build_sota.py --documents-path ../spwla_volve-main --no-contextual
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

# Bootstrap
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / ".env")

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("build_sota")


# Helpers

def _section(title: str) -> None:
    width = 60
    print(f"\n{'-' * width}")
    print(f"  {title}")
    print(f"{'-' * width}")


def _check_prerequisites(docs_path: Path) -> None:
    _section("1/5 Checking prerequisites")

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key or groq_key.startswith("gsk_your"):
        print("  ERROR: GROQ_API_KEY not set. Edit .env and re-run.")
        sys.exit(1)
    print(f"  GROQ_API_KEY     : {'*' * 8}{groq_key[-4:]}")

    if not docs_path.exists():
        print(f"  ERROR: --documents-path not found: {docs_path}")
        sys.exit(1)

    pdfs = list(docs_path.rglob("*.pdf")) + list(docs_path.rglob("*.PDF"))
    docs = list(docs_path.rglob("*.doc")) + list(docs_path.rglob("*.docx"))
    dat  = list(docs_path.rglob("Well_picks_Volve_v1.dat"))
    total = len(pdfs) + len(docs) + len(dat)
    print(f"  Documents found  : {len(pdfs)} PDFs, {len(docs)} DOCs, {len(dat)} DAT")
    if total == 0:
        print("  WARNING: No documents found. The vectorstore will be empty.")
    else:
        print(f"  Total            : {total} files")

    print("  Prerequisites    : OK")


def _set_env_flags(
    contextual: bool,
    raptor: bool,
    raptor_levels: int,
    raptor_clusters: int,
) -> None:
    _section("2/5 Setting SOTA feature flags")

    flags = {
        "LLM_PROVIDER":         "groq",
        "EMBEDDING_PROVIDER":   "huggingface",
        "GROQ_MODEL":           os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "GROQ_FAST_MODEL":      os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant"),
        "LOCAL_EMBEDDING_MODEL": os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"),
        "RAG_CONTEXTUAL":       "true" if contextual else "false",
        "RAG_CONTEXT_MODEL":    os.getenv("RAG_CONTEXT_MODEL", os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")),
        "RAG_RAPTOR":           "true" if raptor else "false",
        "RAG_RAPTOR_MODEL":     os.getenv("RAG_RAPTOR_MODEL", os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")),
        "RAG_RAPTOR_LEVELS":    str(raptor_levels),
        "RAG_RAPTOR_CLUSTERS":  str(raptor_clusters),
        "RAG_HYBRID_FUSION":    "rrf",
        "RAG_RRF_K":            "60",
        "RAG_MMR":              "true",
        "RAG_USE_CROSS_ENCODER":"true",
        "RAG_RERANK":           "llm",
        "RAG_HYDE":             "true",   # also active at query time
        "RAG_HYDE_MODEL":       os.getenv("RAG_HYDE_MODEL", os.getenv("GROQ_FAST_MODEL", "llama-3.1-8b-instant")),
    }
    for k, v in flags.items():
        os.environ[k] = v
        status = "ON " if v in ("true", "rrf", "llm") else "   "
        print(f"  {status}  {k:<28} = {v}")


def _build_index(docs_path: Path, persist_dir: Path) -> None:
    _section("3/5 Building SOTA vectorstore")
    print(f"  Source           : {docs_path}")
    print(f"  Persist dir      : {persist_dir}")
    print()
    if persist_dir.exists():
        print("  Removing existing vectorstore to avoid embedding-dimension conflicts...")
        shutil.rmtree(persist_dir)

    # Import here so env flags set above are visible to the modules
    from src.main import build_index
    t0 = time.perf_counter()
    build_index(
        documents_path=str(docs_path),
        persist_directory=str(persist_dir),
        embedding_model=os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"),
    )
    elapsed = time.perf_counter() - t0
    print(f"\n  Build completed in {elapsed / 60:.1f} min")


def _verify_vectorstore(persist_dir: Path) -> None:
    _section("4/5 Verifying vectorstore")
    chroma = persist_dir / "chroma.sqlite3"
    lexical = persist_dir / "lexical_store.jsonl"
    context_cache = persist_dir.parent / "context_cache" / "context_cache.json"

    ok = True
    for label, path in [
        ("chroma.sqlite3",          chroma),
        ("lexical_store.jsonl",     lexical),
    ]:
        exists = path.exists()
        size = f"{path.stat().st_size / 1024:.0f} KB" if exists else "-"
        mark   = "OK" if exists else "MISSING"
        print(f"  {mark}  {label:<30} {size}")
        if not exists:
            ok = False

    if context_cache.exists():
        n = len(json.loads(context_cache.read_text(encoding="utf-8")))
        print(f"  OK  context_cache.json              {n} cached contexts")
    else:
        print("  -   context_cache.json              (not found - contextual chunking may be off)")

    if not ok:
        print("\n  ERROR: Required vectorstore files are missing. Check build logs above.")
        sys.exit(1)
    print("\n  Vectorstore looks healthy.")


def _zip_vectorstore(persist_dir: Path, output_zip: Path) -> None:
    _section("5/5 Packaging vectorstore for GitHub Releases")

    if output_zip.exists():
        output_zip.unlink()

    # Extra cache files that live in data/ (parent of persist_dir) and must be
    # available at runtime under advanced_rag/data/ on the Streamlit deployment.
    EXTRA_DATA_FILES = [
        "well_picks_cache.json",
        "petro_params_cache.json",
        "section_index.json",
        "structured_facts_cache.json",
        "eval_tables_cache.json",
    ]

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        # Always add all vectorstore files under the "vectorstore/" prefix
        for file in persist_dir.rglob("*"):
            if file.is_file():
                zf.write(file, file.relative_to(persist_dir.parent))

        # Add any extra data/ cache files at the root of the zip so the
        # downloader can place them in advanced_rag/data/ on Streamlit.
        data_dir = persist_dir.parent
        for name in EXTRA_DATA_FILES:
            extra = data_dir / name
            if extra.exists():
                zf.write(extra, name)
                print(f"  Included         : {name} ({extra.stat().st_size / 1024:.1f} KB)")

    size_mb = output_zip.stat().st_size / (1024 * 1024)
    print(f"  Created          : {output_zip}")
    print(f"  Size             : {size_mb:.1f} MB")
    print()
    print("  Next steps:")
    print(f"  1. Upload {output_zip.name} to GitHub Releases")
    print("     https://github.com/<you>/<repo>/releases/new")
    print("  2. Copy the release asset URL")
    print("  3. Add to Streamlit Secrets:")
    print("       VECTORSTORE_URL = \"https://github.com/.../releases/download/.../vectorstore.zip\"")
    print("       GROQ_API_KEY    = \"gsk_...\"")
    print("       LLM_PROVIDER    = \"groq\"")
    print("       EMBEDDING_PROVIDER = \"huggingface\"")


# Main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SOTA VolveRAG vectorstore for Streamlit deployment"
    )
    parser.add_argument(
        "--documents-path",
        required=True,
        help="Path to the Volve dataset directory (contains PDFs, DOCs, Well_picks_Volve_v1.dat)",
    )
    parser.add_argument(
        "--persist-dir",
        default=str(_REPO_ROOT / "data" / "vectorstore"),
        help="Output directory for the ChromaDB vectorstore (default: data/vectorstore)",
    )
    parser.add_argument(
        "--output-zip",
        default=str(_REPO_ROOT / "vectorstore.zip"),
        help="Path for the packaged ZIP (default: vectorstore.zip in repo root)",
    )
    parser.add_argument(
        "--no-contextual",
        action="store_true",
        help="Disable Contextual Chunking (faster, lower cost, lower quality)",
    )
    parser.add_argument(
        "--no-raptor",
        action="store_true",
        help="Disable RAPTOR tree (faster, lower cost, weaker on synthesis queries)",
    )
    parser.add_argument(
        "--raptor-levels",
        type=int,
        default=2,
        help="Number of RAPTOR summary levels (default: 2)",
    )
    parser.add_argument(
        "--raptor-clusters",
        type=int,
        default=8,
        help="Target clusters per RAPTOR level (default: 8)",
    )
    parser.add_argument(
        "--skip-zip",
        action="store_true",
        help="Skip packaging step (useful if you only want to test locally)",
    )
    args = parser.parse_args()

    docs_path   = Path(args.documents_path).resolve()
    persist_dir = Path(args.persist_dir).resolve()
    output_zip  = Path(args.output_zip).resolve()

    print("\n" + "=" * 60)
    print("  VolveRAG SOTA Build")
    print("  Contextual + RAPTOR + HyDE + RRF Fusion")
    print("=" * 60)

    _check_prerequisites(docs_path)
    _set_env_flags(
        contextual=not args.no_contextual,
        raptor=not args.no_raptor,
        raptor_levels=args.raptor_levels,
        raptor_clusters=args.raptor_clusters,
    )
    _build_index(docs_path, persist_dir)
    _verify_vectorstore(persist_dir)

    if not args.skip_zip:
        _zip_vectorstore(persist_dir, output_zip)

    print("\n" + "=" * 60)
    print("  SOTA build complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
