#!/usr/bin/env python3
"""Test script to build the RAG index"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from main_rag_system import AdvancedRAGSystem

    print("🤖 Testing Advanced RAG System Build...")

    # Initialize system
    system = AdvancedRAGSystem()

    # Build index
    print("🏗️  Building index for all 37 documents...")
    stats = system.build_index()

    print("✅ Index built successfully!")
    print(f"📊 Documents processed: {stats.get('documents_processed', 0)}")
    print(f"📄 Chunks created: {stats.get('chunking_stats', {}).get('total_chunks', 0)}")
    print(f"⏱️  Build time: {stats.get('build_time', 'unknown')}")

    # Test a simple query
    print("\n🧪 Testing query...")
    result = system.query("What is porosity?", top_k=3)
    print(f"🤖 Answer: {result['answer'][:200]}...")
    print(f"📊 Confidence: {result['confidence']}")

    print("\n🎉 RAG System is working!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

