#!/usr/bin/env python3
"""
Simple launcher for the Advanced RAG System.
Provides easy commands for building index and querying.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from main_rag_system import AdvancedRAGSystem

def main():
    """Simple command interface."""
    if len(sys.argv) < 2:
        print("🤖 Advanced RAG System for Petrophysical Documents")
        print("\nCommands:")
        print("  build     - Build the document index")
        print("  rebuild   - Rebuild index from scratch")
        print("  query     - Query the system (provide question as argument)")
        print("  stats     - Show system statistics")
        print("  shell     - Interactive query shell")
        print("\nExamples:")
        print("  python run_rag.py build")
        print("  python run_rag.py query 'What is the porosity in well 15_9-F-1?'")
        print("  python run_rag.py shell")
        return

    command = sys.argv[1].lower()

    try:
        system = AdvancedRAGSystem()

        if command == "build":
            print("🏗️  Building index...")
            stats = system.build_index()
            print("✅ Index built successfully!")
            print(f"📊 Documents processed: {stats.get('documents_processed', 0)}")
            print(f"📄 Chunks created: {stats.get('chunking_stats', {}).get('total_chunks', 0)}")
            print(".2f")

        elif command == "rebuild":
            print("🔄 Rebuilding index...")
            stats = system.rebuild_index()
            print("✅ Index rebuilt successfully!")

        elif command == "query":
            if len(sys.argv) < 3:
                print("❌ Please provide a question: python run_rag.py query 'your question'")
                return

            question = " ".join(sys.argv[2:])
            result = system.query(question, detailed_response=True)

            print(f"\n🤖 Answer: {result['answer']}")
            print(f"📊 Confidence: {result['confidence']} | Time: {result['processing_time']}")
            print(f"📚 Sources: {result['sources_count']} documents")

            if result.get('sources'):
                print("\n📋 Top Sources:")
                for i, source in enumerate(result['sources'][:3], 1):
                    print(f"{i}. {source['document']} (score: {source['score']:.3f})")
                    print(f"   Preview: {source['preview'][:100]}...")

        elif command == "stats":
            stats = system.get_system_stats()
            print("📊 System Statistics:")
            print(f"  Status: {stats.get('system_status', 'unknown')}")
            print(f"  Documents processed: {stats.get('documents_processed', 0)}")
            print(f"  Total chunks: {stats.get('vector_store', {}).get('total_chunks', 0)}")
            print(f"  Index size: {stats.get('vector_store', {}).get('index_size_mb', 0):.2f} MB")

        elif command == "shell":
            print("🤖 Interactive RAG Shell")
            print("Type 'quit' or 'exit' to stop, 'help' for commands.")

            while True:
                try:
                    question = input("\n🔍 Query: ").strip()
                    if not question:
                        continue

                    if question.lower() in ['quit', 'exit', 'q']:
                        break
                    elif question.lower() == 'help':
                        print("Commands: quit/exit, help, stats")
                        continue
                    elif question.lower() == 'stats':
                        stats = system.get_system_stats()
                        print(f"Documents: {stats.get('documents_processed', 0)}, Chunks: {stats.get('vector_store', {}).get('total_chunks', 0)}")
                        continue

                    result = system.query(question)

                    print(f"\n🤖 Answer: {result['answer']}")
                    print(f"📊 Confidence: {result['confidence']}")

                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")

            print("\n👋 Goodbye!")

        else:
            print(f"❌ Unknown command: {command}")
            print("Use: build, rebuild, query, stats, or shell")

    except Exception as e:
        print(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

