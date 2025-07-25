#!/usr/bin/env python3
"""
Terminal Interface for Smart Document Analyzer
==============================================

Direct command-line interface to interact with the AI document system
without needing MCP protocol.
"""

import asyncio
import sys
from pathlib import Path
from server.document_processor import DocumentProcessor

class TerminalInterface:
    def __init__(self):
        self.processor = DocumentProcessor()
        
    async def upload_document(self, file_path: str, doc_name: str = ""):
        """Upload and process a document."""
        try:
            print(f"📄 Processing: {file_path}")
            doc_id = await self.processor.process_document(file_path, doc_name)
            print(f"✅ Document processed successfully!")
            print(f"📍 Document ID: {doc_id}")
            return doc_id
        except Exception as e:
            print(f"❌ Error: {e}")
            return None
    
    async def search_documents(self, query: str, limit: int = 3):
        """Search through all documents."""
        try:
            print(f"🔍 Searching for: '{query}'")
            results = await self.processor.search_document(query, limit=limit)
            
            if not results:
                print(f"❌ No results found for: '{query}'")
                return
            
            print(f"📊 Found {len(results)} results:\n")
            
            for i, result in enumerate(results, 1):
                score = result['similarity_score'] * 100
                doc_name = result['metadata'].get('doc_name', 'Unknown')
                content_preview = result['content'][:150] + "..."
                
                print(f"📄 Result {i} (Relevance: {score:.1f}%)")
                print(f"📝 Document: {doc_name}")
                print(f"📖 Content: {content_preview}")
                print("-" * 50)
                
        except Exception as e:
            print(f"❌ Search error: {e}")
    
    async def ask_question(self, question: str):
        """Ask AI a question about documents."""
        try:
            print(f"❓ Question: {question}")
            print("🤖 AI is thinking...")
            answer = await self.processor.ask_question(question)
            print(f"🤖 Answer: {answer}")
        except Exception as e:
            print(f"❌ Q&A error: {e}")
    
    async def list_documents(self):
        """List all processed documents."""
        try:
            docs = self.processor.get_documents()
            if not docs:
                print("📂 No documents processed yet.")
                return
            
            print(f"📚 Processed Documents ({len(docs)}):")
            print("-" * 50)
            
            for doc in docs:
                print(f"📄 {doc['name']}")
                print(f"   ID: {doc['doc_id']}")
                print(f"   📊 {doc['chunk_count']} chunks")
                print(f"   ⏰ {doc['processed_at']}")
                print()
                
        except Exception as e:
            print(f"❌ Error listing documents: {e}")
    
    def print_help(self):
        """Show available commands."""
        print("""
🤖 Smart Document Analyzer - Terminal Interface
==============================================

Core Commands:
  upload <pdf_path> [name]     Upload and process a PDF
  search <query>               Search through documents  
  ask <question>               Ask AI about documents
  list                         List all processed documents
  help                         Show this help
  quit                         Exit

Examples:
  upload test_documents/plan.pdf "My Training Plan"
  search "monday exercises"
  ask "What are the Monday workout exercises?"
  list

🚀 Powered by GPT-4o-mini + ChromaDB
""")

async def main():
    """Main terminal interface loop."""
    interface = TerminalInterface()
    
    print("🚀 Smart Document Analyzer - Terminal Interface")
    print("Type 'help' for available commands")
    print("-" * 50)
    
    while True:
        try:
            command = input("\n📱 Enter command: ").strip()
            
            if not command:
                continue
                
            parts = command.split(maxsplit=1)
            cmd = parts[0].lower()
            args = parts[1] if len(parts) > 1 else ""
            
            if cmd == "quit" or cmd == "exit":
                print("👋 Goodbye!")
                break
            elif cmd == "help":
                interface.print_help()
            elif cmd == "upload":
                if not args:
                    print("❌ Usage: upload <pdf_path> [name]")
                    continue
                file_parts = args.split(maxsplit=1)
                file_path = file_parts[0]
                doc_name = file_parts[1] if len(file_parts) > 1 else ""
                await interface.upload_document(file_path, doc_name)
            elif cmd == "search":
                if not args:
                    print("❌ Usage: search <query>")
                    continue
                await interface.search_documents(args)
            elif cmd == "ask":
                if not args:
                    print("❌ Usage: ask <question>")
                    continue
                await interface.ask_question(args)
            elif cmd == "list":
                await interface.list_documents()
            else:
                print(f"❌ Unknown command: {cmd}")
                print("Type 'help' for available commands")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 