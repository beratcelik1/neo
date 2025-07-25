#!/usr/bin/env python3
"""
Database Cleanup Script for Smart Document Analyzer
==================================================

Simple script to clean up the vector database and document metadata.
"""

import asyncio
import shutil
from pathlib import Path
from server.document_processor import DocumentProcessor

class DatabaseCleaner:
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.data_dir = self.project_root / "data"
        self.documents_dir = self.data_dir / "documents"
        self.vector_db_dir = self.data_dir / "vector_db"
        
    def clean_all(self):
        """Remove all data - complete reset."""
        print("🧹 COMPLETE DATABASE CLEANUP")
        print("=" * 50)
        
        if self.data_dir.exists():
            print(f"🗑️  Removing all data from: {self.data_dir}")
            shutil.rmtree(self.data_dir)
            print("✅ All data removed!")
        else:
            print("📂 No data directory found - already clean!")
            
        # Recreate empty directories
        self.data_dir.mkdir(exist_ok=True)
        self.documents_dir.mkdir(exist_ok=True)
        self.vector_db_dir.mkdir(exist_ok=True)
        print("📁 Empty directories recreated")
        
    def clean_duplicates_only(self):
        """Remove duplicate entries only."""
        print("🧹 REMOVING DUPLICATE ENTRIES ONLY")
        print("=" * 50)
        
        async def cleanup():
            try:
                processor = DocumentProcessor()
                collection = processor.collection
                
                # Get all items
                all_items = collection.get()
                total_before = len(all_items['ids'])
                print(f"📊 Found {total_before} chunks before cleanup")
                
                # Group by content hash to find duplicates
                content_groups = {}
                for i, (id, metadata, document) in enumerate(zip(
                    all_items['ids'], all_items['metadatas'], all_items['documents']
                )):
                    content_hash = hash(document[:200])  # Use first 200 chars
                    if content_hash not in content_groups:
                        content_groups[content_hash] = []
                    content_groups[content_hash].append({
                        'id': id,
                        'doc_name': metadata.get('doc_name', 'unknown')
                    })
                
                # Remove duplicates
                duplicates_removed = 0
                for group in content_groups.values():
                    if len(group) > 1:
                        # Keep first, remove rest
                        for item in group[1:]:
                            collection.delete(ids=[item['id']])
                            duplicates_removed += 1
                            print(f"🗑️  Removed: {item['id']}")
                
                # Check final count
                remaining = collection.get()
                total_after = len(remaining['ids'])
                
                print(f"✅ Cleanup complete!")
                print(f"📊 Before: {total_before} chunks")
                print(f"📊 After: {total_after} chunks") 
                print(f"🗑️  Removed: {duplicates_removed} duplicates")
                
            except Exception as e:
                print(f"❌ Error during cleanup: {e}")
        
        asyncio.run(cleanup())
    
    def show_status(self):
        """Show current database status."""
        print("📊 DATABASE STATUS")
        print("=" * 50)
        
        if not self.data_dir.exists():
            print("📂 No data directory - database is empty")
            return
            
        # Count files
        doc_files = list(self.documents_dir.glob("*.json")) if self.documents_dir.exists() else []
        vector_files = list(self.vector_db_dir.rglob("*")) if self.vector_db_dir.exists() else []
        
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📄 Document metadata files: {len(doc_files)}")
        print(f"🔍 Vector database files: {len([f for f in vector_files if f.is_file()])}")
        
        # Calculate sizes
        total_size = sum(f.stat().st_size for f in self.data_dir.rglob("*") if f.is_file())
        print(f"💾 Total size: {total_size / 1024 / 1024:.1f} MB")
        
        # Show document list
        async def show_docs():
            try:
                processor = DocumentProcessor()
                docs = processor.get_documents()
                print(f"\n📚 Documents in system: {len(docs)}")
                for doc in docs:
                    print(f"   📄 {doc['name']} ({doc['chunk_count']} chunks)")
            except Exception as e:
                print(f"⚠️  Could not load documents: {e}")
        
        asyncio.run(show_docs())

def main():
    """Interactive cleanup menu."""
    cleaner = DatabaseCleaner()
    
    while True:
        print("\n" + "🧹 DATABASE CLEANUP TOOL" + "\n" + "=" * 30)
        print("1. 📊 Show database status")
        print("2. 🧹 Remove duplicates only")
        print("3. 🗑️  Complete cleanup (remove all data)")
        print("4. 🚪 Exit")
        
        choice = input("\n📱 Choose option (1-4): ").strip()
        
        if choice == "1":
            cleaner.show_status()
        elif choice == "2":
            confirm = input("⚠️  Remove duplicates? (y/N): ").strip().lower()
            if confirm == 'y':
                cleaner.clean_duplicates_only()
        elif choice == "3":
            confirm = input("⚠️  Remove ALL data? This cannot be undone! (y/N): ").strip().lower()
            if confirm == 'y':
                cleaner.clean_all()
        elif choice == "4":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main() 