#!/usr/bin/env python3
"""
Document Processor - Fixed Paths Version
======================================

Core AI processing engine for document analysis with STABLE folder structure.
- All paths are relative to PROJECT ROOT (not where script runs)
- Documents stored in: /project_root/data/documents/
- Vector DB stored in: /project_root/data/vector_db/
- No more folders created in random places!
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from datetime import datetime
import hashlib

# Third-party imports
import openai
import chromadb
import tiktoken
from PyPDF2 import PdfReader
import pdfplumber

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Core document processing engine with STABLE folder structure.
    
    Key Features:
    - Real OpenAI integration (GPT + embeddings)
    - ChromaDB vector database for semantic search
    - PDF processing with text extraction
    - Stable paths - no more random folder creation!
    """
    
    def __init__(self):
        """Initialize with STABLE paths relative to project root."""
        
        # 🔧 Get PROJECT ROOT (always /path/to/neo regardless of where we run)
        self.project_root = Path(__file__).parent.parent.absolute()
        logger.info(f"📁 Project root: {self.project_root}")
        
        # 📁 STABLE data directories - always in project root
        self.data_dir = self.project_root / "data"
        self.documents_dir = self.data_dir / "documents"
        self.vector_db_dir = self.data_dir / "vector_db"
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.documents_dir.mkdir(exist_ok=True)
        self.vector_db_dir.mkdir(exist_ok=True)
        
        print(f"✅ Document Processor initialized with stable paths:")
        print(f"📁 Documents: {self.documents_dir}")
        print(f"🔍 Vector DB: {self.vector_db_dir}")
        
        # Load environment variables
        from dotenv import load_dotenv
        load_dotenv(self.project_root / ".env")
        
        # Initialize OpenAI client
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        
        # Configuration from environment or defaults
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002')
        self.chat_model = os.getenv('CHAT_MODEL', 'gpt-4o-mini')
        self.chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
        self.chunk_overlap = int(os.getenv('CHUNK_OVERLAP', '200'))
        self.max_chunks_per_document = int(os.getenv('MAX_CHUNKS_PER_DOCUMENT', '100'))
        
        print(f"🤖 AI Models: {self.embedding_model}, {self.chat_model}")
        
        # Initialize Vector Database (ChromaDB) with STABLE path
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.vector_db_dir),
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name="document_chunks",
                metadata={"description": "Document chunks with embeddings"}
            )
            logger.info(f"✅ ChromaDB initialized at: {self.vector_db_dir}")
        except Exception as e:
            logger.error(f"❌ ChromaDB initialization failed: {e}")
            raise
        
        print("✅ Document processor initialized successfully")
        
    def _generate_document_id(self, file_path: str, document_name: str = "") -> str:
        """Generate a unique document ID."""
        file_path_str = str(file_path)
        name_for_id = document_name or Path(file_path_str).stem
        # Create hash from file path and name for uniqueness
        hash_input = f"{file_path_str}_{name_for_id}_{datetime.now().isoformat()}"
        doc_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        # Format: doc_filename_timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"doc_{name_for_id.replace(' ', '_')}_{timestamp}_{doc_hash}"
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF using multiple methods for robustness."""
        try:
            # Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(file_path) as pdf:
                text_parts = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                
                if text_parts:
                    return "\n\n".join(text_parts)
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, trying PyPDF2...")
        
        try:
            # Fallback to PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text_parts = []
                for page in pdf_reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                
                return "\n\n".join(text_parts)
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {e}")
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into balanced chunks with overlap for better search performance."""
        # Use tiktoken for accurate token counting
        try:
            encoding = tiktoken.encoding_for_model(self.chat_model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length and len(chunks) < self.max_chunks_per_document:
            # Calculate end position
            end = min(start + self.chunk_size, text_length)
            
            # If not at the end of text, find a good break point
            if end < text_length:
                # Look backwards for sentence endings, paragraph breaks, or spaces
                original_end = end
                for i in range(end, max(end - 200, start), -1):
                    if text[i] in '.!?\n':
                        end = i + 1
                        break
                    elif text[i] == ' ':
                        end = i
                        break
                
                # If we couldn't find a good break point, use original end
                if end == start:
                    end = original_end
            
            # Extract chunk
            chunk = text[start:end].strip()
            
            # Only add non-empty chunks that meet minimum size
            if chunk and len(chunk) > 50:  # Minimum chunk size
                chunks.append(chunk)
            
            # Move start position with overlap
            # Ensure we make progress and don't get stuck
            next_start = end - self.chunk_overlap
            if next_start <= start:
                next_start = start + max(1, self.chunk_size // 2)  # Force progress
            
            start = next_start
        
        return chunks
    
    async def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks using OpenAI."""
        try:
            # OpenAI embeddings API call
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    async def _store_in_vector_db(self, doc_id: str, doc_name: str, chunks: List[str], embeddings: List[List[float]]):
        """Store chunks and embeddings in ChromaDB."""
        try:
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    'doc_id': doc_id,
                    'doc_name': doc_name,
                    'chunk_index': i
                } for i in range(len(chunks))
            ]
            
            self.collection.add(
                documents=chunks,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=chunk_ids
            )
            logger.info(f"Stored {len(chunks)} chunks for document '{doc_name}' in ChromaDB.")
        except Exception as e:
            logger.error(f"Error storing in vector DB: {e}")
            raise
    
    def _save_document_metadata(self, doc_id: str, doc_name: str, file_path: str, chunks: List[str]):
        """Save document metadata to disk for tracking."""
        metadata = {
            'doc_id': doc_id,
            'name': doc_name,
            'original_path': file_path,
            'chunk_count': len(chunks),
            'total_chars': sum(len(chunk) for chunk in chunks),
            'embedding_model': self.embedding_model,
            'chat_model': self.chat_model,
            'processed_at': datetime.now().isoformat()
        }
        
        # Save to STABLE path in data directory
        metadata_file = self.documents_dir / f"{doc_id}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata for {doc_name} to {metadata_file}")
    
    async def process_document(self, file_path: str, document_name: str = "") -> str:
        """
        Process a PDF document with real AI analysis.
        
        Args:
            file_path: Path to the PDF file (can be relative or absolute)
            document_name: Optional custom name for the document
        
        Returns:
            Document ID for further operations
        """
        try:
            # Convert to absolute path if needed
            file_path = Path(file_path).absolute()
            
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            if not file_path.suffix.lower() == '.pdf':
                raise ValueError("Only PDF files are supported currently")
            
            # Generate document ID and name
            doc_name = document_name or file_path.stem
            doc_id = self._generate_document_id(str(file_path), doc_name)
            
            logger.info(f"Processing document: {doc_name} (ID: {doc_id})")
            
            # Extract text from PDF
            logger.info("Extracting text from PDF...")
            text = self._extract_text_from_pdf(str(file_path))
            
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF")
            
            logger.info(f"Extracted {len(text)} characters from PDF")
            
            # Chunk the text
            logger.info("Chunking text for processing...")
            chunks = self._chunk_text(text)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Generate embeddings
            logger.info("Generating AI embeddings...")
            embeddings = await self._generate_embeddings(chunks)
            logger.info(f"Generated embeddings for {len(embeddings)} chunks")
            
            # Store in vector database
            logger.info("Storing in vector database...")
            await self._store_in_vector_db(doc_id, doc_name, chunks, embeddings)
            
            # Save metadata
            logger.info("Saving document metadata...")
            self._save_document_metadata(doc_id, doc_name, str(file_path), chunks)
            
            logger.info(f"✅ Document processing complete for: {doc_name}")
            return doc_id
            
        except Exception as e:
            logger.error(f"❌ Document processing failed: {e}")
            raise
    
    async def search_document(self, query: str, doc_id: str = None, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Perform enhanced semantic search across documents with duplicate removal and keyword boosting.
        
        Args:
            query: Search query
            doc_id: Optional specific document ID to search
            limit: Maximum number of results
        
        Returns:
            List of unique search results with content and metadata, ranked by relevance
        """
        try:
            # Create query embedding
            query_embeddings = await self._generate_embeddings([query])
            query_embedding = query_embeddings[0]
            
            # Build ChromaDB query
            where_filter = {"doc_id": doc_id} if doc_id else None
            
            # Search with more results to allow for deduplication
            search_limit = min(limit * 3, 20)  # Get more results to filter duplicates
            
            # Perform vector search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=search_limit,
                where=where_filter,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'][0]:
                logger.info(f"No search results found for query: {query}")
                return []
            
            # Process and deduplicate results
            processed_results = []
            seen_chunks = set()  # Track unique chunk identifiers
            
            documents = results['documents'][0]
            metadatas = results['metadatas'][0] 
            distances = results['distances'][0]
            
            for doc, metadata, distance in zip(documents, metadatas, distances):
                # Create unique identifier for this chunk
                chunk_id = f"{metadata.get('doc_id', 'unknown')}_{metadata.get('chunk_index', 0)}"
                
                # Skip if we've already seen this exact chunk
                if chunk_id in seen_chunks:
                    continue
                seen_chunks.add(chunk_id)
                
                # Convert distance to similarity score (ChromaDB uses cosine distance)
                similarity_score = 1 - distance
                
                # Boost score for keyword matches (simple keyword boosting)
                query_words = query.lower().split()
                doc_lower = doc.lower()
                keyword_matches = sum(1 for word in query_words if word in doc_lower)
                keyword_boost = (keyword_matches / len(query_words)) * 0.2  # Up to 20% boost
                
                final_score = min(similarity_score + keyword_boost, 1.0)
                
                processed_results.append({
                    'content': doc,
                    'metadata': metadata,
                    'similarity_score': final_score,
                    'chunk_id': chunk_id  # For debugging
                })
            
            # Sort by final score (highest first) and limit results
            processed_results.sort(key=lambda x: x['similarity_score'], reverse=True)
            final_results = processed_results[:limit]
            
            logger.info(f"🔍 Search for '{query}': {len(final_results)} unique results (filtered from {len(processed_results)} total)")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    async def ask_question(self, question: str, doc_id: str = None) -> str:
        """
        Ask a question about documents using AI.
        
        Args:
            question: Question to ask
            doc_id: Optional specific document ID to query
        
        Returns:
            AI-generated answer
        """
        try:
            # Search for relevant content
            search_results = await self.search_document(question, doc_id, limit=3)
            
            if not search_results:
                return "I couldn't find any relevant information in the documents to answer your question."
            
            # Prepare context from search results
            context_parts = []
            for result in search_results:
                doc_name = result['metadata'].get('doc_name', 'Unknown Document')
                content = result['content']
                context_parts.append(f"From {doc_name}:\n{content}")
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Create prompt for GPT
            prompt = f"""Based on the following document content, please answer the question accurately and helpfully.

QUESTION: {question}

RELEVANT DOCUMENT CONTENT:
{context}

Please provide a comprehensive answer based on the information above. If the information is insufficient to fully answer the question, please say so."""
            
            # Call OpenAI GPT
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided document content. Be accurate and cite specific information when possible."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=int(os.getenv('MAX_TOKENS', '4000')),
                temperature=float(os.getenv('TEMPERATURE', '0.1'))
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Q&A failed: {e}")
            raise
    
    async def summarize_document(self, doc_id: str) -> str:
        """
        Generate a summary of a specific document.
        
        Args:
            doc_id: Document ID to summarize
        
        Returns:
            AI-generated summary
        """
        try:
            # Get all chunks for the document
            results = self.collection.query(
                query_embeddings=None,
                n_results=100,  # Get many chunks
                where={"doc_id": doc_id}
            )
            
            if not results['documents'] or not results['documents'][0]:
                raise ValueError(f"Document {doc_id} not found")
            
            # Combine chunks
            doc_chunks = results['documents'][0]
            doc_metadata = results['metadatas'][0][0]  # Get metadata from first chunk
            doc_name = doc_metadata.get('doc_name', 'Unknown Document')
            
            # Combine text (limit to reasonable length for GPT)
            combined_text = "\n\n".join(doc_chunks[:20])  # Limit to first 20 chunks
            
            # Create summarization prompt
            prompt = f"""Please provide a comprehensive summary of the following document:

DOCUMENT: {doc_name}

CONTENT:
{combined_text}

Please provide:
1. A brief overview of the document's main purpose
2. Key points and findings
3. Important conclusions or recommendations
4. Any notable data or statistics mentioned

Make the summary clear, concise, and well-structured."""
            
            # Call OpenAI GPT for summarization
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are an expert at creating comprehensive, well-structured document summaries. Focus on extracting the most important information and presenting it clearly."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=int(os.getenv('MAX_TOKENS', '4000')),
                temperature=float(os.getenv('TEMPERATURE', '0.1'))
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise
    
    def get_documents(self) -> List[Dict[str, Any]]:
        """
        Get list of all processed documents.
        
        Returns:
            List of document metadata
        """
        try:
            documents = []
            
            # Read metadata files from STABLE documents directory
            for metadata_file in self.documents_dir.glob("*_metadata.json"):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        documents.append(metadata)
                except Exception as e:
                    logger.warning(f"Failed to read metadata file {metadata_file}: {e}")
            
            # Sort by processing time (newest first)
            documents.sort(key=lambda x: x.get('processed_at', ''), reverse=True)
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to get documents list: {e}")
            return []

# Testing function
async def main():
    """Test the document processor."""
    processor = DocumentProcessor()
    
    # Test document processing
    test_pdf = "../test_documents/Plan.pdf"
    if Path(test_pdf).exists():
        print(f"\n🧪 Testing with: {test_pdf}")
        doc_id = await processor.process_document(test_pdf, "Test Plan Document")
        print(f"✅ Processed document: {doc_id}")
        
        # Test search
        results = await processor.search_document("main objectives")
        print(f"🔍 Search results: {len(results)}")
        
        # Test Q&A
        answer = await processor.ask_question("What are the key goals?")
        print(f"❓ Q&A Answer: {answer[:100]}...")
    else:
        print(f"⚠️ Test file not found: {test_pdf}")

if __name__ == "__main__":
    asyncio.run(main()) 