#!/usr/bin/env python3
"""
Smart Document Analyzer MCP Server - Official MCP Pattern
========================================================

Following the official MCP quickstart patterns from modelcontextprotocol.io
with stable folder structure and professional organization.

Features:
- Official FastMCP API usage
- Stable paths (no random folder creation)
- Real AI integration (OpenAI GPT + embeddings)
- Enterprise-grade document processing
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

# Official MCP imports (from the real MCP Python SDK)
from mcp.server.fastmcp import FastMCP

# Our document processor (with fixed paths)
from document_processor import DocumentProcessor

# Initialize document processor with stable paths
try:
    doc_processor = DocumentProcessor()
    print("✅ Document processor initialized with stable folder structure")
except Exception as e:
    print(f"❌ Failed to initialize document processor: {e}")
    exit(1)

# Create MCP server following official patterns
mcp = FastMCP("Smart Document Analyzer")

# ============================================================================
# CORE TOOLS (Only the essentials!)
# ============================================================================

@mcp.tool()
async def upload_document(file_path: str, doc_name: str = "") -> str:
    """
    Process a PDF document with AI analysis and store it for future queries.
    
    Args:
        file_path: Path to the PDF file to process
        doc_name: Optional custom name for the document
    
    Returns:
        Success message with document ID
    """
    if not doc_processor:
        return "❌ Document processor not initialized. Check OpenAI API key configuration."
    
    try:
        print(f"📄 Processing document: {file_path}")
        doc_id = await doc_processor.process_document(file_path, doc_name)
        return f"✅ Document processed successfully! Document ID: {doc_id}"
    except Exception as e:
        return f"❌ Processing error: {str(e)}"

@mcp.tool()
async def search_document(query: str, document_id: str = "", max_results: int = 5) -> dict:
    """
    Semantic search through documents using AI embeddings.
    
    Args:
        query: What to search for (natural language)
        document_id: Optional specific document ID (empty = search all)
        max_results: Maximum number of results
    
    Returns:
        Structured search results with relevance scores
    """
    if not doc_processor:
        return {"error": "Document processor not initialized. Check OpenAI API key configuration."}
    
    try:
        # Perform AI-powered semantic search
        results = await doc_processor.search_document(
            query=query,
            doc_id=document_id if document_id else None,
            limit=max_results
        )
        
        if not results:
            return {
                "query": query,
                "results": [],
                "message": f"No relevant content found for: '{query}'"
            }
        
        # Return clean, structured results
        formatted_results = []
        for i, result in enumerate(results, 1):
            score_percent = result['similarity_score'] * 100
            doc_name = result['metadata'].get('doc_name', 'Unknown')
            chunk_index = result['metadata'].get('chunk_index', 0)
            content = result['content']
            
            # Create focused preview that shows relevant parts
            smart_preview = create_smart_preview(content, query, max_length=300)
            
            formatted_results.append({
                "rank": i,
                "relevance_score": round(score_percent, 1),
                "document_name": doc_name,
                "chunk_section": f"Chunk {chunk_index + 1}",
                "content": content,
                "focused_preview": smart_preview
            })
        
        return {
            "query": query,
            "total_results": len(formatted_results),
            "results": formatted_results
        }
        
    except Exception as e:
        return {"error": f"Search error: {str(e)}"}

def create_smart_preview(content: str, query: str, max_length: int = 300) -> str:
    """Create a focused preview that shows the most relevant part for the query."""
    query_words = [word.lower() for word in query.split() if len(word) > 2]
    
    if not query_words:
        return content[:max_length] + ("..." if len(content) > max_length else "")
    
    # Find the position with the most query word matches
    content_lower = content.lower()
    best_position = 0
    best_score = 0
    
    # Check every 50-character position
    for start in range(0, len(content), 50):
        end = min(start + max_length, len(content))
        snippet = content_lower[start:end]
        
        # Score based on query word occurrences
        score = sum(snippet.count(word) for word in query_words)
        
        if score > best_score:
            best_score = score
            best_position = start
    
    # Extract the best snippet
    start = best_position
    end = min(start + max_length, len(content))
    
    # Try to start at a word boundary
    while start > 0 and content[start] != ' ':
        start -= 1
    
    # Try to end at a word boundary  
    while end < len(content) and content[end] != ' ':
        end += 1
    
    preview = content[start:end].strip()
    
    # Add ellipsis indicators
    if start > 0:
        preview = "..." + preview
    if end < len(content):
        preview = preview + "..."
    
    return preview

@mcp.tool()
async def ask_question(question: str) -> str:
    """
    Ask an AI question about the processed documents using GPT-4o-mini.
    
    Args:
        question: Natural language question about the documents
    
    Returns:
        AI-generated answer based on document content
    """
    if not doc_processor:
        return "❌ Document processor not initialized. Check OpenAI API key configuration."
    
    try:
        print(f"❓ Processing question: {question}")
        answer = await doc_processor.ask_question(question)
        return f"🤖 AI Answer: {answer}"
    except Exception as e:
        return f"❌ Q&A error: {str(e)}"

@mcp.tool()
async def list_documents() -> str:
    """
    List all processed documents with metadata.
    
    Returns:
        Formatted list of all documents in the system
    """
    if not doc_processor:
        return "❌ Document processor not initialized. Check OpenAI API key configuration."
    
    try:
        documents = doc_processor.get_documents()
        if not documents:
            return "📂 No documents processed yet. Upload a PDF to get started!"
        
        result = f"📚 Processed Documents ({len(documents)}):\n" + "="*50 + "\n"
        for doc in documents:
            result += f"\n📄 {doc['name']}\n"
            result += f"   🆔 ID: {doc['doc_id']}\n"
            result += f"   📊 Chunks: {doc['chunk_count']}\n"
            result += f"   ⏰ Processed: {doc['processed_at']}\n"
        
        return result
    except Exception as e:
        return f"❌ Error listing documents: {str(e)}"

# ============================================================================
# RESOURCES (Read-only data endpoints)
# ============================================================================

@mcp.resource("documents://all")
async def get_all_documents() -> str:
    """Get a complete list of processed documents."""
    if not doc_processor:
        return "❌ Document processor not initialized."
    try:
        documents = doc_processor.get_documents()
        if not documents:
            return "📂 No documents in system yet."
        return json.dumps(documents, indent=2)
    except Exception as e:
        return f"❌ Error accessing documents: {str(e)}"

@mcp.resource("system://status")
async def get_system_status() -> str:
    """Get current system status and configuration."""
    if not doc_processor:
        return json.dumps({"status": "error", "message": "Document processor not initialized"})
    try:
        documents = doc_processor.get_documents()
        total_docs = len(documents)
        total_chunks = sum(doc.get('chunk_count', 0) for doc in documents)

        status = {
            "status": "operational",
            "ai_integration": "OpenAI GPT-4o-mini + Embeddings",
            "vector_database": "ChromaDB",
            "storage_location": str(doc_processor.data_dir),
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "embedding_model": doc_processor.embedding_model,
            "chat_model": doc_processor.chat_model
        }
        return json.dumps(status, indent=2)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})

# ============================================================================
# PROMPTS (Reusable templates)
# ============================================================================

@mcp.prompt("document_analysis_prompt")
async def document_analysis_prompt() -> str:
    """Template for analyzing documents with AI."""
    return """
You are an expert document analyst. Analyze the provided document content and:

1. Identify key themes and topics
2. Extract important facts and figures
3. Summarize main points clearly
4. Answer specific questions about the content

Be precise, factual, and helpful in your analysis.
"""

# ============================================================================
# SERVER STARTUP
# ============================================================================

if __name__ == "__main__":
    import json
    
    print("\n" + "="*60)
    print("🚀 Smart Document Analyzer MCP Server")
    print("📚 Official MCP patterns with real AI integration")
    print(f"📁 Stable data storage in: {doc_processor.data_dir}")
    print("✨ Simplified with core tools only")
    print("="*60)
    print("🛠️  Available Tools:")
    print("   • upload_document - Process PDFs with AI")
    print("   • search_document - Semantic search with embeddings")
    print("   • ask_question - AI Q&A with OpenAI GPT-4o-mini")
    print("   • list_documents - Show all processed documents")
    print("📁 Available Resources:")
    print("   • documents://all - All document metadata")
    print("   • system://status - System configuration")
    print("📝 Available Prompts:")
    print("   • document_analysis_prompt - Analysis templates")
    print("🎯 Server ready for MCP connections!")
    print("💡 Start client with: cd ../client && python document_client.py")
    print("="*60)
    
    # Run the MCP server
    mcp.run() 