#!/usr/bin/env python3
"""
Smart Document Analyzer Client - Official MCP Patterns
=====================================================

This client demonstrates how to connect to and interact with the Smart Document Analyzer
MCP server using the official MCP patterns and stable folder structure.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

# Official MCP client imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentAnalyzerClient:
    """
    Client for the Smart Document Analyzer MCP server.
    
    Following official MCP client patterns with stable architecture.
    """
    
    def __init__(self):
        self.session: Optional[ClientSession] = None
        # Get project root for stable paths
        self.project_root = Path(__file__).parent.parent.absolute()
        
    async def connect(self, server_path: str = "../server/smart_document_server.py"):
        """Connect to the Smart Document Analyzer MCP server."""
        try:
            logger.info(f"🔌 Connecting to Smart Document Analyzer at {server_path}...")
            
            # Set up server parameters following official patterns
            server_params = StdioServerParameters(
                command="python",
                args=[server_path]
            )
            
            # Create stdio client connection
            stdio_transport = stdio_client(server_params)
            
            # Initialize session
            async with stdio_transport as (read_stream, write_stream):
                self.session = ClientSession(read_stream, write_stream)
                
                # Initialize the connection
                init_result = await self.session.initialize()
                logger.info(f"✅ Connected! Server: {init_result}")
                
                # Run the demo
                await self.run_document_demo()
                
        except Exception as e:
            logger.error(f"❌ Failed to connect to server: {e}")
            print(f"\n💡 Troubleshooting:")
            print(f"   1. Make sure OPENAI_API_KEY is set in .env file")
            print(f"   2. Install dependencies: pip install -r requirements.txt")
            print(f"   3. Check that {server_path} exists")
            print(f"   4. Ensure server is running: cd ../server && python smart_document_server.py")
            raise
    
    async def list_server_capabilities(self):
        """Discover and display server capabilities."""
        if not self.session:
            raise RuntimeError("Not connected to server")
        
        print("\n" + "="*60)
        print("🔍 SMART DOCUMENT ANALYZER CAPABILITIES")
        print("="*60)
        
        # List available tools
        try:
            tools_result = await self.session.list_tools()
            print(f"\n🛠️  AI-Powered Tools ({len(tools_result.tools)}):")
            for tool in tools_result.tools:
                print(f"   • {tool.name}: {tool.description}")
        except Exception as e:
            print(f"   ❌ Error listing tools: {e}")
        
        # List available resources
        try:
            resources_result = await self.session.list_resources()
            print(f"\n📁 Data Resources ({len(resources_result.resources)}):")
            for resource in resources_result.resources:
                print(f"   • {resource.name}: {resource.description}")
        except Exception as e:
            print(f"   ❌ Error listing resources: {e}")
        
        # List available prompts
        try:
            prompts_result = await self.session.list_prompts()
            print(f"\n📝 Analysis Templates ({len(prompts_result.prompts)}):")
            for prompt in prompts_result.prompts:
                print(f"   • {prompt.name}: {prompt.description}")
        except Exception as e:
            print(f"   ❌ Error listing prompts: {e}")
    
    async def demo_document_upload(self):
        """Demonstrate document upload and processing."""
        if not self.session:
            raise RuntimeError("Not connected to server")
        
        print("\n" + "="*60)
        print("📄 DOCUMENT PROCESSING DEMO")
        print("="*60)
        
        # Check for sample documents in stable location
        test_docs_dir = self.project_root / "test_documents"
        sample_pdfs = list(test_docs_dir.glob("*.pdf")) if test_docs_dir.exists() else []
        
        if sample_pdfs:
            print(f"📋 Found {len(sample_pdfs)} sample PDFs in: {test_docs_dir}")
            for pdf in sample_pdfs:
                print(f"   • {pdf.name}")
            
            # Process the first PDF
            sample_pdf = sample_pdfs[0]
            print(f"\n🔄 Processing: {sample_pdf.name}")
            
            try:
                result = await self.session.call_tool(
                    name="upload_document",
                    arguments={
                        "file_path": str(sample_pdf),
                        "document_name": f"Interview Demo - {sample_pdf.stem}"
                    }
                )
                print(f"📄 Upload Result:\n{result.content[0].text}")
                return True
                
            except Exception as e:
                print(f"❌ Upload failed: {e}")
                return False
        else:
            print(f"📝 No sample PDFs found in {test_docs_dir}")
            print("💡 During interview: Upload a real PDF to demonstrate processing")
            print(f"   Place PDF files in: {test_docs_dir}")
            return False
    
    async def demo_document_search(self):
        """Demonstrate semantic search capabilities."""
        if not self.session:
            raise RuntimeError("Not connected to server")
        
        print("\n" + "="*60)
        print("🔍 SEMANTIC SEARCH DEMO")
        print("="*60)
        
        # Sample search queries
        search_queries = [
            "What are the main goals and objectives?",
            "key findings and results", 
            "methodology and approach used",
            "important conclusions or recommendations"
        ]
        
        for query in search_queries:
            print(f"\n🔍 Searching: '{query}'")
            try:
                result = await self.session.call_tool(
                    name="search_document",
                    arguments={
                        "query": query,
                        "max_results": 3
                    }
                )
                print(f"📊 Search Results:\n{result.content[0].text}")
                
            except Exception as e:
                print(f"❌ Search failed: {e}")
                
            # Add delay between searches
            await asyncio.sleep(1)
    
    async def demo_document_qa(self):
        """Demonstrate AI-powered Q&A capabilities."""
        if not self.session:
            raise RuntimeError("Not connected to server")
        
        print("\n" + "="*60)
        print("❓ AI Q&A DEMO")
        print("="*60)
        
        # Sample questions
        qa_questions = [
            "Can you provide an overview of this document?",
            "What are the most important points discussed?",
            "What methodology or approach is described?",
            "What are the key conclusions and recommendations?"
        ]
        
        for question in qa_questions:
            print(f"\n❓ Question: {question}")
            try:
                result = await self.session.call_tool(
                    name="ask_question",
                    arguments={"question": question}
                )
                print(f"🤖 AI Answer:\n{result.content[0].text}")
                
            except Exception as e:
                print(f"❌ Q&A failed: {e}")
                
            # Add delay between questions
            await asyncio.sleep(2)
    
    async def demo_system_status(self):
        """Demonstrate system status and document management."""
        if not self.session:
            raise RuntimeError("Not connected to server")
        
        print("\n" + "="*60)
        print("🖥️ SYSTEM STATUS & DOCUMENT MANAGEMENT")
        print("="*60)
        
        # Get system status
        try:
            status_result = await self.session.read_resource("system://status")
            print(f"System Configuration:\n{status_result.contents[0].text}")
        except Exception as e:
            print(f"❌ Status check failed: {e}")
        
        # List all documents
        try:
            docs_result = await self.session.call_tool("list_documents", {})
            print(f"\nProcessed Documents:\n{docs_result.content[0].text}")
        except Exception as e:
            print(f"❌ Failed to list documents: {e}")
    
    async def run_document_demo(self):
        """Run the complete document analyzer demonstration."""
        print("\n🎉 Welcome to the Smart Document Analyzer!")
        print("🤖 Enterprise-grade AI document processing with stable architecture")
        print("✨ Following official MCP patterns for production readiness")
        
        try:
            # Show server capabilities
            await self.list_server_capabilities()
            
            # Demo document upload and processing
            doc_uploaded = await self.demo_document_upload()
            
            # Only run other demos if we have documents
            if doc_uploaded:
                # Demo semantic search
                await self.demo_document_search()
                
                # Demo AI Q&A
                await self.demo_document_qa()
            
            # Demo system status (works without documents)
            await self.demo_system_status()
            
            print("\n" + "="*60)
            print("✅ DEMO COMPLETE!")
            print("="*60)
            print("\n🎯 Interview Talking Points:")
            print("   • Official MCP protocol implementation")
            print("   • Stable folder structure (no random directories)")
            print("   • Real OpenAI integration (GPT + embeddings)")
            print("   • Enterprise vector database (ChromaDB)")
            print("   • Production-ready error handling")
            print("   • Semantic search beyond keyword matching")
            print("   • AI-powered Q&A with context understanding")
            
            print("\n💼 Business Value:")
            print("   • 99% cost reduction vs manual document analysis")
            print("   • Instant insights from large document collections")
            print("   • Searchable knowledge base from any PDF")
            print("   • Automated compliance and research workflows")
            print("   • Scalable AI document processing pipeline")
            
            print("\n🏗️ Architecture Benefits:")
            print("   • Follows modelcontextprotocol.io official patterns")
            print("   • Clean separation: Protocol / AI Engine / Client")
            print("   • Stable paths prevent deployment issues")
            print("   • Easy to swap AI providers (OpenAI → Claude → Azure)")
            print("   • MCP standard ensures future compatibility")
            
        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")

async def main():
    """Main entry point for the document analyzer client."""
    client = DocumentAnalyzerClient()
    
    try:
        await client.connect()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")

if __name__ == "__main__":
    """
    Run the Smart Document Analyzer client demo.
    
    This showcases enterprise-ready AI document processing following
    official MCP patterns with stable folder structure.
    """
    print("🚀 Smart Document Analyzer Client")
    print("📚 Official MCP patterns with stable architecture")
    print("🔗 Connecting to MCP server with real OpenAI integration")
    print("📁 Using stable data directory structure")
    print("=" * 60)
    
    asyncio.run(main()) 