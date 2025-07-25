#!/usr/bin/env python3
"""
Web UI for Smart Document Analyzer
==================================

Simple Flask web interface for document processing and AI interaction.
"""

import os
import sys
import asyncio
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename
import json

# Add parent directory to path to import our modules
sys.path.append(str(Path(__file__).parent.parent))
from server.document_processor import DocumentProcessor

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'

# Configuration
UPLOAD_FOLDER = Path(__file__).parent.parent / 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Ensure upload directory exists
UPLOAD_FOLDER.mkdir(exist_ok=True)

# Global document processor
doc_processor = None

def init_processor():
    """Initialize the document processor."""
    global doc_processor
    try:
        doc_processor = DocumentProcessor()
        return True
    except Exception as e:
        print(f"Failed to initialize document processor: {e}")
        return False

def allowed_file(filename):
    """Check if uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main dashboard page."""
    docs = []
    if doc_processor:
        docs = doc_processor.get_documents()
    return render_template('index.html', documents=docs)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and processing."""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = app.config['UPLOAD_FOLDER'] / filename
        file.save(filepath)
        
        # Process the document
        doc_name = request.form.get('doc_name', filename.replace('.pdf', ''))
        
        async def process_doc():
            try:
                doc_id = await doc_processor.process_document(str(filepath), doc_name)
                return doc_id
            except Exception as e:
                return str(e)
        
        result = asyncio.run(process_doc())
        
        if result and not result.startswith('Error'):
            flash(f'Document "{doc_name}" processed successfully!')
        else:
            flash(f'Error processing document: {result}')
    else:
        flash('Invalid file type. Please upload a PDF.')
    
    return redirect(url_for('index'))

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests."""
    query = request.form.get('query', '').strip()
    if not query:
        return jsonify({'error': 'No search query provided'})
    
    async def do_search():
        try:
            results = await doc_processor.search_document(query, limit=5)
            return results
        except Exception as e:
            return {'error': str(e)}
    
    results = asyncio.run(do_search())
    return jsonify(results)

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle Q&A requests."""
    question = request.form.get('question', '').strip()
    if not question:
        return jsonify({'error': 'No question provided'})
    
    async def do_ask():
        try:
            answer = await doc_processor.ask_question(question)
            return {'answer': answer}
        except Exception as e:
            return {'error': str(e)}
    
    result = asyncio.run(do_ask())
    return jsonify(result)

@app.route('/documents')
def list_documents():
    """API endpoint to list all documents."""
    if not doc_processor:
        return jsonify({'error': 'Document processor not initialized'})
    
    docs = doc_processor.get_documents()
    return jsonify(docs)

@app.route('/api/search')
def api_search():
    """API endpoint for search (GET request)."""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify({'error': 'No search query provided'})
    
    async def do_search():
        try:
            results = await doc_processor.search_document(query, limit=5)
            formatted_results = []
            for result in results:
                # Create focused preview that shows relevant parts
                content = result['content']
                query_words = query.lower().split()
                content_lower = content.lower()
                
                # Find the best position based on query matches
                best_start = 0
                best_score = 0
                
                # Check every 50-character position
                for start in range(0, len(content), 50):
                    end = min(start + 400, len(content))
                    snippet = content_lower[start:end]
                    
                    # Score based on query word occurrences
                    score = sum(snippet.count(word) for word in query_words if len(word) > 2)
                    
                    if score > best_score:
                        best_score = score
                        best_start = start
                
                # Extract focused preview
                end_pos = min(best_start + 600, len(content))
                focused_preview = content[best_start:end_pos]
                
                if best_start > 0:
                    focused_preview = "..." + focused_preview
                if end_pos < len(content):
                    focused_preview = focused_preview + "..."
                
                formatted_results.append({
                    'document_name': result['metadata'].get('doc_name', 'Unknown'),
                    'relevance_score': round(result['similarity_score'] * 100, 1),
                    'content': content,  # Full content for JavaScript to use
                    'focused_preview': focused_preview
                })
            return {'query': query, 'results': formatted_results}
        except Exception as e:
            return {'error': str(e)}
    
    result = asyncio.run(do_search())
    return jsonify(result)

def find_available_port():
    """Find an available port starting from 5001."""
    import socket
    for port in range(5001, 5010):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return 5001  # fallback

if __name__ == '__main__':
    print("🚀 Initializing Smart Document Analyzer Web UI...")
    
    if init_processor():
        print("✅ Document processor initialized successfully!")
        
        # Find an available port (skip 5000 as it's used by macOS)
        port = find_available_port()
        
        print(f"🌐 Starting web server at http://localhost:{port}")
        print("📄 Upload PDFs and interact with AI through the web interface")
        print("")
        print("🎯 Open your browser and go to:")
        print(f"   👉 http://localhost:{port}")
        print("")
        
        try:
            app.run(debug=True, host='0.0.0.0', port=port, use_reloader=False)
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            print("💡 Try restarting your terminal or killing Python processes")
    else:
        print("❌ Failed to initialize document processor")
        print("💡 Make sure OPENAI_API_KEY is set in .env file") 