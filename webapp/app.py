#!/usr/bin/env python3
"""
Web UI for Smart Document Analyzer
==================================

Simple Flask web interface for document processing and AI interaction.
"""

import os
import sys
import asyncio
import sqlite3
from pathlib import Path
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import uuid

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

# Database for search history
DB_PATH = Path(__file__).parent.parent / 'data' / 'search_history.db'

# Ensure upload directory exists
UPLOAD_FOLDER.mkdir(exist_ok=True)

def init_search_history_db():
    """Initialize the search history database."""
    try:
        # Ensure data directory exists
        DB_PATH.parent.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Create search_history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                query TEXT NOT NULL,
                search_type TEXT NOT NULL DEFAULT 'semantic',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                results_count INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Search history database initialized")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize search history database: {e}")
        return False

def get_user_session_id():
    """Get or create a user session ID."""
    if 'user_session_id' not in session:
        session['user_session_id'] = str(uuid.uuid4())
    return session['user_session_id']

def save_search_query(query: str, search_type: str = 'semantic', results_count: int = 0):
    """Save a search query to the database."""
    try:
        session_id = get_user_session_id()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO search_history (session_id, query, search_type, results_count)
            VALUES (?, ?, ?, ?)
        ''', (session_id, query, search_type, results_count))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Failed to save search query: {e}")
        return False

def get_search_history(limit: int = 10):
    """Get search history for the current user session."""
    try:
        session_id = get_user_session_id()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT query, search_type, timestamp, results_count
            FROM search_history
            WHERE session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (session_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        # Convert to list of dictionaries
        history = []
        for row in results:
            history.append({
                'query': row[0],
                'search_type': row[1],
                'timestamp': row[2],
                'results_count': row[3]
            })
        
        return history
    except Exception as e:
        print(f"Failed to get search history: {e}")
        return []

# Global document processor
doc_processor = None
current_model = 'gpt-4o-mini'  # Default model

def init_processor():
    """Initialize the document processor."""
    global doc_processor
    try:
        # Initialize with default model - we'll handle model switching in requests
        doc_processor = DocumentProcessor(chat_model=current_model)
        return True
    except Exception as e:
        print(f"Failed to initialize document processor: {e}")
        return False

def reinit_processor_with_model(model: str):
    """Reinitialize processor with a specific model."""
    global doc_processor, current_model
    try:
        current_model = model
        # Only set session if we're in a request context
        try:
            session['selected_model'] = model
        except RuntimeError:
            # Not in request context, skip session setting
            pass
        doc_processor = DocumentProcessor(chat_model=model)
        return True
    except Exception as e:
        print(f"Failed to reinitialize with model {model}: {e}")
        return False

def allowed_file(filename):
    """Check if uploaded file is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main dashboard page."""
    global current_model
    
    # Check if user has a model preference in session
    if 'selected_model' in session and session['selected_model'] != current_model:
        # User has a different model preference, reinitialize if needed
        preferred_model = session['selected_model']
        if preferred_model in DocumentProcessor.get_available_models():
            if not doc_processor or doc_processor.chat_model != preferred_model:
                reinit_processor_with_model(preferred_model)
    
    docs = []
    model_info = {}
    if doc_processor:
        docs = doc_processor.get_documents()
        model_info = doc_processor.get_current_model_info()
    
    # Get available models for the UI
    available_models = DocumentProcessor.get_available_models()
    
    # Get last uploaded summary if available
    last_summary = session.pop('last_uploaded_summary', None)
    
    # Get search history
    search_history = get_search_history(limit=10)
    
    return render_template('index.html', 
                         documents=docs, 
                         model_info=model_info,
                         available_models=available_models,
                         current_model=current_model,
                         last_summary=last_summary,
                         search_history=search_history)

@app.route('/search_history')
def get_search_history_api():
    """API endpoint to get search history."""
    limit = request.args.get('limit', 10, type=int)
    history = get_search_history(limit)
    
    return jsonify({
        'history': history,
        'total': len(history)
    })

@app.route('/clear_search_history', methods=['POST'])
def clear_search_history():
    """Clear search history for current user."""
    try:
        session_id = get_user_session_id()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM search_history WHERE session_id = ?', (session_id,))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Search history cleared'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

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
        
        # Process the document using original filename (without extension)
        doc_name = filename.replace('.pdf', '')
        
        async def process_doc():
            try:
                doc_id = await doc_processor.process_document(str(filepath), doc_name)
                
                # Generate immediate summary
                summary = await doc_processor.summarize_document(doc_id)
                
                # Create a concise 2-3 sentence summary
                summary_prompt = f"""Please create a very concise 2-3 sentence summary of this document analysis:

{summary}

Make it brief and informative, focusing on the main purpose and key points of the document."""
                
                response = doc_processor.openai_client.chat.completions.create(
                    model=doc_processor.chat_model,
                    messages=[
                        {"role": "system", "content": "You are an expert at creating concise document summaries. Provide exactly 2-3 sentences that capture the essence of the document."},
                        {"role": "user", "content": summary_prompt}
                    ],
                    max_tokens=150,
                    temperature=0.1
                )
                
                concise_summary = response.choices[0].message.content.strip()
                
                return {'doc_id': doc_id, 'summary': concise_summary}
            except Exception as e:
                return {'error': str(e)}
        
        result = asyncio.run(process_doc())
        
        if 'error' not in result:
            flash(f'✅ Document "{doc_name}" processed successfully!')
            session['last_uploaded_summary'] = {
                'doc_name': doc_name,
                'summary': result['summary'],
                'doc_id': result['doc_id']
            }
        else:
            flash(f'❌ Error processing document: {result["error"]}')
    else:
        flash('❌ Invalid file type. Please upload a PDF.')
    
    return redirect(url_for('index'))

@app.route('/search', methods=['POST'])
def search():
    """Handle search requests."""
    query = request.form.get('query', '').strip()
    if not query:
        return jsonify({'error': 'No search query provided'})
    
    async def do_search():
        try:
            results = await doc_processor.search_document(query, limit=3)
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
            
            # Save question to search history
            save_search_query(question, 'question', 1 if answer else 0)
            
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
            results = await doc_processor.search_document(query, limit=3)
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
            
            # Save search query to history
            save_search_query(query, 'semantic', len(formatted_results))
            
            return {'query': query, 'results': formatted_results}
        except Exception as e:
            return {'error': str(e)}
    
    result = asyncio.run(do_search())
    return jsonify(result)

@app.route('/models')
def get_models():
    """API endpoint to get available models."""
    models = DocumentProcessor.get_available_models()
    current_info = doc_processor.get_current_model_info() if doc_processor else {}
    
    return jsonify({
        'available_models': models,
        'current_model': current_info
    })

@app.route('/set_model', methods=['POST'])
def set_model():
    """Change the AI model."""
    global current_model
    
    model = request.form.get('model', '').strip()
    
    if not model:
        return jsonify({'error': 'No model specified'})
    
    # Check if model is available
    available_models = DocumentProcessor.get_available_models()
    if model not in available_models:
        return jsonify({'error': f'Model {model} not available'})
    
    # Reinitialize processor with new model
    if reinit_processor_with_model(model):
        current_model = model  # Ensure global variable is updated
        model_info = available_models[model]
        flash(f'✅ Switched to {model_info["name"]}')
        return jsonify({
            'success': True, 
            'model': model,
            'model_info': model_info,
            'message': f'Successfully switched to {model_info["name"]}'
        })
    else:
        return jsonify({'error': f'Failed to switch to model {model}'})

@app.route('/model_info')
def model_info():
    """Get current model information."""
    if not doc_processor:
        return jsonify({'error': 'Document processor not initialized'})
    
    # Get fresh model info from the processor
    model_info = doc_processor.get_current_model_info()
    
    # Also update global current_model to match processor
    global current_model
    current_model = model_info['model_id']
    
    return jsonify(model_info)

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
    
    # Initialize search history database first
    init_search_history_db()
    
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