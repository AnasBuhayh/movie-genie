#!/bin/bash

# Serve MkDocs Documentation
# This script starts the MkDocs development server

echo "🎬 Starting Movie Genie Documentation Server..."
echo ""
echo "📚 Documentation will be available at: http://127.0.0.1:8000"
echo "🔧 Press Ctrl+C to stop"
echo ""

# Check if mkdocs is installed
if ! command -v mkdocs &> /dev/null; then
    echo "❌ MkDocs not found. Installing..."
    pip install mkdocs-material
fi

# Start MkDocs server
mkdocs serve
