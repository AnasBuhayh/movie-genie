#!/bin/bash
# Documentation management script for Movie Genie

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE} Movie Genie Documentation Tool${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
}

# Function to check if MkDocs is installed
check_mkdocs() {
    if ! command -v mkdocs &> /dev/null; then
        print_error "MkDocs is not installed. Installing..."
        pip install mkdocs-material mkdocs-git-revision-date-localized-plugin mkdocs-git-committers-plugin-2 mkdocs-minify-plugin
    else
        print_status "MkDocs is installed"
    fi
}

# Function to serve documentation locally
serve_docs() {
    print_status "Starting MkDocs development server..."
    print_status "Documentation will be available at: http://127.0.0.1:8000"
    print_status "Press Ctrl+C to stop the server"
    echo ""
    mkdocs serve
}

# Function to build documentation
build_docs() {
    print_status "Building documentation..."
    mkdocs build
    print_status "Documentation built successfully in site/ directory"
}

# Function to deploy to GitHub Pages
deploy_docs() {
    print_status "Deploying documentation to GitHub Pages..."
    mkdocs gh-deploy --force
    print_status "Documentation deployed successfully"
}

# Function to show help
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  serve    Start development server (default)"
    echo "  build    Build documentation"
    echo "  deploy   Deploy to GitHub Pages"
    echo "  check    Check MkDocs installation"
    echo "  help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              # Start development server"
    echo "  $0 serve        # Start development server"
    echo "  $0 build        # Build documentation"
    echo "  $0 deploy       # Deploy to GitHub Pages"
}

# Function to validate documentation
validate_docs() {
    print_status "Validating documentation structure..."

    # Check required files
    required_files=(
        "docs/index.md"
        "docs/getting-started/index.md"
        "docs/machine-learning/README.md"
        "mkdocs.yml"
    )

    for file in "${required_files[@]}"; do
        if [[ -f "$file" ]]; then
            print_status "✓ $file exists"
        else
            print_warning "✗ $file is missing"
        fi
    done

    # Check MkDocs configuration
    print_status "Checking MkDocs configuration..."
    if mkdocs build --quiet; then
        print_status "✓ MkDocs configuration is valid"
    else
        print_error "✗ MkDocs configuration has errors"
        return 1
    fi
}

# Main function
main() {
    print_header

    # Change to script directory
    cd "$(dirname "$0")/.."

    # Check if we're in the right directory
    if [[ ! -f "mkdocs.yml" ]]; then
        print_error "mkdocs.yml not found. Are you in the Movie Genie root directory?"
        exit 1
    fi

    # Parse command line arguments
    COMMAND=${1:-serve}

    case $COMMAND in
        serve|s)
            check_mkdocs
            validate_docs
            serve_docs
            ;;
        build|b)
            check_mkdocs
            validate_docs
            build_docs
            ;;
        deploy|d)
            check_mkdocs
            validate_docs
            build_docs
            deploy_docs
            ;;
        check|c)
            check_mkdocs
            validate_docs
            ;;
        help|h|-h|--help)
            show_help
            ;;
        *)
            print_error "Unknown command: $COMMAND"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"