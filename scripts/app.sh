#!/bin/bash
# Movie Genie Application Manager
# DEPRECATED: This script is no longer recommended for use.
#
# For production deployment, use Docker:
#   dvc repro
#   docker-compose build
#   docker-compose up -d
#
# For local development, run services directly:
#   cd movie_genie/frontend && npm run dev
#   python scripts/start_server.py
#
# This script remains for backward compatibility only.
# See README.md for updated workflows.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
BACKEND_PORT=${FLASK_PORT:-5001}
FRONTEND_PORT=${VITE_PORT:-5173}
MLFLOW_PORT=${MLFLOW_PORT:-5002}
DOCS_PORT=${DOCS_PORT:-8000}

# PID file directory
PID_DIR="$PROJECT_ROOT/.pids"
mkdir -p "$PID_DIR"

#######################################
# Helper Functions
#######################################

print_header() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  🎬 Movie Genie - $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port in use
    else
        return 1  # Port free
    fi
}

wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=30
    local attempt=1

    echo -n "Waiting for $name to start"
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            echo ""
            print_success "$name is ready"
            return 0
        fi
        echo -n "."
        sleep 1
        ((attempt++))
    done
    echo ""
    print_error "$name failed to start"
    return 1
}

#######################################
# Service Management
#######################################

start_backend() {
    print_info "Starting backend server on port $BACKEND_PORT..."

    if check_port $BACKEND_PORT; then
        print_warning "Backend already running on port $BACKEND_PORT"
        return 0
    fi

    FLASK_PORT=$BACKEND_PORT FLASK_ENV=development \
        nohup python scripts/start_server.py > logs/backend.log 2>&1 &

    echo $! > "$PID_DIR/backend.pid"
    wait_for_service "http://127.0.0.1:$BACKEND_PORT/api/health" "Backend"
}

start_frontend() {
    print_info "Starting frontend dev server on port $FRONTEND_PORT..."

    if check_port $FRONTEND_PORT; then
        print_warning "Frontend already running on port $FRONTEND_PORT"
        return 0
    fi

    cd movie_genie/frontend
    nohup npm run dev > ../../logs/frontend.log 2>&1 &
    echo $! > "$PID_DIR/frontend.pid"
    cd "$PROJECT_ROOT"

    sleep 3
    print_success "Frontend started"
}

start_mlflow() {
    print_info "Starting MLflow UI on port $MLFLOW_PORT..."

    if check_port $MLFLOW_PORT; then
        print_warning "MLflow UI already running on port $MLFLOW_PORT"
        return 0
    fi

    nohup mlflow ui --host 127.0.0.1 --port $MLFLOW_PORT > logs/mlflow.log 2>&1 &
    echo $! > "$PID_DIR/mlflow.pid"

    wait_for_service "http://127.0.0.1:$MLFLOW_PORT" "MLflow UI"
}

start_docs() {
    print_info "Starting documentation server on port $DOCS_PORT..."

    if check_port $DOCS_PORT; then
        print_warning "Documentation already running on port $DOCS_PORT"
        return 0
    fi

    nohup mkdocs serve > logs/docs.log 2>&1 &
    echo $! > "$PID_DIR/docs.pid"

    wait_for_service "http://127.0.0.1:$DOCS_PORT" "Documentation"
}

stop_service() {
    local service=$1
    local pid_file="$PID_DIR/$service.pid"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if ps -p $pid > /dev/null 2>&1; then
            print_info "Stopping $service (PID: $pid)..."
            kill $pid 2>/dev/null || true
            sleep 1
            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                kill -9 $pid 2>/dev/null || true
            fi
            print_success "$service stopped"
        fi
        rm -f "$pid_file"
    fi
}

#######################################
# Main Commands
#######################################

cmd_setup() {
    print_header "Initial Setup"

    print_info "Installing Python dependencies..."
    pip install -e ".[llm]"

    print_info "Installing frontend dependencies..."
    cd movie_genie/frontend && npm install && cd "$PROJECT_ROOT"

    print_info "Creating necessary directories..."
    mkdir -p logs data/raw data/processed models results

    print_success "Setup complete!"
}

cmd_pipeline() {
    print_header "Running Data Pipeline"

    print_info "Running DVC pipeline (data processing + training)..."
    dvc repro integrated_evaluation

    print_success "Pipeline complete!"
}

cmd_build() {
    print_header "Building Frontend"

    print_info "Building production frontend..."
    cd movie_genie/frontend && npm run build && cd "$PROJECT_ROOT"

    print_info "Copying frontend to backend..."
    dvc repro frontend_build

    print_success "Build complete!"
}

cmd_start() {
    print_header "Starting All Services"

    # Create logs directory
    mkdir -p logs

    # Start services in order
    start_backend
    start_frontend
    start_mlflow
    start_docs

    echo ""
    print_success "All services started!"
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  🚀 Movie Genie is Ready!${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    echo "📱 Application:     http://127.0.0.1:$FRONTEND_PORT"
    echo "🔧 Backend API:     http://127.0.0.1:$BACKEND_PORT/api"
    echo "📊 MLflow UI:       http://127.0.0.1:$MLFLOW_PORT"
    echo "📚 Documentation:   http://127.0.0.1:$DOCS_PORT"
    echo ""
    echo "📋 Logs:"
    echo "   • Backend:       tail -f logs/backend.log"
    echo "   • Frontend:      tail -f logs/frontend.log"
    echo "   • MLflow:        tail -f logs/mlflow.log"
    echo "   • Documentation: tail -f logs/docs.log"
    echo ""
    echo "To stop all services: ./scripts/app.sh stop"
    echo ""
}

cmd_stop() {
    print_header "Stopping All Services"

    stop_service "backend"
    stop_service "frontend"
    stop_service "mlflow"
    stop_service "docs"

    # Clean up any orphaned processes
    pkill -f "flask run" 2>/dev/null || true
    pkill -f "vite" 2>/dev/null || true
    pkill -f "mlflow ui" 2>/dev/null || true
    pkill -f "mkdocs serve" 2>/dev/null || true

    print_success "All services stopped"
}

cmd_restart() {
    cmd_stop
    sleep 2
    cmd_start
}

cmd_status() {
    print_header "Service Status"
    echo ""

    check_service_status() {
        local name=$1
        local port=$2
        local pid_file="$PID_DIR/$3.pid"

        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if ps -p $pid > /dev/null 2>&1; then
                echo -e "${GREEN}✅ $name${NC} - Running (PID: $pid, Port: $port)"
                return 0
            fi
        fi

        if check_port $port; then
            echo -e "${YELLOW}⚠️  $name${NC} - Running (port $port in use, no PID file)"
            return 0
        fi

        echo -e "${RED}❌ $name${NC} - Stopped"
        return 1
    }

    check_service_status "Backend    " "$BACKEND_PORT" "backend"
    check_service_status "Frontend   " "$FRONTEND_PORT" "frontend"
    check_service_status "MLflow UI  " "$MLFLOW_PORT" "mlflow"
    check_service_status "Documentation" "$DOCS_PORT" "docs"
    echo ""
}

cmd_logs() {
    local service=${1:-all}

    case $service in
        backend)
            tail -f logs/backend.log
            ;;
        frontend)
            tail -f logs/frontend.log
            ;;
        mlflow)
            tail -f logs/mlflow.log
            ;;
        docs)
            tail -f logs/docs.log
            ;;
        all)
            tail -f logs/*.log
            ;;
        *)
            print_error "Unknown service: $service"
            echo "Available services: backend, frontend, mlflow, docs, all"
            exit 1
            ;;
    esac
}

cmd_dev() {
    print_header "Development Mode"

    print_info "Starting backend and frontend in development mode..."
    start_backend
    start_frontend

    echo ""
    print_success "Development services started!"
    echo ""
    echo "Frontend (dev): http://127.0.0.1:$FRONTEND_PORT"
    echo "Backend API:    http://127.0.0.1:$BACKEND_PORT/api"
    echo ""
}

cmd_full() {
    print_header "Full Pipeline + Services"

    # Run full pipeline
    cmd_pipeline

    # Build frontend
    cmd_build

    # Start all services
    cmd_start
}

cmd_help() {
    cat << EOF
Movie Genie Application Manager

Usage: ./scripts/app.sh [command] [options]

Commands:
  setup         Install dependencies and initialize project
  pipeline      Run DVC pipeline (data processing + training)
  build         Build production frontend

  start         Start all services (backend, frontend, mlflow, docs)
  stop          Stop all services
  restart       Restart all services
  dev           Start only backend + frontend (development mode)

  status        Show status of all services
  logs [name]   Show logs (backend|frontend|mlflow|docs|all)

  full          Run complete pipeline + build + start services
  help          Show this help message

Examples:
  ./scripts/app.sh setup          # Initial setup
  ./scripts/app.sh pipeline       # Run data pipeline
  ./scripts/app.sh dev            # Start dev environment
  ./scripts/app.sh start          # Start all services
  ./scripts/app.sh logs backend   # View backend logs
  ./scripts/app.sh stop           # Stop everything

Environment Variables:
  FLASK_PORT      Backend port (default: 5001)
  VITE_PORT       Frontend port (default: 5173)
  MLFLOW_PORT     MLflow UI port (default: 5002)
  DOCS_PORT       Documentation port (default: 8000)

EOF
}

#######################################
# Main Entry Point
#######################################

main() {
    local command=${1:-help}

    case $command in
        setup)
            cmd_setup
            ;;
        pipeline)
            cmd_pipeline
            ;;
        build)
            cmd_build
            ;;
        start)
            cmd_start
            ;;
        stop)
            cmd_stop
            ;;
        restart)
            cmd_restart
            ;;
        status)
            cmd_status
            ;;
        logs)
            cmd_logs "${2:-all}"
            ;;
        dev)
            cmd_dev
            ;;
        full)
            cmd_full
            ;;
        help|--help|-h)
            cmd_help
            ;;
        *)
            print_error "Unknown command: $command"
            echo ""
            cmd_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
