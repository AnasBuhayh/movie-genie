# 🔧 Installation Guide

Complete step-by-step installation instructions for Movie Genie development environment.

## 📋 Prerequisites

### System Requirements
- **OS**: macOS, Linux, or Windows (WSL recommended)
- **Python**: 3.8 or higher
- **Node.js**: 16.0 or higher
- **Git**: Latest version
- **Memory**: 8GB+ RAM (for ML model training)
- **Storage**: 5GB+ free disk space

### Check Your System
```bash
# Check Python version
python --version  # Should be 3.8+

# Check Node.js version
node --version    # Should be 16+

# Check Git
git --version

# Check available memory
free -h  # Linux/macOS
```

---

## 🚀 Installation Steps

### Step 1: Clone the Repository
```bash
# Clone the project
git clone <repository-url>
cd movie-genie

# Verify the directory structure
ls -la
```

You should see:
```
movie_genie/
├── configs/
├── data/
├── docs/
├── movie_genie/
├── scripts/
├── dvc.yaml
├── params.yaml
└── pyproject.toml
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# OR
.venv\Scripts\activate     # Windows
```

### Step 3: Install Python Dependencies
```bash
# Install project with development dependencies
pip install -e ".[dev]"

# Verify installation
pip list | grep movie-genie
```

### Step 4: Install Frontend Dependencies
```bash
# Navigate to frontend directory
cd movie_genie/frontend

# Install Node.js dependencies
npm install

# Verify installation
npm list --depth=0

# Return to project root
cd ../..
```

### Step 5: Initialize DVC
```bash
# Initialize DVC (if not already done)
dvc init --no-scm  # Use if not using git
# OR
dvc init          # Use if using git

# Check DVC status
dvc status
```

---

## 🔧 Development Environment Setup

### Configure Environment Variables

#### Backend Environment
```bash
# Create backend environment file
cat > movie_genie/backend/.env << EOF
FLASK_ENV=development
FLASK_DEBUG=1
FLASK_PORT=5001
DATABASE_URL=sqlite:///movie_genie.db
LOG_LEVEL=DEBUG
EOF
```

#### Frontend Environment
```bash
# Create frontend environment file
cat > movie_genie/frontend/.env.development << EOF
VITE_API_URL=http://127.0.0.1:5001/api
VITE_USE_REAL_POPULAR=false
VITE_USE_REAL_SEARCH=false
VITE_USE_REAL_RECOMMENDATIONS=false
VITE_USE_REAL_MOVIE_DETAILS=false
EOF
```

### IDE Configuration

#### VS Code Setup
```bash
# Install recommended extensions
code --install-extension ms-python.python
code --install-extension bradlc.vscode-tailwindcss
code --install-extension esbenp.prettier-vscode
code --install-extension ms-vscode.vscode-typescript-next

# Open project
code .
```

#### PyCharm Setup
1. Open the project directory
2. Configure Python interpreter to use `.venv/bin/python`
3. Mark `movie_genie/` as source root
4. Enable DVC integration if available

---

## 📦 Data Setup

### Download Initial Data
```bash
# If using remote DVC storage
dvc pull

# OR manually download MovieLens data
mkdir -p data/raw/ml-100k
curl -O http://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip -d data/raw/
```

### Initialize Database
```bash
# Run initial data processing
dvc repro data_processing

# Verify database creation
ls -la movie_genie/backend/movie_genie.db
```

---

## 🧪 Verify Installation

### Test Backend
```bash
# Start backend server
cd movie_genie/backend
python app.py

# In another terminal, test API
curl http://127.0.0.1:5001/api/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "version": "1.0.0"
}
```

### Test Frontend
```bash
# Start frontend development server
cd movie_genie/frontend
npm run dev

# Open browser to http://localhost:5173
```

### Test Full Pipeline
```bash
# Run complete pipeline
dvc repro

# Check all stages completed
dvc status
```

---

## 🛠️ Tool Installation

### Optional Development Tools

#### Code Formatting
```bash
# Python formatters
pip install black isort flake8 mypy

# Frontend formatters are included in package.json
```

#### Database Tools
```bash
# SQLite browser (optional)
# macOS
brew install --cask db-browser-for-sqlite

# Ubuntu/Debian
sudo apt install sqlitebrowser

# Windows
# Download from https://sqlitebrowser.org/
```

#### Monitoring Tools
```bash
# System monitoring
pip install htop psutil

# GPU monitoring (if using CUDA)
nvidia-smi
```

---

## 🔧 Configuration

### DVC Configuration
```bash
# Configure DVC cache (optional)
dvc config cache.type symlink
dvc config cache.protected true

# Add remote storage (optional)
dvc remote add -d myremote s3://my-bucket/dvc-storage
```

### Git Configuration
```bash
# Configure Git for DVC
git config core.autocrlf input  # Handle line endings
```

### Python Path Configuration
```bash
# Add to your shell profile (~/.bashrc, ~/.zshrc)
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

---

## 🆘 Troubleshooting Installation

### Common Issues

#### Python Version Conflicts
```bash
# Use pyenv to manage Python versions
curl https://pyenv.run | bash
pyenv install 3.9.0
pyenv local 3.9.0
```

#### Permission Errors
```bash
# Fix pip permissions
pip install --user -e .

# OR use virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

#### Node.js Issues
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf movie_genie/frontend/node_modules
cd movie_genie/frontend
npm install
```

#### DVC Issues
```bash
# Reset DVC cache
dvc cache dir --unset
dvc cache dir .dvc/cache

# Reinitialize DVC
rm -rf .dvc
dvc init
```

#### Database Permissions
```bash
# Fix SQLite permissions
chmod 664 movie_genie/backend/movie_genie.db
chmod 775 movie_genie/backend/
```

### Performance Issues

#### Slow Model Training
```bash
# Check if using GPU
python -c "import torch; print(torch.cuda.is_available())"

# Reduce model size in configs/
# Edit configs/bert4rec_config.yaml:
# hidden_size: 64  # Reduce from 128
# num_layers: 2    # Reduce from 4
```

#### Memory Issues
```bash
# Monitor memory usage
htop

# Reduce batch sizes in configs/
# Edit configs/two_tower_config.yaml:
# batch_size: 128  # Reduce from 512
```

---

## ✅ Installation Checklist

- [ ] Python 3.8+ installed and activated in virtual environment
- [ ] Node.js 16+ installed
- [ ] Git configured
- [ ] Project dependencies installed (`pip install -e .`)
- [ ] Frontend dependencies installed (`npm install`)
- [ ] Environment variables configured
- [ ] DVC initialized
- [ ] Database created and accessible
- [ ] Backend API responding (`curl http://127.0.0.1:5001/api/health`)
- [ ] Frontend development server running (`npm run dev`)
- [ ] Full pipeline executable (`dvc repro`)

---

## 🚀 Next Steps

1. **Quick Start**: Follow [Quick Start Guide](quick-start.md) to see the system in action
2. **Architecture**: Read [Project Overview](project-overview.md) to understand the system
3. **Development**: Start with [Commands Reference](commands-reference.md) for daily operations
4. **ML Models**: Explore [ML Documentation](../machine-learning/README.md) for model details

---

*Installation complete! You're ready to explore modern ML-powered recommendation systems. 🎬*