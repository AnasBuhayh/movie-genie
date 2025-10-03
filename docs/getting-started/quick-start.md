# ⚡ Quick Start Guide

Get Movie Genie up and running in 5 minutes! This guide gets you from zero to a working AI-powered movie recommendation system.

## 🚀 5-Minute Setup

### Step 1: Clone and Install (2 minutes)
```bash
# Clone the repository
git clone <repository-url>
cd movie-genie

# Install the project
pip install -e .
```

### Step 2: Run the Pipeline (2 minutes)
```bash
# Run the complete DVC pipeline
dvc repro
```

This command will:
- ✅ Process the MovieLens dataset
- ✅ Train all ML models (BERT4Rec, Two-Tower, Semantic Search)
- ✅ Set up the database
- ✅ Start the backend server

### Step 3: Access the Application (1 minute)
```bash
# Open your browser to:
http://127.0.0.1:5001
```

🎉 **You're done!** The application is now running with:
- **Frontend**: React interface served by Flask
- **Backend**: Flask API with ML models
- **Database**: SQLite with MovieLens data
- **ML Models**: Trained and ready for recommendations

---

## 🎯 What You'll See

### 1. User Selection
- Enter a user ID (1-610) to simulate different users
- Each user has unique viewing history and preferences

### 2. Homepage
- **Popular Movies**: Most-watched films from the dataset
- **Personalized Recommendations**: ML-powered suggestions
- **Genre Collections**: Movies organized by category

### 3. Search & Discovery
- **Semantic Search**: Find movies using natural language
- **Grid Results**: Browse search results in an organized layout
- **Movie Details**: Click any movie for detailed information

---

## 🧪 Test the Features

### Test ML Recommendations
```bash
# Try different user IDs to see personalized results
User ID 123: Sci-fi and action movie fan
User ID 456: Romance and comedy preferences
User ID 789: Horror and thriller enthusiast
```

### Test Semantic Search
```bash
# Try these search queries:
"funny movies for family night"
"sci-fi movies with time travel"
"action movies with robots"
"romantic comedies from the 90s"
```

### Test API Endpoints
```bash
# Check API health
curl http://127.0.0.1:5001/api/health

# Get user info
curl http://127.0.0.1:5001/api/users/info

# Search movies
curl "http://127.0.0.1:5001/api/search/semantic?q=action%20movies"
```

---

## 🔧 Common Quick Commands

### Restart Everything
```bash
# If something goes wrong, restart the pipeline
dvc repro --force
```

### Check What's Running
```bash
# Check if backend is running
curl http://127.0.0.1:5001/api/health

# Check database
sqlite3 movie_genie/backend/movie_genie.db "SELECT COUNT(*) FROM movies;"
```

### View Logs
```bash
# Backend logs (if running in terminal)
# Check the terminal where you ran `dvc repro`

# Or check DVC logs
dvc status
```

---

## 🆘 Quick Troubleshooting

### Issue: "Port already in use"
```bash
# Find and kill the process using port 5001
lsof -ti:5001 | xargs kill -9
dvc repro
```

### Issue: "No module named 'movie_genie'"
```bash
# Reinstall the project
pip install -e .
```

### Issue: "Database not found"
```bash
# Re-run database setup
dvc repro data_processing setup_database
```

### Issue: "Models not loading"
```bash
# Check if models were trained
ls -la models/
# If empty, retrain models
dvc repro train_bert4rec train_two_tower setup_semantic_search
```

---

## 🎯 Next Steps

Once you have the basic system running:

1. **Explore the Code**: Check out [Project Overview](project-overview.md) for architecture details
2. **Understand ML Models**: Read [ML Models Documentation](../machine-learning/README.md)
3. **Customize**: Modify configurations in `configs/` directory
4. **Develop**: Follow [Installation Guide](installation.md) for development setup

---

## 🚀 Development Mode

If you want to develop the frontend separately:

```bash
# Terminal 1: Backend only
cd movie_genie/backend
python app.py

# Terminal 2: Frontend development server
cd movie_genie/frontend
npm install
npm run dev
```

Frontend will be available at `http://localhost:5173` with hot reload.

---

*That's it! You now have a complete AI-powered movie recommendation system running locally. Time to explore what modern ML can do! 🎬*