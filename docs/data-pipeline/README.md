# 🔄 Data Pipeline Overview

Complete guide to Movie Genie's data processing and ML pipeline powered by DVC (Data Version Control).

## 🎯 Pipeline Overview

Movie Genie's data pipeline transforms raw MovieLens data into trained ML models through a series of reproducible stages:

```
Raw Data → Processing → Feature Engineering → Model Training → Deployment
```

## 📋 Table of Contents

### [📦 DVC Workflows](dvc-workflows.md)
**Perfect for**: Understanding data versioning and pipeline automation
- DVC setup and configuration
- Pipeline stage definitions
- Dependency management and caching
- Remote storage and collaboration

### [🧹 Data Processing](data-processing.md)
**Perfect for**: Understanding data transformation and cleaning
- MovieLens dataset structure
- Data cleaning and validation
- User sequence generation
- Database preparation

### [⚙️ Feature Engineering](feature-engineering.md)
**Perfect for**: Creating ML-ready features
- Content feature extraction
- User behavior encoding
- Sequence preprocessing
- Feature scaling and normalization

---

## 🔄 Complete Pipeline Flow

### Stage 1: Data Ingestion
```bash
# Download and extract MovieLens data
data/raw/ml-100k/
├── u.data      # 100k ratings
├── u.item      # Movie information
├── u.user      # User demographics
└── u.genre     # Genre definitions
```

### Stage 2: Data Processing
```bash
dvc repro data_processing
```
- Clean and validate raw data
- Merge movie and rating information
- Create user interaction sequences
- Generate train/validation splits

### Stage 3: Feature Engineering
```bash
dvc repro feature_engineering
```
- Extract content features from movie descriptions
- Create user behavior embeddings
- Generate item similarity matrices
- Prepare model-specific input formats

### Stage 4: Model Training
```bash
dvc repro train_bert4rec train_two_tower setup_semantic_search
```
- Train sequential recommendation model (BERT4Rec)
- Train collaborative filtering model (Two-Tower)
- Setup semantic search with pre-trained embeddings

### Stage 5: Database Setup
```bash
dvc repro setup_database
```
- Create SQLite database schema
- Populate with processed data
- Create indexes for fast queries
- Validate data integrity

### Stage 6: Model Deployment
```bash
dvc repro backend_server
```
- Load trained models into Flask application
- Start API server with ML endpoints
- Serve frontend interface

---

## 📊 Data Statistics

### MovieLens 100K Dataset
| Metric | Value | Description |
|--------|-------|-------------|
| **Users** | 943 | Unique users with ratings |
| **Movies** | 1,682 | Unique movies in catalog |
| **Ratings** | 100,000 | User-movie rating interactions |
| **Genres** | 19 | Movie genre categories |
| **Rating Scale** | 1-5 | Integer ratings from users |
| **Sparsity** | 93.7% | Percentage of missing user-movie pairs |

### Processed Data Output
| Dataset | Records | Features | Description |
|---------|---------|----------|-------------|
| **Movies** | 1,682 | 25 | Movie metadata with content features |
| **Users** | 943 | 10 | User demographics and behavior stats |
| **Ratings** | 100,000 | 8 | Timestamped user-movie interactions |
| **Sequences** | 943 | Variable | User interaction sequences for training |
| **Content Features** | 1,682 | 512 | Semantic embeddings for movies |

---

## 🛠️ Pipeline Configuration

### DVC Pipeline Definition (`dvc.yaml`)
```yaml
stages:
  data_processing:
    cmd: python scripts/process_movielens.py --input data/raw/ml-100k/ --output data/processed/
    deps:
      - data/raw/ml-100k/
      - scripts/process_movielens.py
    outs:
      - data/processed/movies.parquet
      - data/processed/ratings.parquet
      - data/processed/users.parquet

  feature_engineering:
    cmd: python movie_genie/data/content_features.py --input data/processed/ --output data/processed/content_features.parquet
    deps:
      - data/processed/movies.parquet
      - movie_genie/data/content_features.py
    outs:
      - data/processed/content_features.parquet

  sequential_processing:
    cmd: python movie_genie/data/sequential_processing.py --input data/processed/ --output data/processed/sequences_with_metadata.parquet
    deps:
      - data/processed/ratings.parquet
      - data/processed/movies.parquet
      - movie_genie/data/sequential_processing.py
    outs:
      - data/processed/sequences_with_metadata.parquet
```

### Parameter Configuration (`params.yaml`)
```yaml
data_processing:
  min_ratings_per_user: 20
  min_ratings_per_movie: 5
  test_split_ratio: 0.2
  random_seed: 42

feature_engineering:
  embedding_dim: 512
  max_sequence_length: 50
  content_features:
    - title
    - genres
    - release_year

model_training:
  bert4rec:
    hidden_size: 128
    num_layers: 4
    num_heads: 8
    batch_size: 256
    learning_rate: 0.001
    epochs: 50

  two_tower:
    embedding_dim: 64
    hidden_dims: [128, 64]
    batch_size: 512
    learning_rate: 0.001
    epochs: 100
```

---

## 🔧 Pipeline Management

### Running the Pipeline
```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro data_processing

# Run with force (ignore cache)
dvc repro --force

# Run downstream stages
dvc repro --downstream train_bert4rec
```

### Monitoring Pipeline Status
```bash
# Check what needs to be reproduced
dvc status

# Show pipeline graph
dvc dag

# Show detailed pipeline info
dvc pipeline show --ascii
```

### Cache Management
```bash
# Check cache status
dvc cache dir

# Clean unused cache
dvc gc

# Show cache statistics
dvc cache size
```

---

## 📈 Data Quality Monitoring

### Validation Checks
```python
# Data quality validation in pipeline
def validate_data_quality(df):
    checks = {
        'no_null_ids': df['movie_id'].notna().all(),
        'ratings_in_range': df['rating'].between(1, 5).all(),
        'valid_timestamps': df['timestamp'] > 0,
        'sufficient_interactions': len(df) >= 10000
    }
    return all(checks.values()), checks
```

### Monitoring Metrics
| Metric | Threshold | Description |
|--------|-----------|-------------|
| **Data Completeness** | > 95% | Percentage of non-null values |
| **Rating Distribution** | 1-5 range | Valid rating values |
| **User Activity** | 20+ ratings | Minimum interactions per user |
| **Movie Popularity** | 5+ ratings | Minimum ratings per movie |

---

## 🚀 Performance Optimization

### Pipeline Optimization
```bash
# Use parallel processing
dvc repro --force-downstream -j 4

# Cache optimization
dvc config cache.type symlink
dvc config cache.protected true

# Remote cache for team collaboration
dvc remote add -d myremote s3://my-bucket/dvc-cache
```

### Data Processing Optimization
```python
# Use efficient data formats
df.to_parquet('output.parquet', compression='snappy')

# Batch processing for large datasets
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process_chunk(chunk)

# Memory-efficient operations
df = df.pipe(clean_data).pipe(add_features).pipe(validate)
```

---

## 🔍 Debugging Pipeline Issues

### Common Issues and Solutions

#### "Stage not found" Error
```bash
# Check pipeline definition
cat dvc.yaml

# Verify stage name
dvc dag | grep stage_name
```

#### "Dependencies not found" Error
```bash
# Check file paths
ls -la data/raw/ml-100k/

# Verify dependencies
dvc status --verbose
```

#### "Out of memory" Error
```bash
# Reduce batch size in params.yaml
batch_size: 128  # Instead of 512

# Use chunked processing
chunk_size: 1000
```

#### "Permission denied" Error
```bash
# Fix file permissions
chmod +x scripts/process_movielens.py

# Check directory permissions
ls -la data/
```

---

## 📚 Additional Resources

### Pipeline Best Practices
- **Modular Stages**: Keep stages small and focused
- **Clear Dependencies**: Explicit input/output relationships
- **Parameterization**: Use `params.yaml` for configuration
- **Validation**: Add data quality checks at each stage
- **Documentation**: Comment complex processing logic

### Scaling Considerations
- **Large Datasets**: Use Dask or Spark for parallel processing
- **Remote Compute**: Run pipeline on cloud instances
- **Distributed Storage**: Use cloud storage for data artifacts
- **Pipeline Orchestration**: Consider Airflow for complex workflows

---

*This data pipeline provides the foundation for reproducible ML workflows. Each stage is designed to be modular, testable, and scalable for real-world applications.* 🔄