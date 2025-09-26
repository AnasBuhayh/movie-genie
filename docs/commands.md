### database test

`
rm -f movie_genie.db*  # Remove any existing database files
python test_db.py
`

### running scrapper

`
python scripts/imdb_featured_reviews.py \
  --links-csv data/raw/ml-100k/links.csv \
  --limit 25 \
  --out data/raw/imdb-reviews/ml-100k_reviews.csv \
  --lang en \
  --min-delay 0.05 \
  --max-delay 0.1 \
  --checkpoint data/raw/imdb-reviews/ml-100k_checkpoint.json \
  --filter-by-movies
`