# Two-Tower Model Test Suite

This directory contains comprehensive tests for the two-tower recommendation model, with a focus on temporal splitting functionality and the fixes implemented to resolve training issues.

## Test Structure

### Core Test Files

- **`test_temporal_splitting.py`** - Tests for temporal splitting functionality
- **`test_two_tower_integration.py`** - Integration tests for the complete training pipeline
- **`test_edge_cases.py`** - Edge cases and error handling tests
- **`conftest.py`** - Shared fixtures and test configuration
- **`test_loaders.py`** - Existing tests for data loaders
- **`test_embedding.py`** - Tests for embedding functionality

### Test Categories

#### Temporal Splitting Tests (`test_temporal_splitting.py`)
Tests the core temporal splitting functionality that was fixed to resolve the "0 positive examples in validation" issue:

- ✅ **Basic temporal splitting** - Verifies proper chronological ordering
- ✅ **Timestamp column handling** - Tests both 'datetime' and 'timestamp' columns
- ✅ **Fallback behavior** - Tests behavior when no timestamp columns exist
- ✅ **Example creation from sequences** - Tests the extracted example generation logic
- ✅ **Train-validation split integration** - Tests the complete temporal splitting pipeline
- ✅ **Split overlap analysis** - Tests the overlap analysis functionality
- ✅ **Data loading pipeline** - Tests the complete load and split workflow

#### Integration Tests (`test_two_tower_integration.py`)
Tests the complete end-to-end training pipeline:

- ✅ **Complete training pipeline** - Full workflow from data loading to model training
- ✅ **Model forward pass** - Tests model inference with real data
- ✅ **Evaluator functionality** - Tests model evaluation metrics
- ✅ **Temporal consistency** - Verifies temporal ordering throughout pipeline
- ✅ **Movie feature consistency** - Tests movie feature mapping fixes
- ✅ **Error handling** - Tests graceful error handling (division by zero fix)
- ✅ **Different model configurations** - Tests various model architectures

#### Edge Cases Tests (`test_edge_cases.py`)
Tests unusual scenarios and error conditions:

- ✅ **Minimum viable training** - Tests with absolute minimum data
- ✅ **All positive/negative ratings** - Tests extreme rating distributions
- ✅ **Mismatched movie IDs** - Tests when sequences/features don't align
- ✅ **Missing required columns** - Tests proper error handling
- ✅ **Unbalanced validation splits** - Tests the division by zero fix
- ✅ **Extreme sampling ratios** - Tests with very high/low negative sampling
- ✅ **Single movie datasets** - Tests degenerate cases
- ✅ **Corrupted embeddings** - Tests handling of invalid data

## Key Bug Fixes Tested

### 1. Division by Zero Fix
**Issue**: Validation sets with 0 positive examples caused division by zero errors.
**Fix**: Added safety check in `_evaluate_epoch()` method.
**Tests**: `test_extremely_unbalanced_validation_split`, `test_error_handling_in_pipeline`

### 2. Temporal Splitting Fix
**Issue**: Temporal splitting was applied to training examples (no timestamps) instead of sequences (with timestamps).
**Fix**: Modified `_create_train_val_split()` to split sequences first, then create examples.
**Tests**: `test_temporal_train_val_split_integration`, `test_temporal_consistency_throughout_pipeline`

### 3. Movie Feature Mapping Fix
**Issue**: KeyError when accessing movie features due to index mapping mismatch.
**Fix**: Updated `movie_feature_map` to use consistent indexing.
**Tests**: `test_movie_feature_consistency`, `test_mismatched_movie_ids`

### 4. Rating Analysis Fix
**Issue**: Mixed output showing both 1-5 scale and thumbs ratings.
**Fix**: Updated analysis to use `thumbs_rating` column instead of `rating` column.
**Tests**: Tests validate proper thumbs rating distributions throughout

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Categories
```bash
# Temporal splitting tests only
pytest tests/test_temporal_splitting.py

# Integration tests only
pytest tests/test_two_tower_integration.py

# Edge cases only
pytest tests/test_edge_cases.py
```

### Run Tests by Marker
```bash
# Fast tests only (exclude slow integration tests)
pytest -m "not slow"

# Temporal-related tests only
pytest -m temporal

# Edge case tests only
pytest -m edge_case

# Integration tests only
pytest -m integration
```

### Verbose Output
```bash
# See detailed test output
pytest -v tests/

# See print statements and logging
pytest -s tests/
```

## Test Data

Tests use realistic synthetic data created by fixtures in `conftest.py`:

- **Sequences**: Include proper temporal ordering, realistic rating distributions, and thumbs ratings
- **Movies**: Include comprehensive feature sets (numerical, categorical, language, embeddings)
- **Temporal patterns**: Data spans multiple time periods to test temporal splitting

### Data Characteristics
- **Rating Distribution**: 60% positive (thumbs up/two thumbs up), 20% negative (thumbs down), 20% other
- **Temporal Span**: Multiple years of synthetic data with realistic progression
- **Feature Dimensions**: 768-dimensional text embeddings (matching EmbeddingGemma)
- **Scale**: Tests range from 50 sequences (fast unit tests) to 1000+ sequences (integration tests)

## Test Fixtures

### Shared Fixtures (in `conftest.py`)
- `realistic_test_data` - Comprehensive test dataset for most tests
- `small_test_data` - Minimal dataset for fast unit tests
- `sample_rating_distribution` - Standard rating distribution parameters
- `standard_movie_features` - Standard movie feature configuration
- `suppress_warnings` - Suppresses common warnings during testing

### Helper Functions
- `create_realistic_sequences()` - Generate realistic sequence data
- `create_realistic_movies()` - Generate realistic movie features

## Continuous Integration

These tests are designed to:
- ✅ **Validate all bug fixes** - Ensure temporal splitting, mapping, and division by zero fixes work
- ✅ **Prevent regressions** - Catch any future changes that break the fixes
- ✅ **Test edge cases** - Handle unusual data patterns gracefully
- ✅ **Validate integration** - Ensure the complete pipeline works end-to-end

## Test Performance

- **Fast tests** (~10-30 seconds): Basic functionality and edge cases
- **Integration tests** (~1-3 minutes): Complete training pipeline
- **All tests** (~5 minutes): Full test suite

Run `pytest -m "not slow"` for development to skip slow integration tests.

## Adding New Tests

When adding new tests:

1. **Use existing fixtures** from `conftest.py` when possible
2. **Add appropriate markers** (@pytest.mark.slow, @pytest.mark.temporal, etc.)
3. **Test both success and failure cases**
4. **Include realistic data scenarios**
5. **Document the specific functionality being tested**

### Example Test Structure
```python
@pytest.mark.temporal
def test_new_temporal_feature(realistic_test_data):
    \"\"\"Test description of what this validates.\"\"\"
    # Arrange
    data_loader = TwoTowerDataLoader(...)

    # Act
    result = data_loader.new_method()

    # Assert
    assert result is not None
    assert len(result) > 0
```

## Test Coverage

The test suite provides comprehensive coverage of:
- ✅ **Core functionality** - All main two-tower model operations
- ✅ **Bug fixes** - All identified and resolved issues
- ✅ **Edge cases** - Unusual data patterns and error conditions
- ✅ **Integration** - Complete end-to-end workflows
- ✅ **Performance** - Various model configurations and data sizes

This ensures the two-tower model training pipeline is robust, reliable, and maintainable.