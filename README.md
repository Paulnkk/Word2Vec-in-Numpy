# Word2Vec in NumPy (Skip-gram + Negative Sampling)

This project implements a lightweight Word2Vec training pipeline using only NumPy and Python standard libraries.  
It includes preprocessing, negative sampling, model training, and simple similarity checks.

## Project Structure

- `preprocessing.py`: text preprocessing, vocabulary creation, context-pair generation, and unigram-based negative sampling
- `model.py`: skip-gram model with negative sampling update step
- `main.py`: dataset download/load, training loop, and embedding similarity examples

## Features

- Pure NumPy implementation
- Automatic download and extraction of the Text8 dataset
- Unigram distribution with frequency^0.75 for negative sampling
- Simple end-to-end training script
- Configurable hyperparameters through environment variables

## Requirements

- Python 3.9+ (recommended)
- NumPy

Install dependency:

```bash
pip install numpy
```

## Run

```bash
python main.py
```

On the first run, the script downloads `text8.zip` (about 31MB) into the project folder.

## Optional Configuration

You can override defaults with environment variables:

- `MAX_WORDS` (default: `50000`)
- `EMBED_DIM` (default: `50`)
- `EPOCHS` (default: `2`)
- `LEARNING_RATE` (default: `0.01`)
- `WINDOW_SIZE` (default: `2`)
- `NEGATIVE_SAMPLES` (default: `5`)

PowerShell example:

```powershell
$env:MAX_WORDS=20000
$env:EPOCHS=1
python main.py
```

## Notes

- Training with `MAX_WORDS=50000` can take time in pure NumPy depending on your CPU.
- Start with smaller values (for example, `MAX_WORDS=10000`, `EPOCHS=1`) for quick iteration.
- Similarity examples at the end are a sanity check, not a benchmark.
