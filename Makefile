.PHONY: help install format lint check test clean

help:
	@echo "Available commands:"
	@echo "  install    - Install dependencies and pre-commit hooks"
	@echo "  format     - Format code with ruff"
	@echo "  lint       - Run linting checks"
	@echo "  check      - Run both format and lint checks"
	@echo "  test       - Run tests with test data"
	@echo "  clean      - Clean up generated files"

install:
	uv sync
	uv run pre-commit install

format:
	uv run ruff format .

lint:
	uv run ruff check . --fix

check: format lint

test:
	uv run python create_test_data.py
	uv run python prom_seg_clus.py mfcc -1 test_data/audio test_data/features test_data/boundaries test_data/output 5 --extension .wav

clean:
	rm -rf test_data/
	rm -rf data/
	rm -f dev-clean.tar.gz
	rm -rf LibriSpeech/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete