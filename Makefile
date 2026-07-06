# Makefile for clinical-entity-recognition

.PHONY: clean lint help

.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo "  clean   : Remove caches and build artifacts"
	@echo "  lint    : Run black, isort, flake8 over src/"
	@echo "  help    : Show this help message"

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache

lint:
	@echo "Running linters..."
	black src
	isort src
	flake8 src
