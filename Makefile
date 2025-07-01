# Amazon Deep Insights - Makefile
# This Makefile provides commands for setting up the environment, running the API,
# running the web interface, and running tests.

# Environment variables
PYTHON := python
VENV := .venv
PIP := $(VENV)/bin/pip
PYTHON_VENV := $(VENV)/bin/python
UVICORN := $(VENV)/bin/uvicorn
STREAMLIT := $(VENV)/bin/streamlit

# Project directories
SRC_DIR := src
DATA_DIR := data
RAW_DATA_DIR := $(DATA_DIR)/raw
PROCESSED_DATA_DIR := $(DATA_DIR)/processed
VECTOR_DB_DIR := $(DATA_DIR)/vector_db
CORPUS_DIR := $(DATA_DIR)/raw/corpus

# API settings
API_HOST := 0.0.0.0
API_PORT := 8080

# Default target
.PHONY: all
all: setup

# Setup environment
.PHONY: setup
setup: $(VENV) create-dirs

$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

.PHONY: create-dirs
create-dirs:
	mkdir -p $(RAW_DATA_DIR)
	mkdir -p $(PROCESSED_DATA_DIR)
	mkdir -p $(VECTOR_DB_DIR)
	mkdir -p $(CORPUS_DIR)
	mkdir -p $(RAW_DATA_DIR)/lidar
	mkdir -p $(RAW_DATA_DIR)/rasters
	mkdir -p $(RAW_DATA_DIR)/vectors
	mkdir -p $(PROCESSED_DATA_DIR)/dem
	mkdir -p $(PROCESSED_DATA_DIR)/dsm
	mkdir -p $(PROCESSED_DATA_DIR)/chm

# Run API
.PHONY: api
api: $(VENV)
	$(UVICORN) src.rag.api:app --host $(API_HOST) --port $(API_PORT) --reload

# Run web interface
.PHONY: app
app: $(VENV)
	$(STREAMLIT) run src/visualization/app.py

# Run both API and web interface (in separate terminals)
.PHONY: run
run:
	@echo "Please run the API and web interface in separate terminals:"
	@echo "  make api    # In terminal 1"
	@echo "  make app    # In terminal 2"

# Run tests
.PHONY: test
test: $(VENV)
	$(PYTHON_VENV) test_integration.py

# Build knowledge base
.PHONY: build-kb
build-kb: $(VENV)
	$(PYTHON_VENV) -m src.rag.embeddings build --corpus $(CORPUS_DIR) --persist-dir $(VECTOR_DB_DIR) --collection amazon_insights

# Download sample data
.PHONY: download-samples
download-samples: $(VENV)
	$(PYTHON_VENV) -m src.data_ingestion.download download --source url --url https://github.com/PDAL/data/raw/master/autzen/autzen-classified.las --output $(RAW_DATA_DIR)/lidar/sample.las

# Process sample data
.PHONY: process-samples
process-samples: $(VENV) download-samples
	$(PYTHON_VENV) -m src.preprocessing.lidar_processing --input $(RAW_DATA_DIR)/lidar/sample.las --output-dir $(PROCESSED_DATA_DIR) --resolution 1.0 --products dem dsm chm

# Clean up
.PHONY: clean
clean:
	rm -rf __pycache__
	rm -rf $(SRC_DIR)/__pycache__
	rm -rf $(SRC_DIR)/*/__pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov

# Deep clean (remove virtual environment and generated data)
.PHONY: clean-all
clean-all: clean
	rm -rf $(VENV)
	rm -rf $(PROCESSED_DATA_DIR)/*
	rm -rf $(VECTOR_DB_DIR)/*

# Install development dependencies
.PHONY: dev-setup
dev-setup: $(VENV)
	$(PIP) install -e .[dev]

# Create a new .env file from .env.example if it doesn't exist
.PHONY: env
env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo ".env file created from .env.example"; \
		echo "Please edit .env file with your API keys and configuration"; \
	else \
		echo ".env file already exists"; \
	fi

# Help command
.PHONY: help
help:
	@echo "Amazon Deep Insights - Makefile Commands"
	@echo ""
	@echo "Setup commands:"
	@echo "  make setup          Create virtual environment and install dependencies"
	@echo "  make create-dirs    Create necessary directories"
	@echo "  make env            Create .env file from .env.example"
	@echo "  make dev-setup      Install development dependencies"
	@echo ""
	@echo "Run commands:"
	@echo "  make api            Run the RAG API server"
	@echo "  make app            Run the Streamlit web interface"
	@echo "  make run            Instructions for running both API and web interface"
	@echo ""
	@echo "Data commands:"
	@echo "  make download-samples    Download sample LiDAR data"
	@echo "  make process-samples     Process sample LiDAR data"
	@echo "  make build-kb            Build knowledge base from corpus"
	@echo ""
	@echo "Test commands:"
	@echo "  make test           Run integration tests"
	@echo ""
	@echo "Clean commands:"
	@echo "  make clean          Clean up temporary files"
	@echo "  make clean-all      Clean up everything including virtual environment"
