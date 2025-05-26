# KB Nova Pipeline Models

A comprehensive AI/ML pipeline for knowledge base processing and model training with production-ready folder structure and security standards.

## 🏗️ Project Structure

```
kb-nova-pipeline-models/
├── src/                          # Source code
│   ├── data/                     # Data processing modules
│   ├── models/                   # Model definitions and architectures
│   ├── features/                 # Feature engineering
│   ├── visualization/            # Data visualization utilities
│   ├── utils/                    # Utility functions
│   ├── api/                      # API endpoints and services
│   ├── training/                 # Model training scripts
│   ├── inference/                # Model inference and prediction
│   └── evaluation/               # Model evaluation and metrics
├── data/                         # Data storage
│   ├── raw/                      # Raw, immutable data
│   ├── processed/                # Cleaned and processed data
│   ├── external/                 # External data sources
│   └── interim/                  # Intermediate data transformations
├── models/                       # Trained models and artifacts
│   ├── trained/                  # Serialized trained models
│   └── artifacts/                # Model artifacts and metadata
├── notebooks/                    # Jupyter notebooks
│   ├── exploratory/              # Exploratory data analysis
│   └── experiments/              # Model experiments
├── tests/                        # Test suite
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
├── configs/                      # Configuration files
├── scripts/                      # Utility scripts
├── docs/                         # Documentation
├── logs/                         # Application logs
├── reports/                      # Generated reports
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── pyproject.toml               # Modern Python project config
├── Makefile                     # Development automation
└── .gitignore                   # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd kb-nova-pipeline-models
   ```

2. **Set up virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Unix/MacOS
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   make install-dev
   # or
   pip install -r requirements.txt
   ```

4. **Set up environment:**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

5. **Run setup script:**
   ```bash
   python scripts/setup.py
   ```

## 🛠️ Development

### Available Commands

```bash
make help                 # Show all available commands
make install             # Install production dependencies
make install-dev         # Install development dependencies
make test                # Run tests
make lint                # Run linting checks
make format              # Format code
make security            # Run security checks
make clean               # Clean up temporary files
make train               # Train models
make predict             # Run predictions
make serve               # Start API server
make docs                # Generate documentation
```

### Code Quality

This project enforces high code quality standards:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **Bandit** for security analysis
- **Pytest** for testing with coverage
- **Pre-commit hooks** for automated checks

### Testing

```bash
# Run all tests
make test

# Run specific test types
make test-unit
make test-integration

# Run with coverage
pytest --cov=src --cov-report=html
```

## 📊 ML Pipeline

### Data Processing

```bash
# Process raw data
make process-data
# or
python -m src.data.process
```

### Model Training

```bash
# Train models
make train
# or
python -m src.training.train
```

### Model Evaluation

```bash
# Evaluate models
make evaluate
# or
python -m src.evaluation.evaluate
```

### Inference

```bash
# Run predictions
make predict
# or
python -m src.inference.predict
```

## 🌐 API

Start the API server:

```bash
# Development
make serve

# Production
make serve-prod
```

The API will be available at `http://localhost:8000`

### Chroma DB Integration

This project integrates with [Chroma](https://www.trychroma.com/), an open-source AI application database that provides:

- **Vector embeddings** using your fine-tuned SentenceTransformer models
- **Semantic search** across knowledge base items and chat history
- **Persistent storage** for conversations and documents
- **Metadata filtering** for advanced queries
- **Multi-modal support** for future enhancements

#### Key Features:
- 🔍 **Semantic Search**: Find relevant solutions using vector similarity
- 💬 **Context-Aware Chat**: Conversations that remember previous interactions
- 📚 **Knowledge Base Integration**: Automatic indexing of troubleshooting solutions
- 🔄 **Real-time Updates**: Dynamic addition of new knowledge and conversations
- 📊 **Analytics**: Track usage patterns and improve responses

#### Chat API Endpoints:
- `POST /chat` - Send a chat message with context awareness
- `POST /chat/session` - Create a new chat session
- `GET /chat/session/{id}/history` - Get chat history
- `GET /chat/search` - Search across conversations
- `GET /chroma/stats` - Get Chroma database statistics

#### Demo:
```bash
# Run the Chroma integration demo
python examples/chroma_demo.py
```

#### Learn More:
- 🌐 **Chroma Official Website**: [https://www.trychroma.com/](https://www.trychroma.com/)
- 📖 **Chroma Documentation**: Available on their website
- 💬 **Chroma Community**: Join their Discord for support

## 📈 MLOps

### Experiment Tracking

- **MLflow**: Track experiments and model versions
- **Weights & Biases**: Advanced experiment tracking
- **TensorBoard**: Visualization of training metrics

```bash
# Start MLflow UI
make mlflow-ui
```

### Model Registry

Models are automatically registered and versioned using MLflow.

## 🔒 Security

This project follows security best practices:

- Environment variables for sensitive data
- Security scanning with Bandit
- Dependency vulnerability checking
- Secure defaults in configurations
- Input validation and sanitization

## 📝 Configuration

Configuration is managed through:

- `configs/config.yaml` - Main configuration
- `.env` - Environment variables
- `pyproject.toml` - Tool configurations

## 🧪 Jupyter Notebooks

```bash
# Start Jupyter Lab
make jupyter
```

Notebooks are organized in:
- `notebooks/exploratory/` - Data exploration
- `notebooks/experiments/` - Model experiments

## 📚 Documentation

Generate and serve documentation:

```bash
make docs
make docs-serve
```

## 🐳 Docker (Optional)

```bash
# Build Docker image
make docker-build

# Run container
make docker-run
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:

- Create an issue in the repository
- Check the documentation in `docs/`
- Review the configuration files in `configs/`

##### Tags: ml-models, transformers, pipeline, rag, ai, data-science, mlops
