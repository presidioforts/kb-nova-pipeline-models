# LangChain Hybrid Knowledge Base Service

> **📋 [Complete Project Summary](docs/project_summary.md)** - Comprehensive overview of the entire project evolution, architecture, and achievements.

## 🚀 Overview

Production-ready knowledge base service with **LangChain integration** and intelligent routing between in-memory and ChromaDB storage tiers.

### **Key Features**
- **LangChain Integration**: Full ecosystem compatibility with ChromaDB
- **Intelligent Routing**: Automatic tier selection (1-5ms hot, 10-15ms warm, 25-50ms cold)
- **Unlimited Scalability**: ChromaDB for massive document storage
- **Advanced Search**: SentenceTransformers + LangChain retrievers
- **Production Ready**: Health monitoring, metrics, deployment guides

# Knowledge Base Nova Pipeline Models

A production-ready FastAPI service for knowledge base management using SentenceTransformer models. This service provides semantic search capabilities for troubleshooting and supports fine-tuning with new training data.

## Features

- **Semantic Search**: Find relevant solutions using sentence similarity
- **Fine-tuning**: Incrementally train the model with new data
- **Background Training**: Non-blocking model training with job tracking
- **Model Versioning**: Automatic versioning of fine-tuned models
- **Thread-safe**: Concurrent request handling with model safety
- **Hot Reloading**: Automatic loading of newly trained models

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch
- CUDA (optional, for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd kb-nova-pipeline-models
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Service

Start the FastAPI server:
```bash
python -m src.main
```

Or with uvicorn:
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8080
```

The API will be available at `http://localhost:8080`

## API Documentation

### Endpoints

#### 1. Troubleshoot Query
**POST** `/troubleshoot`

Find the best solution for a given problem description.

**Request Body:**
```json
{
  "text": "npm install is hanging"
}
```

**Response:**
```json
{
  "query": "npm install is hanging",
  "response": "Clear npm cache (npm cache clean --force) or check network.",
  "similarity_score": 0.85
}
```

#### 2. Submit Training Data
**POST** `/train`

Submit new training pairs to fine-tune the model.

**Request Body:**
```json
{
  "data": [
    {
      "input": "Docker container won't start",
      "target": "Check docker logs and verify port availability"
    },
    {
      "input": "Python import error",
      "target": "Verify module installation and PYTHONPATH"
    }
  ]
}
```

**Response:**
```json
{
  "job_id": "123e4567-e89b-12d3-a456-426614174000",
  "note": "2 pairs accepted"
}
```

#### 3. Check Training Status
**GET** `/train/{job_id}`

Check the status of a training job.

**Response:**
```json
{
  "status": "finished",
  "msg": "saved to breakfix-kb-model/all-mpnet-base-v2/fine-tuned-runs/fine-tuned-20231201-143022"
}
```

**Status Values:**
- `queued`: Job is waiting to start
- `running`: Training in progress
- `finished`: Training completed successfully
- `failed`: Training failed (check `msg` for details)

## Architecture

### Project Structure

```
src/
├── main.py                         # FastAPI application entry point
├── api/
│   └── routes.py                   # API endpoint handlers
├── models/
│   ├── schemas.py                  # Pydantic data models
│   └── sentence_transformer.py    # Model management & training
├── data/
│   └── knowledge_base.py          # Knowledge base data
└── utils/
    └── file_utils.py              # File operations & paths
```

## 📁 Folder Structure

### Complete Project Layout

```
kb-nova-pipeline-models/
├── 📁 .git/                       # Git repository metadata
├── 📁 .idea/                      # IDE configuration files
├── 📁 venv/                       # Python virtual environment
├── 📁 tests/                      # Test suite
│   ├── 📁 integration/            # Integration tests
│   │   ├── __init__.py
│   │   ├── test_api.py            # API endpoint tests
│   │   └── test_training.py       # Training workflow tests
│   ├── 📁 unit/                   # Unit tests
│   │   ├── __init__.py
│   │   ├── test_models.py         # Model validation tests
│   │   ├── test_services.py       # Service logic tests
│   │   └── test_utils.py          # Utility function tests
│   ├── 📁 fixtures/               # Test data and mocks
│   │   ├── sample_training_data.json
│   │   └── mock_responses.json
│   └── conftest.py                # Pytest configuration
├── 📁 src/                        # Main source code
│   ├── __init__.py                # Package initialization
│   ├── main.py                    # FastAPI application entry point
│   ├── 📁 api/                    # API layer
│   │   ├── __init__.py
│   │   ├── routes.py              # Endpoint handlers and business logic
│   │   ├── dependencies.py       # FastAPI dependencies (future)
│   │   └── middleware.py          # Custom middleware (future)
│   ├── 📁 models/                 # Data models and ML models
│   │   ├── __init__.py
│   │   ├── schemas.py             # Pydantic request/response models
│   │   └── sentence_transformer.py # Model management and training
│   ├── 📁 data/                   # Data access layer
│   │   ├── __init__.py
│   │   ├── knowledge_base.py      # In-memory knowledge base
│   │   ├── repositories.py       # Data access patterns (future)
│   │   └── 📁 migrations/         # Database migrations (future)
│   ├── 📁 utils/                  # Utility functions
│   │   ├── __init__.py
│   │   ├── file_utils.py          # File operations and path management
│   │   ├── validation.py          # Input validation helpers (future)
│   │   └── monitoring.py          # Metrics and monitoring (future)
│   └── 📁 core/                   # Core business logic (future expansion)
│       ├── __init__.py
│       ├── config.py              # Configuration management
│       ├── security.py            # Security utilities
│       ├── logging.py             # Logging configuration
│       └── exceptions.py          # Custom exceptions
├── 📁 docs/                       # Documentation
│   ├── api.md                     # Detailed API documentation
│   ├── deployment.md              # Production deployment guide
│   └── development.md             # Development setup and guidelines
├── 📁 scripts/                    # Utility scripts (future)
│   ├── setup.sh                   # Environment setup script
│   ├── deploy.sh                  # Deployment automation
│   └── backup_models.py           # Model backup utility
├── 📁 models/                     # Model storage (gitignored)
│   ├── 📁 base/                   # Base models
│   ├── 📁 fine-tuned/             # Fine-tuned models
│   └── 📁 backups/                # Model backups
├── 📁 logs/                       # Application logs (gitignored)
│   ├── app.log                    # Main application log
│   ├── training.log               # Training job logs
│   └── error.log                  # Error logs
├── 📄 README.md                   # Main project documentation
├── 📄 requirements.txt            # Python dependencies
├── 📄 .env.example                # Environment variables template
├── 📄 .gitignore                  # Git ignore rules
├── 📄 docker-compose.yml          # Docker composition (future)
├── 📄 Dockerfile                  # Container definition (future)
└── 📄 pyproject.toml              # Modern Python project config (future)
```

### Directory Descriptions

#### 🔧 **Core Application (`src/`)**
- **`main.py`**: FastAPI application initialization and endpoint registration
- **`api/`**: HTTP layer handling requests, responses, and routing
- **`models/`**: Data structures (Pydantic) and ML model management
- **`data/`**: Data access, storage, and knowledge base management
- **`utils/`**: Shared utilities and helper functions

#### 🧪 **Testing (`tests/`)**
- **`unit/`**: Isolated component testing
- **`integration/`**: End-to-end workflow testing
- **`fixtures/`**: Test data and mock objects

#### 📚 **Documentation (`docs/`)**
- **`api.md`**: Complete API reference with examples
- **`deployment.md`**: Production deployment instructions
- **`development.md`**: Development environment setup

#### 🗄️ **Data Storage**
- **`models/`**: Sentence transformer models and fine-tuned versions
- **`logs/`**: Application and training logs
- **`breakfix-kb-model/`**: Model storage directory structure

### Model Storage Structure

```
breakfix-kb-model/
└── all-mpnet-base-v2/             # Base model directory
    ├── 📁 fine-tuned-runs/        # Timestamped training runs
    │   ├── 📁 fine-tuned-20231201-143022/
    │   │   ├── config.json         # Model configuration
    │   │   ├── pytorch_model.bin   # Model weights
    │   │   ├── tokenizer.json      # Tokenizer configuration
    │   │   ├── tokenizer_config.json
    │   │   ├── vocab.txt           # Vocabulary
    │   │   └── pairs.json          # Training data used
    │   ├── 📁 fine-tuned-20231202-091545/
    │   └── 📁 fine-tuned-20231203-154321/
    └── 📁 fine-tuned/              # Legacy directory (backward compatibility)
```

### File Naming Conventions

#### **Python Files**
- **Snake case**: `file_utils.py`, `sentence_transformer.py`
- **Descriptive names**: Clearly indicate file purpose
- **Module organization**: Group related functionality

#### **Model Directories**
- **Timestamp format**: `fine-tuned-YYYYMMDD-HHMMSS`
- **Chronological ordering**: Latest models appear first when sorted
- **Atomic operations**: Complete model saves to prevent corruption

#### **Configuration Files**
- **Environment specific**: `.env.development`, `.env.production`
- **Template files**: `.env.example` for documentation
- **Standard formats**: JSON for data, YAML for configuration

### Key Design Decisions

#### **Modular Architecture**
- **Separation of concerns**: Each directory has a specific responsibility
- **Loose coupling**: Modules interact through well-defined interfaces
- **Scalability**: Easy to add new features without affecting existing code

#### **Data Organization**
- **Immutable models**: Fine-tuned models are never overwritten
- **Version tracking**: Timestamp-based versioning for all models
- **Backup strategy**: Separate backup directory for disaster recovery

#### **Development Workflow**
- **Test-driven**: Comprehensive test coverage for all components
- **Documentation-first**: Every feature documented before implementation
- **Quality gates**: Automated code quality checks and formatting

### Future Expansion Areas

#### **Planned Additions**
- **`src/core/`**: Configuration management and security
- **`src/services/`**: Business logic abstraction layer
- **`scripts/`**: Automation and deployment scripts
- **`docker/`**: Container configuration and orchestration

#### **Scalability Considerations**
- **Database integration**: Replace in-memory storage
- **Microservices**: Split into smaller, focused services
- **Monitoring**: Add comprehensive observability
- **Security**: Implement authentication and authorization

This folder structure provides a solid foundation for both current functionality and future enhancements while maintaining clean separation of concerns and professional development practices.

### Model Storage

```
breakfix-kb-model/
└── all-mpnet-base-v2/
    ├── fine-tuned-runs/           # Timestamped training runs
    │   ├── fine-tuned-20231201-143022/
    │   │   ├── model files
    │   │   └── pairs.json         # Training data used
    │   └── ...
    └── fine-tuned/                # Legacy directory
```

### Key Components

- **SentenceTransformer**: Uses `all-mpnet-base-v2` as base model
- **Training**: Cosine similarity loss with learning rate 1e-5
- **Threading**: Thread-safe model access with locks
- **Persistence**: JSON storage for training pairs and model versioning

## Configuration

### Environment Variables

- `HF_HUB_DISABLE_PROGRESS_BARS=1`: Disable Hugging Face progress bars
- `HF_HUB_OFFLINE=1`: Use offline mode for Hugging Face

### Model Paths

- Base model: `breakfix-kb-model/all-mpnet-base-v2/`
- Fine-tuned models: `breakfix-kb-model/all-mpnet-base-v2/fine-tuned-runs/`

## Development

### Code Organization

The codebase follows a modular architecture:

- **Separation of Concerns**: Each module has a single responsibility
- **Clean Imports**: Organized dependencies between modules
- **Thread Safety**: Proper locking for concurrent access
- **Error Handling**: Comprehensive exception handling

### Adding New Features

1. **API Changes**: Modify `src/api/routes.py`
2. **Data Models**: Update `src/models/schemas.py`
3. **Business Logic**: Extend appropriate service modules
4. **Utilities**: Add helpers to `src/utils/`

## Production Considerations

### Security
- Input validation on all endpoints
- Error handling without information leakage
- Thread-safe operations

### Performance
- Model loading at startup
- Efficient similarity computation
- Background training jobs

### Monitoring
- Structured logging
- Job status tracking
- Model version management

## Troubleshooting

### Common Issues

1. **Model Loading Fails**
   - Check if base model directory exists
   - Verify PyTorch installation
   - Check available memory

2. **Training Jobs Fail**
   - Verify training data format
   - Check disk space for model storage
   - Review logs for specific errors

3. **Import Errors**
   - Ensure all dependencies are installed
   - Check Python path configuration
   - Verify virtual environment activation

### Logs

The service uses structured logging. Check logs for:
- Model loading status
- Training job progress
- API request errors
- File system operations

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
