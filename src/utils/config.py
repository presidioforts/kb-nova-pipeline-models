"""
Configuration management for KB Nova Pipeline Models.
"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field
from functools import lru_cache


class ModelConfig(BaseSettings):
    """Model configuration."""
    type: str = "transformer"
    architecture: str = "all-mpnet-base-v2"
    max_sequence_length: int = 512
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    dropout_rate: float = 0.1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01


class TrainingConfig(BaseSettings):
    """Training configuration."""
    epochs: int = 10
    batch_size: int = 16
    gradient_accumulation_steps: int = 2
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 50
    early_stopping_patience: int = 3
    metric_for_best_model: str = "eval_loss"


class DataConfig(BaseSettings):
    """Data configuration."""
    raw_path: str = "data/raw"
    processed_path: str = "data/processed"
    external_path: str = "data/external"
    interim_path: str = "data/interim"
    batch_size: int = 32
    validation_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42


class PathsConfig(BaseSettings):
    """Paths configuration."""
    models: str = "models"
    trained_models: str = "models/trained"
    artifacts: str = "models/artifacts"
    logs: str = "logs"
    reports: str = "reports"
    notebooks: str = "notebooks"


class APIConfig(BaseSettings):
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = True
    log_level: str = "info"


class MLOpsConfig(BaseSettings):
    """MLOps configuration."""
    experiment_tracking: str = "mlflow"
    model_registry: str = "mlflow"
    artifact_store: str = "local"


class SecurityConfig(BaseSettings):
    """Security configuration."""
    allowed_hosts: list = ["localhost", "127.0.0.1"]
    cors_origins: list = ["http://localhost:3000", "http://localhost:8080"]
    max_request_size: str = "10MB"
    rate_limit: str = "100/minute"


class MonitoringConfig(BaseSettings):
    """Monitoring configuration."""
    enable_metrics: bool = True
    enable_health_checks: bool = True
    metrics_port: int = 9090


class ProjectConfig(BaseSettings):
    """Project configuration."""
    name: str = "kb-nova-pipeline-models"
    version: str = "0.1.0"
    description: str = "AI/ML Pipeline Models for Knowledge Base Nova"


class EnvironmentConfig(BaseSettings):
    """Environment configuration."""
    name: str = "development"
    debug: bool = True
    log_level: str = "INFO"


class Settings(BaseSettings):
    """Main settings class."""
    
    # Sub-configurations
    project: ProjectConfig = ProjectConfig()
    environment: EnvironmentConfig = EnvironmentConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    paths: PathsConfig = PathsConfig()
    api: APIConfig = APIConfig()
    mlops: MLOpsConfig = MLOpsConfig()
    security: SecurityConfig = SecurityConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


def load_yaml_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        return {}
    
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Failed to load config from {config_path}: {e}")
        return {}


def merge_configs(base_config: Dict[str, Any], yaml_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge YAML configuration with base configuration.
    
    Args:
        base_config: Base configuration dictionary
        yaml_config: YAML configuration dictionary
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in yaml_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings with caching.
    
    Returns:
        Settings instance
    """
    # Load YAML configuration
    yaml_config = load_yaml_config()
    
    # Create base settings from environment variables and defaults
    settings = Settings()
    
    # Override with YAML configuration if available
    if yaml_config:
        # Convert settings to dict for merging
        settings_dict = settings.dict()
        
        # Merge configurations
        merged_config = merge_configs(settings_dict, yaml_config)
        
        # Create new settings instance with merged config
        settings = Settings(**merged_config)
    
    return settings


def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get environment variable with optional default.
    
    Args:
        key: Environment variable key
        default: Default value if key not found
        
    Returns:
        Environment variable value or default
    """
    return os.getenv(key, default)


def is_development() -> bool:
    """Check if running in development environment."""
    settings = get_settings()
    return settings.environment.name.lower() == "development"


def is_production() -> bool:
    """Check if running in production environment."""
    settings = get_settings()
    return settings.environment.name.lower() == "production"


def get_log_level() -> str:
    """Get the configured log level."""
    settings = get_settings()
    return settings.environment.log_level


def get_model_path() -> str:
    """Get the model path."""
    settings = get_settings()
    return settings.paths.models


def get_data_path() -> str:
    """Get the data path."""
    settings = get_settings()
    return settings.data.processed_path 