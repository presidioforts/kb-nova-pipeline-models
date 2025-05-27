# Notebooks Directory

This directory contains Jupyter notebooks for data science and machine learning workflows in the KB Nova Pipeline Models project.

## 📁 Directory Structure

### `exploratory/` - Exploratory Data Analysis (EDA)
Notebooks for initial data exploration, understanding, and visualization:

- **`01_data_exploration.ipynb`** - Comprehensive analysis of knowledge base data
  - Data distribution analysis
  - Text characteristics and patterns
  - Category and tag analysis
  - Data quality assessment
  - Insights for model training

### `experiments/` - Model Experiments and Training
Notebooks for conducting ML experiments, hyperparameter tuning, and model comparisons:

- **`01_model_training_experiment.ipynb`** - SentenceTransformer fine-tuning experiments
  - Compare different base models
  - Hyperparameter optimization
  - Performance evaluation
  - MLflow experiment tracking

- **`02_chroma_vector_search_optimization.ipynb`** - Chroma DB optimization experiments
  - Embedding model benchmarking
  - Search parameter optimization
  - Metadata filtering performance
  - Production deployment recommendations

## 🚀 Getting Started

### Prerequisites
```bash
# Install Jupyter Lab
pip install jupyterlab

# Install additional dependencies for notebooks
pip install matplotlib seaborn plotly
```

### Running Notebooks
```bash
# Start Jupyter Lab
jupyter lab

# Or use the make command
make jupyter
```

### Environment Setup
Make sure to add the `src` directory to your Python path in each notebook:
```python
import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent / 'src'))
```

## 📊 What Goes in Each Directory?

### Exploratory Notebooks Should Include:
- **Data Loading and Inspection**: Understanding data structure and quality
- **Statistical Analysis**: Distributions, correlations, and patterns
- **Visualizations**: Charts, plots, and graphs to understand data
- **Data Quality Checks**: Missing values, outliers, inconsistencies
- **Feature Analysis**: Understanding feature importance and relationships
- **Insights Documentation**: Key findings and recommendations

### Experiment Notebooks Should Include:
- **Hypothesis Definition**: Clear experiment goals and expectations
- **Experiment Configuration**: Parameters, models, and settings to test
- **Training and Evaluation**: Model training with proper validation
- **Results Analysis**: Performance metrics and comparisons
- **Hyperparameter Tuning**: Systematic parameter optimization
- **Experiment Tracking**: MLflow or W&B integration
- **Conclusions and Recommendations**: Best practices and next steps

## 🔬 Experiment Tracking

### MLflow Integration
Experiments are tracked using MLflow:
```python
import mlflow
mlflow.set_experiment("your-experiment-name")

with mlflow.start_run():
    mlflow.log_param("parameter_name", value)
    mlflow.log_metric("metric_name", score)
    mlflow.log_artifact("model_path")
```

### Weights & Biases (Optional)
For advanced experiment tracking:
```python
import wandb
wandb.init(project="kb-nova-models")
wandb.log({"accuracy": 0.95, "loss": 0.05})
```

## 📝 Best Practices

### Notebook Organization
1. **Clear Structure**: Use markdown headers to organize sections
2. **Documentation**: Explain what each cell does and why
3. **Reproducibility**: Set random seeds and document dependencies
4. **Version Control**: Save important results and configurations
5. **Clean Code**: Remove debugging code and unused cells before committing

### Data Handling
1. **Relative Paths**: Use relative paths from notebook location
2. **Data Validation**: Check data integrity before processing
3. **Memory Management**: Be mindful of large datasets
4. **Backup Results**: Save important outputs to files

### Experiment Management
1. **Hypothesis-Driven**: Start with clear questions and hypotheses
2. **Systematic Testing**: Test one variable at a time when possible
3. **Result Documentation**: Record both successes and failures
4. **Reproducible Results**: Document exact configurations for reproduction

## 🔄 Workflow Integration

### From Exploration to Production
1. **Explore** → Use exploratory notebooks to understand data
2. **Experiment** → Test different approaches in experiment notebooks
3. **Validate** → Confirm results with proper validation
4. **Productionize** → Move successful experiments to `src/` modules
5. **Monitor** → Track production performance and iterate

### Collaboration
- **Clear Naming**: Use descriptive notebook names with version numbers
- **Documentation**: Include context and findings in markdown cells
- **Code Quality**: Follow project coding standards even in notebooks
- **Results Sharing**: Export key findings to reports or presentations

## 📈 Example Workflows

### Data Science Workflow
1. `exploratory/01_data_exploration.ipynb` - Understand the data
2. `exploratory/02_feature_engineering.ipynb` - Create and test features
3. `experiments/01_baseline_models.ipynb` - Establish baseline performance
4. `experiments/02_advanced_models.ipynb` - Test sophisticated approaches
5. `experiments/03_hyperparameter_tuning.ipynb` - Optimize best models

### ML Pipeline Development
1. Explore data characteristics and quality
2. Design and test preprocessing pipelines
3. Experiment with different model architectures
4. Optimize hyperparameters and training procedures
5. Validate on holdout data and production scenarios
6. Document best practices and deployment recommendations

## 🛠️ Troubleshooting

### Common Issues
- **Import Errors**: Ensure `src` is in Python path
- **Memory Issues**: Use data sampling for large datasets
- **Slow Performance**: Consider using smaller datasets for initial exploration
- **Version Conflicts**: Use virtual environments for dependency management

### Getting Help
- Check the main project README for setup instructions
- Review existing notebooks for examples and patterns
- Use the project's issue tracker for technical problems
- Consult MLflow UI for experiment history and results 