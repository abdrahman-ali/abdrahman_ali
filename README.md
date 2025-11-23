# Data Engineering, ML, DL, RL, AI Pipeline

A comprehensive, production-ready template for data engineering, machine learning, deep learning, reinforcement learning, and AI projects. This framework provides everything you need to build end-to-end ML/AI solutions - just add your data!

## 🚀 Features

### Data Engineering
- **Universal Data Loader**: Support for CSV, JSON, Parquet, Excel, Feather, Pickle
- **Data Validation**: Comprehensive quality checks and validation
- **Data Pipeline**: Automated preprocessing and feature engineering
- **ETL Utilities**: Extract, transform, load operations

### Data Analysis
- **Exploratory Data Analysis (EDA)**: Automated statistical analysis
- **Visualization**: Rich plotting and visualization tools
- **Reporting**: Automated report generation

### Machine Learning
- **Multiple Algorithms**: Random Forest, XGBoost, LightGBM, CatBoost, and more
- **AutoML**: Hyperparameter tuning with Optuna
- **Model Evaluation**: Comprehensive metrics and visualization
- **Cross-validation**: Built-in CV support

### Deep Learning
- **PyTorch Support**: Custom neural network architectures
- **TensorFlow/Keras Support**: Easy model building
- **CNN, RNN, LSTM**: Pre-built architectures
- **Training Utilities**: Callbacks, early stopping, checkpointing

### Reinforcement Learning
- **DQN Agent**: Deep Q-Network implementation
- **OpenAI Gym Integration**: Ready for RL experiments
- **Experience Replay**: Memory-efficient training

### Model Deployment
- **FastAPI Server**: Production-ready REST API
- **Model Registry**: Centralized model management
- **Batch Prediction**: Efficient batch processing
- **Docker Support**: Containerized deployment

### MLOps & Experiment Tracking
- **MLflow Integration**: Track experiments, parameters, metrics
- **Model Versioning**: Automatic model versioning
- **Artifact Storage**: Store and retrieve artifacts

## 📦 Installation

### Option 1: Using pip (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd abdrahman_ali

# Install dependencies
pip install -r requirements.txt

# Or install minimal version
pip install -r requirements-minimal.txt
```

### Option 2: Using Docker

```bash
# Build and run with Docker Compose
docker-compose up jupyter  # For development
docker-compose up api      # For serving models
docker-compose up training # For training
```

## 🎯 Quick Start

### 1. Configure Your Project

Edit `config/config.yaml` with your project settings:

```yaml
project:
  name: "my-ml-project"

data:
  raw_data_path: "data/raw/my_data.csv"

ml:
  task_type: "classification"  # or "regression"
  target_column: "target"
```

### 2. Add Your Data

Place your data files in the appropriate directories:

```
data/
  ├── raw/          # Your raw data files
  ├── processed/    # Processed data (auto-generated)
  ├── interim/      # Intermediate data
  └── external/     # External data sources
```

### 3. Run the Pipeline

```bash
# Full pipeline (EDA + Training + Evaluation)
python main.py --mode full --config config/config.yaml

# Only training
python main.py --mode train

# Only EDA
python main.py --mode eda

# Serve models via API
python main.py --mode serve

# Make predictions
python main.py --mode predict --model models/saved_models/model.pkl --data data/test.csv
```

## 📚 Usage Examples

### Example 1: Train a Classification Model

```python
from src.data_engineering import load_data, split_data, DataPipeline
from src.ml import MLTrainer, ModelEvaluator

# Load data
data = load_data('data/raw/data.csv')

# Prepare features and target
X = data.drop('target', axis=1)
y = data['target']

# Split data
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# Preprocess
pipeline = DataPipeline()
X_train = pipeline.fit_transform(X_train)
X_val = pipeline.transform(X_val)

# Train model
trainer = MLTrainer(task_type='classification')
model = trainer.train(X_train, y_train, 'random_forest')

# Evaluate
evaluator = ModelEvaluator(task_type='classification')
y_pred = trainer.predict('random_forest', X_val)
metrics = evaluator.evaluate(y_val, y_pred)

print(metrics)
```

### Example 2: Deep Learning with PyTorch

```python
from src.deep_learning import FeedForwardNN, PyTorchTrainer, create_data_loaders
import torch.nn as nn

# Create model
model = FeedForwardNN(
    input_size=20,
    hidden_layers=[128, 64, 32],
    output_size=10,
    dropout=0.2
)

# Create data loaders
train_loader, val_loader = create_data_loaders(
    X_train, y_train, X_val, y_val, batch_size=32
)

# Train
trainer = PyTorchTrainer(model)
history = trainer.train(
    train_loader,
    val_loader,
    epochs=100,
    learning_rate=0.001
)
```

### Example 3: Reinforcement Learning

```python
from src.reinforcement_learning import DQNAgent
import gym

# Create environment
env = gym.make('CartPole-v1')

# Create agent
agent = DQNAgent(
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.n
)

# Train agent
history = agent.train(env, episodes=1000)

# Save agent
agent.save('models/saved_models/dqn_agent.pt')
```

### Example 4: Model Serving API

```python
from src.deployment import create_model_server

# Create server
server = create_model_server(
    model_dir='models/saved_models',
    auto_load=True
)

# Run server
server.run(host='0.0.0.0', port=8000)
```

Then make requests:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[1.0, 2.0, 3.0, 4.0]],
    "model_name": "random_forest"
  }'
```

## 📁 Project Structure

```
.
├── config/                 # Configuration files
│   └── config.yaml        # Main configuration
├── data/                  # Data directory
│   ├── raw/              # Raw data
│   ├── processed/        # Processed data
│   ├── interim/          # Intermediate data
│   └── external/         # External data
├── docker/               # Docker configurations
├── docs/                 # Documentation
│   └── reports/         # Analysis reports and figures
├── logs/                # Log files
├── models/              # Model storage
│   ├── saved_models/   # Trained models
│   └── checkpoints/    # Training checkpoints
├── notebooks/          # Jupyter notebooks
├── src/               # Source code
│   ├── data_engineering/   # Data loading, validation, pipelines
│   ├── data_analysis/      # EDA and visualization
│   ├── ml/                 # Machine learning
│   ├── deep_learning/      # Deep learning (PyTorch, TensorFlow)
│   ├── reinforcement_learning/  # RL agents
│   ├── deployment/         # Model serving
│   └── utils/             # Utilities (config, tracking)
├── tests/             # Unit tests
├── scripts/           # Utility scripts
├── main.py           # Main entry point
├── Dockerfile        # Docker configuration
├── docker-compose.yml # Docker Compose configuration
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## 🔧 Configuration

The `config/config.yaml` file contains all configurable parameters:

- **Project Settings**: Name, version, description
- **Data Configuration**: Paths, validation rules, splitting
- **Feature Engineering**: Scaling, encoding, selection
- **ML Configuration**: Algorithms, hyperparameters, evaluation
- **DL Configuration**: Architecture, training parameters, callbacks
- **RL Configuration**: Environment, algorithm, training
- **Deployment**: API settings, serving configuration
- **Experiment Tracking**: MLflow, Weights & Biases
- **Logging**: Log levels, formats

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 Experiment Tracking

### Using MLflow

```bash
# Start MLflow UI
mlflow ui --port 5000

# Or using Docker
docker-compose up mlflow
```

Access at: http://localhost:5000

### Using Weights & Biases

```python
from src.utils import get_tracker

tracker = get_tracker(experiment_name='my-experiment')
tracker.start_run()
tracker.log_params({'learning_rate': 0.001})
tracker.log_metrics({'accuracy': 0.95})
tracker.end_run()
```

## 🐳 Docker Usage

```bash
# Development with Jupyter
docker-compose up jupyter
# Access at: http://localhost:8888

# Run training
docker-compose up training

# Serve models
docker-compose up api
# Access at: http://localhost:8000

# MLflow tracking
docker-compose up mlflow
# Access at: http://localhost:5000
```

## 🤝 Contributing

This is a template project. Customize it for your needs!

## 📄 License

MIT License - feel free to use this template for your projects!

## 🎓 Usage Tips

1. **Start with the config**: Customize `config/config.yaml` for your project
2. **Add your data**: Place data files in `data/raw/`
3. **Run EDA first**: Use `python main.py --mode eda` to understand your data
4. **Experiment**: Try different models and hyperparameters
5. **Track experiments**: Use MLflow to compare results
6. **Deploy**: Use the FastAPI server for production serving

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

## 🌟 Features Coming Soon

- AutoML capabilities
- Advanced feature engineering
- Time series forecasting
- Computer vision utilities
- NLP pipelines
- Distributed training
- Cloud deployment guides

---

**Happy ML/AI Development! 🚀**
