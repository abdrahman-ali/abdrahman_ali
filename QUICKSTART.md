# Quick Start Guide

Get started with the ML/AI Pipeline in 5 minutes!

## Step 1: Installation (2 minutes)

```bash
# Clone or download the repository
git clone <your-repo-url>
cd abdrahman_ali

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Step 2: Add Your Data (1 minute)

Place your data file in the `data/raw/` directory:

```bash
# Example: Copy your CSV file
cp /path/to/your/data.csv data/raw/my_data.csv
```

## Step 3: Configure (1 minute)

Edit `config/config.yaml`:

```yaml
data:
  raw_data_path: "data/raw/my_data.csv"

ml:
  task_type: "classification"  # or "regression"
  target_column: "target"      # Your target column name
```

## Step 4: Run! (1 minute)

### Option A: Full Pipeline (Recommended for first time)

```bash
python main.py --mode full
```

This will:
1. Load your data
2. Run exploratory data analysis
3. Train multiple models
4. Evaluate and compare models
5. Save the best models

### Option B: Step by Step

```bash
# 1. Run EDA first
python main.py --mode eda

# 2. Check generated plots in docs/reports/figures/

# 3. Train models
python main.py --mode train

# 4. Serve models via API
python main.py --mode serve
# Access at: http://localhost:8000
```

### Option C: Using Jupyter Notebooks

```bash
# Start Jupyter
jupyter lab

# Open: notebooks/01_getting_started.ipynb
# Follow the interactive tutorial
```

## Step 5: Next Steps

### Make Predictions

```bash
python main.py --mode predict \
    --model models/saved_models/random_forest.pkl \
    --data data/raw/new_data.csv \
    --output predictions.csv
```

### Start API Server

```bash
python main.py --mode serve

# In another terminal, test the API:
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[1.0, 2.0, 3.0, ...]],
    "model_name": "random_forest"
  }'
```

### Use Docker

```bash
# Development environment with Jupyter
docker-compose up jupyter
# Access: http://localhost:8888

# Or serve models
docker-compose up api
# Access: http://localhost:8000
```

### Track Experiments with MLflow

```bash
# Start MLflow UI
mlflow ui --port 5000
# Access: http://localhost:5000

# Or with Docker
docker-compose up mlflow
```

## Common Tasks

### Train Specific Model

```python
from src.ml import MLTrainer

trainer = MLTrainer(task_type='classification')
model = trainer.train(X_train, y_train, 'xgboost')
```

### Hyperparameter Tuning

```python
from src.ml import HyperparameterTuner, get_default_param_space
from sklearn.ensemble import RandomForestClassifier

param_space = get_default_param_space('random_forest')
tuner = HyperparameterTuner(RandomForestClassifier, n_trials=50)
best_params = tuner.tune(X_train, y_train, param_space)
```

### Deep Learning

```python
from src.deep_learning import FeedForwardNN, PyTorchTrainer

model = FeedForwardNN(
    input_size=20,
    hidden_layers=[128, 64, 32],
    output_size=2
)

trainer = PyTorchTrainer(model)
history = trainer.train(train_loader, val_loader, epochs=100)
```

### Reinforcement Learning

```python
from src.reinforcement_learning import DQNAgent
import gym

env = gym.make('CartPole-v1')
agent = DQNAgent(
    state_size=env.observation_space.shape[0],
    action_size=env.action_space.n
)
history = agent.train(env, episodes=1000)
```

## Troubleshooting

### Import Errors

```bash
# Make sure you're in the project root and installed requirements
pip install -r requirements.txt
```

### CUDA/GPU Issues

```bash
# For PyTorch CPU-only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For TensorFlow CPU-only
pip install tensorflow-cpu
```

### Data Loading Issues

- Check file path in config
- Ensure data file exists in `data/raw/`
- Verify file format is supported (CSV, JSON, Parquet, Excel)

## Getting Help

1. Check the full [README.md](README.md) for detailed documentation
2. Explore [notebooks/](notebooks/) for examples
3. Review [config/config.yaml](config/config.yaml) for all options
4. Look at [src/](src/) modules for API documentation

## What's Included

✅ Data loading for multiple formats
✅ Data validation and quality checks
✅ Automated EDA and visualization
✅ ML training (10+ algorithms)
✅ Hyperparameter tuning with Optuna
✅ Deep Learning (PyTorch & TensorFlow)
✅ Reinforcement Learning (DQN)
✅ Model evaluation and comparison
✅ REST API for model serving
✅ MLflow experiment tracking
✅ Docker deployment
✅ Testing infrastructure

## That's It!

You now have a complete ML/AI pipeline. Just add your data and start building!

**Happy Coding! 🚀**
