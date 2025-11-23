"""
FastAPI model serving API.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import uvicorn

logger = logging.getLogger(__name__)


# Request/Response models
class PredictionRequest(BaseModel):
    """Prediction request model."""
    features: List[List[float]]
    model_name: Optional[str] = "default"


class PredictionResponse(BaseModel):
    """Prediction response model."""
    predictions: List[Any]
    model_name: str
    probabilities: Optional[List[List[float]]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: List[str]


class ModelServer:
    """Model serving server using FastAPI."""

    def __init__(self, model_dir: str = "models/saved_models"):
        """
        Initialize model server.

        Args:
            model_dir: Directory containing saved models
        """
        self.model_dir = Path(model_dir)
        self.models = {}
        self.app = FastAPI(
            title="ML Model Serving API",
            description="API for serving machine learning models",
            version="1.0.0"
        )

        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.get("/", response_model=HealthResponse)
        async def root():
            """Root endpoint."""
            return HealthResponse(
                status="healthy",
                models_loaded=list(self.models.keys())
            )

        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy",
                models_loaded=list(self.models.keys())
            )

        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict(request: PredictionRequest):
            """
            Prediction endpoint.

            Args:
                request: Prediction request

            Returns:
                Prediction response
            """
            model_name = request.model_name

            if model_name not in self.models:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{model_name}' not found. Available models: {list(self.models.keys())}"
                )

            try:
                model = self.models[model_name]
                X = np.array(request.features)

                # Make predictions
                predictions = model.predict(X)

                # Get probabilities if available
                probabilities = None
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(X).tolist()

                return PredictionResponse(
                    predictions=predictions.tolist(),
                    model_name=model_name,
                    probabilities=probabilities
                )

            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/models")
        async def list_models():
            """List available models."""
            return {"models": list(self.models.keys())}

        @self.app.post("/load_model/{model_name}")
        async def load_model(model_name: str, file_name: Optional[str] = None):
            """
            Load a model.

            Args:
                model_name: Name to assign to the model
                file_name: Model file name (defaults to model_name.pkl)
            """
            if file_name is None:
                file_name = f"{model_name}.pkl"

            model_path = self.model_dir / file_name

            if not model_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Model file '{file_name}' not found in {self.model_dir}"
                )

            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

                self.models[model_name] = model
                logger.info(f"Model '{model_name}' loaded successfully")

                return {"message": f"Model '{model_name}' loaded successfully"}

            except Exception as e:
                logger.error(f"Error loading model: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.delete("/models/{model_name}")
        async def unload_model(model_name: str):
            """
            Unload a model.

            Args:
                model_name: Name of the model to unload
            """
            if model_name not in self.models:
                raise HTTPException(
                    status_code=404,
                    detail=f"Model '{model_name}' not found"
                )

            del self.models[model_name]
            logger.info(f"Model '{model_name}' unloaded")

            return {"message": f"Model '{model_name}' unloaded successfully"}

    def load_models(self, model_files: Optional[List[str]] = None):
        """
        Load models from disk.

        Args:
            model_files: List of model file names to load (loads all if None)
        """
        if not self.model_dir.exists():
            logger.warning(f"Model directory {self.model_dir} does not exist")
            return

        if model_files is None:
            model_files = list(self.model_dir.glob("*.pkl"))
        else:
            model_files = [self.model_dir / f for f in model_files]

        for model_path in model_files:
            if not model_path.exists():
                logger.warning(f"Model file {model_path} not found")
                continue

            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

                model_name = model_path.stem
                self.models[model_name] = model
                logger.info(f"Loaded model: {model_name}")

            except Exception as e:
                logger.error(f"Error loading {model_path}: {e}")

        logger.info(f"Loaded {len(self.models)} models")

    def run(self, host: str = "0.0.0.0", port: int = 8000, workers: int = 1):
        """
        Run the server.

        Args:
            host: Host address
            port: Port number
            workers: Number of workers
        """
        logger.info(f"Starting model server on {host}:{port}")
        logger.info(f"Loaded models: {list(self.models.keys())}")

        uvicorn.run(
            self.app,
            host=host,
            port=port,
            workers=workers,
            log_level="info"
        )


def create_model_server(model_dir: str = "models/saved_models",
                       auto_load: bool = True) -> ModelServer:
    """
    Create and configure model server.

    Args:
        model_dir: Directory containing saved models
        auto_load: Whether to automatically load all models

    Returns:
        ModelServer instance
    """
    server = ModelServer(model_dir)

    if auto_load:
        server.load_models()

    return server


if __name__ == "__main__":
    # Example usage
    server = create_model_server()
    server.run()
