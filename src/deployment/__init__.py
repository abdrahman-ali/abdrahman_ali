"""Deployment module."""
from .api import ModelServer, create_model_server

__all__ = [
    'ModelServer',
    'create_model_server',
]
