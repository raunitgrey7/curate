"""FastAPI serving layer: recommendations, similar items, explanations, health and Prometheus metrics."""

from curate.serving.app import create_app

__all__ = ["create_app"]
