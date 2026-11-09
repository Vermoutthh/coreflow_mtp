"""Abstract base class for latency predictors."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class BasePredictor(ABC):
    """Abstract base for all latency prediction models.

    Subclasses must implement :meth:`_load_model` and :meth:`predict`.
    Models are lazy-loaded on first access to avoid import-time I/O.
    """

    def __init__(self) -> None:
        self._model: Any = None
        self._loaded: bool = False

    @abstractmethod
    def _load_model(self) -> Any:
        """Load or build the prediction model. Called once lazily."""

    @abstractmethod
    def predict(self, **kwargs: Any) -> float:
        """Predict latency given input parameters.

        Returns:
            Predicted latency in seconds.
        """

    def _ensure_loaded(self) -> None:
        """Lazy-load the model on first access."""
        if not self._loaded:
            self._model = self._load_model()
            self._loaded = True

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded."""
        return self._loaded
