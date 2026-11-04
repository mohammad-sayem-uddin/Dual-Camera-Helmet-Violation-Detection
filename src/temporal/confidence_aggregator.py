"""Temporal Confidence Aggregators for Multi-Frame Evidence Smoothing."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class Observation:
    """A single detection observation."""
    class_name: str
    confidence: float
    is_detected: bool


class BaseConfidenceAggregator(ABC):
    """Abstract base class for temporal confidence aggregation algorithms."""

    def __init__(self, window_size: int = 10) -> None:
        self.window_size = window_size
        self.history: deque[Observation] = deque(maxlen=window_size)

    def add_observation(self, class_name: str, confidence: float, is_detected: bool = True) -> None:
        """Append a new frame observation into the history."""
        self.history.append(Observation(class_name=class_name, confidence=float(confidence), is_detected=is_detected))

    @abstractmethod
    def get_aggregated_confidence(self) -> dict[str, float]:
        """Compute aggregated confidence scores across the temporal window."""
        raise NotImplementedError

    def reset(self) -> None:
        """Clear temporal history."""
        self.history.clear()

    def __len__(self) -> int:
        return len(self.history)


class SimpleVotingAggregator(BaseConfidenceAggregator):
    """
    Computes aggregated confidence as the unweighted proportion of positive detections
    within the sliding window.
    """

    def get_aggregated_confidence(self) -> dict[str, float]:
        valid_obs = [obs for obs in self.history if obs.is_detected]
        if not valid_obs:
            return {"no_helmet": 0.0, "helmet": 0.0}

        total = float(len(valid_obs))
        no_h_count = sum(1 for obs in valid_obs if obs.class_name == "no_helmet")
        h_count = sum(1 for obs in valid_obs if obs.class_name == "helmet")

        return {
            "no_helmet": round(no_h_count / total, 4),
            "helmet": round(h_count / total, 4),
        }


class ConfidenceWeightedAggregator(BaseConfidenceAggregator):
    """
    Weights each detection by its model confidence score so high-confidence predictions
    dominate over marginal or ambiguous detections.
    """

    def get_aggregated_confidence(self) -> dict[str, float]:
        valid_obs = [obs for obs in self.history if obs.is_detected]
        if not valid_obs:
            return {"no_helmet": 0.0, "helmet": 0.0}

        total_conf = sum(obs.confidence for obs in valid_obs)
        if total_conf <= 1e-8:
            return {"no_helmet": 0.0, "helmet": 0.0}

        no_h_weight = sum(obs.confidence for obs in valid_obs if obs.class_name == "no_helmet")
        h_weight = sum(obs.confidence for obs in valid_obs if obs.class_name == "helmet")

        return {
            "no_helmet": round(no_h_weight / total_conf, 4),
            "helmet": round(h_weight / total_conf, 4),
        }


class EMAAggregator(BaseConfidenceAggregator):
    """
    Applies an Exponential Moving Average (EMA) with smoothing factor alpha.
    Produces smooth, continuous confidence tracks responsive to real trends while
    filtering high-frequency noise spikes.
    """

    def __init__(self, window_size: int = 15, alpha: float = 0.35) -> None:
        super().__init__(window_size=window_size)
        self.alpha = float(alpha)
        self.ema_score: float = 0.0  # -1.0 (pure no_helmet) to +1.0 (pure helmet)
        self.initialized: bool = False

    def add_observation(self, class_name: str, confidence: float, is_detected: bool = True) -> None:
        super().add_observation(class_name, confidence, is_detected)
        if not is_detected:
            return

        # Target value: +confidence for helmet, -confidence for no_helmet
        val = float(confidence) if class_name == "helmet" else -float(confidence)

        if not self.initialized:
            self.ema_score = val
            self.initialized = True
        else:
            self.ema_score = (self.alpha * val) + ((1.0 - self.alpha) * self.ema_score)

    def get_aggregated_confidence(self) -> dict[str, float]:
        if not self.initialized or not any(obs.is_detected for obs in self.history):
            return {"no_helmet": 0.0, "helmet": 0.0}

        # Linear mapping from [-1.0, 1.0] to [0.0, 1.0] probabilities
        # score = +1.0 -> 100% helmet; score = 0.0 -> 50/50; score = -1.0 -> 100% no_helmet
        clamped_score = max(-1.0, min(1.0, self.ema_score))
        p_helmet = (clamped_score + 1.0) / 2.0
        p_no_helmet = 1.0 - p_helmet

        return {
            "helmet": round(p_helmet, 4),
            "no_helmet": round(p_no_helmet, 4),
        }

    def reset(self) -> None:
        super().reset()
        self.ema_score = 0.0
        self.initialized = False


def create_aggregator(method: str = "confidence_weighted", window_size: int = 10, **kwargs: Any) -> BaseConfidenceAggregator:
    """Factory helper to instantiate an aggregator by name."""
    m = method.lower()
    if m in ("voting", "simple_voting"):
        return SimpleVotingAggregator(window_size=window_size)
    elif m in ("weighted", "confidence_weighted"):
        return ConfidenceWeightedAggregator(window_size=window_size)
    elif m in ("ema", "exponential"):
        alpha = kwargs.get("alpha", 0.35)
        return EMAAggregator(window_size=window_size, alpha=alpha)
    else:
        raise ValueError(f"Unknown aggregator method: {method}. Choose from 'voting', 'weighted', 'ema'.")
