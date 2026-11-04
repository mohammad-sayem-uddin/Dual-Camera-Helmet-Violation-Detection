"""Temporal Verification Package for Context-Aware Motorcycle Helmet Violation Detection."""
from src.temporal.state_machine import TemporalState, TemporalStateMachine, StateMachineConfig
from src.temporal.confidence_aggregator import (
    BaseConfidenceAggregator,
    SimpleVotingAggregator,
    ConfidenceWeightedAggregator,
    EMAAggregator,
    create_aggregator,
)
from src.temporal.rider_tracker import RiderTracker, Track, BoundingBox
from src.temporal.violation_decision import ViolationDecisionEngine, ViolationAlert, DecisionConfig
from src.temporal.temporal_engine import TemporalVerificationEngine, TemporalEngineConfig

__all__ = [
    "TemporalState",
    "TemporalStateMachine",
    "StateMachineConfig",
    "BaseConfidenceAggregator",
    "SimpleVotingAggregator",
    "ConfidenceWeightedAggregator",
    "EMAAggregator",
    "create_aggregator",
    "RiderTracker",
    "Track",
    "BoundingBox",
    "ViolationDecisionEngine",
    "ViolationAlert",
    "DecisionConfig",
    "TemporalVerificationEngine",
    "TemporalEngineConfig",
]
