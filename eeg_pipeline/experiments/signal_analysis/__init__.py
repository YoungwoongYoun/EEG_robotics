"""Leakage-controlled task-relevant EEG signal analysis."""

from .config import SignalAnalysisConfig, load_signal_analysis_config
from .runner import run_signal_analysis

__all__ = ["SignalAnalysisConfig", "load_signal_analysis_config", "run_signal_analysis"]
