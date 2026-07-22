# /nav: public package boundary.
# Exposes only the task-level facade and its domain-owned result/session types.
# Heavy model construction stays lazy inside BSMamba2Session.load().
# Reads: __about__.py and api.py.
from .__about__ import __version__
from .api import BSMamba2Session, SeparationResult, separate

__all__ = ["BSMamba2Session", "SeparationResult", "__version__", "separate"]
