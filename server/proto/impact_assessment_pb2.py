"""Simple protobuf stubs for testing."""

from .change_event_pb2 import *

class ImpactResponse:
    def __init__(self):
        self.success = False
        self.message = ""
        self.assessment_id = ""
        self.overall_severity = 0
        self.assessed_at = None

class ImpactHistoryResponse:
    def __init__(self):
        pass

MEDIUM = 2
