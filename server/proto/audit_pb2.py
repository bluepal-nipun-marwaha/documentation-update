"""Simple protobuf stubs for testing."""

from .change_event_pb2 import *

class AuditResponse:
    def __init__(self):
        self.success = False
        self.message = ""
        self.audit_id = ""
        self.timestamp = None

class AuditHistoryResponse:
    def __init__(self):
        pass
