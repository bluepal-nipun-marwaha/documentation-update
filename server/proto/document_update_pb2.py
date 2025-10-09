"""Simple protobuf stubs for testing."""

from .change_event_pb2 import *

class UpdateResponse:
    def __init__(self):
        self.success = False
        self.message = ""
        self.update_operation_id = ""
        self.new_document_version = ""
        self.updated_at = None
        self.upload_url = ""

class PreviewResponse:
    def __init__(self):
        self.success = False
        self.message = ""

class RollbackResponse:
    def __init__(self):
        self.success = False
        self.message = ""

class UpdateStatusResponse:
    def __init__(self):
        self.status = 0
        self.message = ""
        self.progress_percentage = 0.0
        self.started_at = None
        self.completed_at = None

COMPLETED = 3
