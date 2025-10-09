"""Simple protobuf stubs for testing."""

# Import from the main protobuf file
from .change_event_pb2 import *

# Additional stubs
class ImpactAssessmentServiceStub:
    def __init__(self, channel):
        self.channel = channel

class DocumentUpdateServiceStub:
    def __init__(self, channel):
        self.channel = channel

class AuditServiceStub:
    def __init__(self, channel):
        self.channel = channel

class IndexingServiceStub:
    def __init__(self, channel):
        self.channel = channel

# Service registration functions
def add_ImpactAssessmentServiceServicer_to_server(servicer, server):
    pass

def add_DocumentUpdateServiceServicer_to_server(servicer, server):
    pass

def add_AuditServiceServicer_to_server(servicer, server):
    pass

def add_IndexingServiceServicer_to_server(servicer, server):
    pass
