"""Simple protobuf stubs for testing."""

# Simple protobuf message classes for testing
class ChangeEventRequest:
    def __init__(self):
        self.change_id = ""
        self.provider = 0
        self.event_type = 0
        self.repository_url = ""
        self.commit_hash = ""
        self.commit_message = ""
        self.author = ""
        self.first_word = ""
        self.branch = ""
        self.timestamp = None

class ChangeEventResponse:
    def __init__(self):
        self.success = False
        self.message = ""
        self.change_event_id = ""
        self.impacted_entities = []
        self.status = 0

class ImpactResponse:
    def __init__(self):
        self.success = False
        self.message = ""
        self.assessment_id = ""
        self.overall_severity = 0
        self.assessed_at = None

class UpdateResponse:
    def __init__(self):
        self.success = False
        self.message = ""
        self.update_operation_id = ""
        self.new_document_version = ""
        self.updated_at = None
        self.upload_url = ""

class AuditResponse:
    def __init__(self):
        self.success = False
        self.message = ""
        self.audit_id = ""
        self.timestamp = None

class IndexResponse:
    def __init__(self):
        self.success = False
        self.message = ""
        self.index_id = ""
        self.indexed_at = None

# Enums
GITHUB = 1
GITLAB = 2
COMMIT = 1
PROCESSING = 2
QUEUED = 1
FAILED = 4
MEDIUM = 2
COMPLETED = 3

# Service stubs
class ChangeEventServiceStub:
    def __init__(self, channel):
        self.channel = channel
    
    def SubmitChange(self, request):
        return ChangeEventResponse()

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
def add_ChangeEventServiceServicer_to_server(servicer, server):
    pass

def add_ImpactAssessmentServiceServicer_to_server(servicer, server):
    pass

def add_DocumentUpdateServiceServicer_to_server(servicer, server):
    pass

def add_AuditServiceServicer_to_server(servicer, server):
    pass

def add_IndexingServiceServicer_to_server(servicer, server):
    pass
