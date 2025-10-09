"""Simple protobuf stubs for testing."""

from .change_event_pb2 import *

class IndexResponse:
    def __init__(self):
        self.success = False
        self.message = ""
        self.index_id = ""
        self.indexed_at = None

class SearchResponse:
    def __init__(self):
        self.success = False
        self.message = ""
        self.results = []
