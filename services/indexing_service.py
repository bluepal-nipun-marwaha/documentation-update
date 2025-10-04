"""Simple indexing service for testing."""

import grpc
from datetime import datetime
import structlog

logger = structlog.get_logger("IndexingService")

class IndexingService:
    def __init__(self) -> None:
        """Initialize the indexing service."""
        self.logger = logger

    def IndexDocument(self, request, context):
        """Index a document for semantic search."""
        try:
            self.logger.info(
                "Indexing document",
                document_id=request.document_id
            )

            from server.proto import indexing_pb2
            resp = indexing_pb2.IndexResponse()
            resp.success = True
            resp.message = "Document indexed successfully"
            resp.index_id = "index-123"
            resp.indexed_at.GetCurrentTime()

            return resp

        except Exception as e:
            self.logger.error("Document indexing failed", error=str(e), operation="index_document")
            from server.proto import indexing_pb2
            resp = indexing_pb2.IndexResponse()
            resp.success = False
            resp.message = f"Document indexing failed: {str(e)}"
            return resp

    def SearchDocuments(self, request, context):
        """Search documents using semantic search."""
        from server.proto import indexing_pb2
        resp = indexing_pb2.SearchResponse()
        resp.success = True
        resp.message = "Search completed"
        resp.results.extend(["doc1", "doc2"])
        return resp
