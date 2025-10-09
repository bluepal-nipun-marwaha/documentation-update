"""Simple change event service for testing."""

import grpc
from datetime import datetime
import json
import structlog

logger = structlog.get_logger("ChangeEventService")

class ChangeEventService:
    def __init__(self) -> None:
        """Initialize the change event service."""
        self.logger = logger

    def SubmitChange(self, request, context):
        """Submit a normalized change event."""
        try:
            self.logger.info(
                "Processing change event",
                change_id=request.change_id,
                commit_hash=request.commit_hash,
                first_word=request.first_word
            )

            # Create a simple response
            from server.proto import change_event_pb2
            resp = change_event_pb2.ChangeEventResponse()
            resp.success = True
            resp.message = "Change event processed successfully"
            resp.change_event_id = request.change_id
            resp.impacted_entities.extend(["docs/README.md"])
            resp.status = change_event_pb2.PROCESSING
            
            log_audit_event(
                self.logger,
                event_type="CHANGE_EVENT_PROCESSED",
                resource_id=request.change_id,
                action="SUBMIT",
                result="SUCCESS",
                context={"files_to_update": ["docs/README.md"]}
            )

            return resp

        except Exception as e:
            log_error(self.logger, e, {"operation": "submit_change"})
            from server.proto import change_event_pb2
            resp = change_event_pb2.ChangeEventResponse()
            resp.success = False
            resp.message = f"Failed to process change event: {str(e)}"
            resp.status = change_event_pb2.FAILED
            return resp

    def IngestWebhook(self, request, context):
        """Ingest raw webhook event and normalize."""
        try:
            from server.proto import change_event_pb2
            resp = change_event_pb2.ChangeEventResponse()
            resp.success = True
            resp.message = "Webhook ingested and normalized"
            resp.change_event_id = "webhook-test"
            resp.status = change_event_pb2.QUEUED
            return resp

        except Exception as e:
            log_error(self.logger, e, {"operation": "ingest_webhook"})
            from server.proto import change_event_pb2
            resp = change_event_pb2.ChangeEventResponse()
            resp.success = False
            resp.message = f"Failed to ingest webhook: {str(e)}"
            resp.status = change_event_pb2.FAILED
            return resp
