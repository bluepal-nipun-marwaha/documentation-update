"""Simple audit service for testing."""

import grpc
from datetime import datetime
import structlog


logger = structlog.get_logger("AuditService")

class AuditService:
    def __init__(self) -> None:
        """Initialize the audit service."""
        self.logger = logger

    def LogEvent(self, request, context):
        """Log an audit event."""
        try:
            self.logger.info(
                "Audit event logged",
                event_type=request.event_type,
                resource_id=request.resource_id
            )

            from server.proto import audit_pb2
            resp = audit_pb2.AuditResponse()
            resp.success = True
            resp.message = "Audit event logged successfully"
            resp.audit_id = "audit-123"
            resp.timestamp.GetCurrentTime()

            return resp

        except Exception as e:
            log_error(self.logger, e, {"operation": "log_event"})
            from server.proto import audit_pb2
            resp = audit_pb2.AuditResponse()
            resp.success = False
            resp.message = f"Audit logging failed: {str(e)}"
            return resp

    def GetAuditHistory(self, request, context):
        """Get audit history."""
        from server.proto import audit_pb2
        return audit_pb2.AuditHistoryResponse()
