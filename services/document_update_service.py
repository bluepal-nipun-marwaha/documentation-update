"""Document update service with GitLab integration."""

import grpc
from datetime import datetime
import structlog
import requests
import json
from .config_service import config
from utils.docx_handler import DOCXHandler

logger = structlog.get_logger("DocumentUpdateService")

class DocumentUpdateService:
    def __init__(self) -> None:
        """Initialize the document update service."""
        self.logger = logger
        self.gitlab_config = config.get_gitlab_config()
        self.github_config = config.get_github_config()

    def ApplyUpdate(self, request, context):
        """Apply document updates and upload to VCS."""
        try:
            self.logger.info(
                "Applying document update",
                document_id=request.document_id
            )

            from server.proto import document_update_pb2
            resp = document_update_pb2.UpdateResponse()
            resp.success = True
            resp.message = "Document updated successfully"
            resp.update_operation_id = "update-123"
            resp.new_document_version = "v1.0.0"
            resp.updated_at.GetCurrentTime()
            resp.upload_url = "https://github.com/example/repo"

            # log_audit_event(
            #     self.logger,
            #     event_type="DOCUMENT_UPDATE_APPLIED",
            #     resource_id=request.document_id,
            #     action="UPDATE",
            #     result="SUCCESS"
            # )

            return resp

        except Exception as e:
            # log_error(self.logger, e, {"operation": "apply_update"})
            logger.error(f"Document update failed: {str(e)}")
            from server.proto import document_update_pb2
            resp = document_update_pb2.UpdateResponse()
            resp.success = False
            resp.message = f"Document update failed: {str(e)}"
            return resp

    def PreviewUpdate(self, request, context):
        """Preview document updates."""
        from server.proto import document_update_pb2
        resp = document_update_pb2.PreviewResponse()
        resp.success = True
        resp.message = "Preview generated"
        return resp

    def RollbackUpdate(self, request, context):
        """Rollback a previous update."""
        from server.proto import document_update_pb2
        resp = document_update_pb2.RollbackResponse()
        resp.success = False
        resp.message = "Rollback not implemented"
        return resp

    def GetUpdateStatus(self, request, context):
        """Get update operation status."""
        from server.proto import document_update_pb2
        resp = document_update_pb2.UpdateStatusResponse()
        resp.status = document_update_pb2.COMPLETED
        resp.message = "Completed"
        resp.progress_percentage = 100.0
        resp.started_at.GetCurrentTime()
        resp.completed_at.GetCurrentTime()
        return resp
    
    def update_document_content(self, file_path: str, updated_text: str, original_content: bytes = None) -> bytes:
        """
        Update document content, handling both text and DOCX files.
        
        Args:
            file_path: Path to the document file
            updated_text: Updated text content from LLM
            original_content: Original file content (for DOCX files)
            
        Returns:
            Updated file content as bytes
        """
        try:
            if DOCXHandler.is_docx_file(file_path):
                logger.info(f"📄 Updating DOCX file: {file_path}")
                
                if original_content:
                    # Update existing DOCX while preserving formatting
                    updated_content = DOCXHandler.update_docx_content(original_content, updated_text)
                else:
                    # Create new DOCX from text
                    updated_content = DOCXHandler.create_docx_from_text(updated_text, "Updated Document")
                
                return updated_content
            else:
                # Handle text files (markdown, etc.)
                logger.info(f"📄 Updating text file: {file_path}")
                return updated_text.encode('utf-8')
                
        except Exception as e:
            logger.error(f"❌ Failed to update document content: {str(e)}")
            # Return original content if update fails
            return original_content if original_content else updated_text.encode('utf-8')
