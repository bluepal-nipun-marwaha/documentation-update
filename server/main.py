# -*- coding: utf-8 -*-
import grpc
from concurrent import futures
import os
import threading
from flask import Flask, request, jsonify
import json

# Import all gRPC services
from ..services.change_event_service import ChangeEventService
from ..services.impact_assessment_service import ImpactAssessmentService
from ..services.document_update_service import DocumentUpdateService
from ..services.audit_service import AuditService
from ..services.indexing_service import IndexingService

# Import gRPC service definitions
from .proto import (
    change_event_pb2_grpc,
    impact_assessment_pb2_grpc,
    document_update_pb2_grpc,
    audit_pb2_grpc,
    indexing_pb2_grpc
)

# Import configuration
from ..utils.config import get_settings
from ..utils.logger import get_logger, log_error

# Import protobuf messages
from .proto import change_event_pb2
from google.protobuf.timestamp_pb2 import Timestamp

def create_grpc_server():
    """Create and configure the gRPC server with all services."""
    config = get_settings()
    logger = get_logger()
    
    logger.info("Starting gRPC server with all services...")
    
    # Create server with thread pool
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=config.server.max_workers)
    )
    
    # Register all services
    change_event_pb2_grpc.add_ChangeEventServiceServicer_to_server(
        ChangeEventService(), server
    )
    impact_assessment_pb2_grpc.add_ImpactAssessmentServiceServicer_to_server(
        ImpactAssessmentService(), server
    )
    document_update_pb2_grpc.add_DocumentUpdateServiceServicer_to_server(
        DocumentUpdateService(), server
    )
    audit_pb2_grpc.add_AuditServiceServicer_to_server(
        AuditService(), server
    )
    indexing_pb2_grpc.add_IndexingServiceServicer_to_server(
        IndexingService(), server
    )
    
    # Configure listening address
    listen_addr = f"{config.server.host}:{config.server.port}"
    server.add_insecure_port(listen_addr)
    
    logger.info(f"gRPC server listening on {listen_addr}")
    logger.info("Registered services: ChangeEvent, ImpactAssessment, DocumentUpdate, Audit, Indexing")
    
    return server

def create_webhook_app():
    """Create Flask app for webhook endpoints."""
    app = Flask(__name__)
    logger = get_logger()
    config = get_settings()
    
    # Create gRPC channel for internal service calls
    grpc_channel = grpc.insecure_channel(f"{config.server.host}:{config.server.port}")
    change_event_stub = change_event_pb2_grpc.ChangeEventServiceStub(grpc_channel)
    
    def validate_github_signature(payload, signature):
        """Validate GitHub webhook signature."""
        import hmac
        import hashlib
        
        if not config.vcs.webhook_secret:
            return True  # Skip validation if no secret configured
            
        expected_signature = 'sha256=' + hmac.new(
            config.vcs.webhook_secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    
    def process_github_payload(payload):
        """Extract relevant data from GitHub webhook payload."""
        if payload.get('head_commit'):
            commit = payload['head_commit']
            return {
                'commit_hash': commit.get('id', ''),
                'commit_message': commit.get('message', ''),
                'author': commit.get('author', {}).get('name', ''),
                'timestamp': commit.get('timestamp', ''),
                'repository_url': payload.get('repository', {}).get('html_url', ''),
                'branch': payload.get('ref', '').replace('refs/heads/', ''),
                'commits': payload.get('commits', []),
                'compare_url': payload.get('compare', '')
            }
        return None
    
    def process_gitlab_payload(payload):
        """Extract relevant data from GitLab webhook payload."""
        return {
            'commit_hash': payload.get('after', ''),
            'commit_message': payload.get('message', ''),
            'author': payload.get('author', {}).get('name', ''),
            'timestamp': payload.get('timestamp', ''),
            'repository_url': payload.get('project', {}).get('web_url', ''),
            'branch': payload.get('ref', '').replace('refs/heads/', ''),
            'commits': payload.get('commits', [])
        }
    
    @app.route('/webhook', methods=['POST'])
    def webhook():
        """Handle webhook events from any provider."""
        try:
            payload = request.get_json()
            headers = dict(request.headers)
            raw_payload = request.get_data()
            
            # Detect provider based on headers
            github_event = headers.get('X-GitHub-Event')
            gitlab_event = headers.get('X-Gitlab-Event')
            
            if github_event:
                return handle_github_webhook(payload, headers, raw_payload)
            elif gitlab_event:
                return handle_gitlab_webhook(payload, headers)
            else:
                logger.warning("Unknown webhook provider", headers=dict(headers))
                return jsonify({"status": "error", "message": "Unknown webhook provider"}), 400
                
        except Exception as e:
            log_error(logger, e, {"operation": "webhook"})
            return jsonify({"status": "error", "message": str(e)}), 500
    
    def handle_github_webhook(payload, headers, raw_payload):
        """Handle GitHub webhook events."""
        try:
            # Validate signature
            signature = headers.get('X-Hub-Signature-256', '')
            if not validate_github_signature(raw_payload, signature):
                logger.warning("Invalid GitHub webhook signature")
                return jsonify({"status": "error", "message": "Invalid signature"}), 401
            
            event_type = headers.get('X-GitHub-Event')
            repository = payload.get('repository', {}).get('full_name', 'unknown')
            
            logger.info("Received GitHub webhook", 
                       event_type=event_type,
                       repository=repository,
                       commits_count=len(payload.get('commits', [])))
            
            # Process push events
            if event_type == 'push':
                commit_data = process_github_payload(payload)
                if commit_data:
                    # Extract first word from commit message
                    first_word = commit_data['commit_message'].split()[0] if commit_data['commit_message'] else ''
                    
                    # Create gRPC request
                    grpc_request = {
                        'change_id': commit_data['commit_hash'],
                        'provider': 'GITHUB',
                        'event_type': 'COMMIT',
                        'repository_url': commit_data['repository_url'],
                        'commit_hash': commit_data['commit_hash'],
                        'commit_message': commit_data['commit_message'],
                        'author': commit_data['author'],
                        'first_word': first_word,
                        'branch': commit_data['branch'],
                        'commits': commit_data['commits']
                    }
                    
                    logger.info("Processing GitHub commit", 
                               commit_hash=commit_data['commit_hash'],
                               first_word=first_word,
                               author=commit_data['author'])
                    
                    # Call gRPC ChangeEventService
                    try:
                        # Create protobuf request
                        grpc_req = change_event_pb2.ChangeEventRequest()
                        grpc_req.change_id = grpc_request['change_id']
                        grpc_req.provider = change_event_pb2.GITHUB
                        grpc_req.event_type = change_event_pb2.COMMIT
                        grpc_req.repository_url = grpc_request['repository_url']
                        grpc_req.commit_hash = grpc_request['commit_hash']
                        grpc_req.commit_message = grpc_request['commit_message']
                        grpc_req.author = grpc_request['author']
                        grpc_req.first_word = grpc_request['first_word']
                        grpc_req.branch = grpc_request['branch']
                        
                        # Set timestamp
                        timestamp = Timestamp()
                        timestamp.GetCurrentTime()
                        grpc_req.timestamp.CopyFrom(timestamp)
                        
                        # Call the service
                        response = change_event_stub.SubmitChange(grpc_req)
                        
                        logger.info("GitHub webhook processed successfully", 
                                   change_id=grpc_request['change_id'],
                                   success=response.success,
                                   impacted_entities=list(response.impacted_entities))
                        
                    except Exception as grpc_error:
                        logger.error("Failed to call gRPC service", error=str(grpc_error))
                        # Continue processing even if gRPC call fails
            
            return jsonify({
                "status": "success", 
                "message": f"GitHub {event_type} webhook processed",
                "repository": repository
            })
            
        except Exception as e:
            log_error(logger, e, {"operation": "github_webhook"})
            return jsonify({"status": "error", "message": str(e)}), 500
    
    def handle_gitlab_webhook(payload, headers):
        """Handle GitLab webhook events."""
        try:
            event_type = headers.get('X-Gitlab-Event')
            project = payload.get('project', {}).get('name', 'unknown')
            
            logger.info("Received GitLab webhook",
                       event_type=event_type,
                       project=project,
                       commits_count=len(payload.get('commits', [])))
            
            # Process push events
            if event_type == 'Push Hook':
                commit_data = process_gitlab_payload(payload)
                
                # Extract first word from commit message
                first_word = commit_data['commit_message'].split()[0] if commit_data['commit_message'] else ''
                
                # Create gRPC request
                grpc_request = {
                    'change_id': commit_data['commit_hash'],
                    'provider': 'GITLAB',
                    'event_type': 'COMMIT',
                    'repository_url': commit_data['repository_url'],
                    'commit_hash': commit_data['commit_hash'],
                    'commit_message': commit_data['commit_message'],
                    'author': commit_data['author'],
                    'first_word': first_word,
                    'branch': commit_data['branch'],
                    'commits': commit_data['commits']
                }
                
                logger.info("Processing GitLab commit", 
                           commit_hash=commit_data['commit_hash'],
                           first_word=first_word,
                           author=commit_data['author'])
                
                # Call gRPC ChangeEventService
                try:
                    # Create protobuf request
                    grpc_req = change_event_pb2.ChangeEventRequest()
                    grpc_req.change_id = grpc_request['change_id']
                    grpc_req.provider = change_event_pb2.GITLAB
                    grpc_req.event_type = change_event_pb2.COMMIT
                    grpc_req.repository_url = grpc_request['repository_url']
                    grpc_req.commit_hash = grpc_request['commit_hash']
                    grpc_req.commit_message = grpc_request['commit_message']
                    grpc_req.author = grpc_request['author']
                    grpc_req.first_word = grpc_request['first_word']
                    grpc_req.branch = grpc_request['branch']
                    
                    # Set timestamp
                    timestamp = Timestamp()
                    timestamp.GetCurrentTime()
                    grpc_req.timestamp.CopyFrom(timestamp)
                    
                    # Call the service
                    response = change_event_stub.SubmitChange(grpc_req)
                    
                    logger.info("GitLab webhook processed successfully", 
                               change_id=grpc_request['change_id'],
                               success=response.success,
                               impacted_entities=list(response.impacted_entities))
                    
                except Exception as grpc_error:
                    logger.error("Failed to call gRPC service", error=str(grpc_error))
                    # Continue processing even if gRPC call fails
            
            return jsonify({
                "status": "success", 
                "message": f"GitLab {event_type} webhook processed",
                "project": project
            })
            
        except Exception as e:
            log_error(logger, e, {"operation": "gitlab_webhook"})
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        return jsonify({"status": "healthy", "service": "rework-document-update"})
    
    @app.route('/status', methods=['GET'])
    def status():
        """Service status endpoint."""
        return jsonify({
            "service": "rework-document-update",
            "version": "0.1.0",
            "status": "running",
            "endpoints": {
                "grpc": f"0.0.0.0:{get_settings().server.port}",
                "webhooks": {
                    "unified": "/webhook"
                },
                "health": "/health"
            }
        })
    
    return app

def serve():
    """Start both gRPC server and webhook HTTP server."""
    config = get_settings()
    logger = get_logger()
    
    # Create gRPC server
    grpc_server = create_grpc_server()
    
    # Create webhook Flask app
    webhook_app = create_webhook_app()
    
    # Start gRPC server in a separate thread
    grpc_thread = threading.Thread(target=grpc_server.start)
    grpc_thread.daemon = True
    grpc_thread.start()
    
    logger.info("Starting webhook HTTP server on port 8000...")
    
    # Start Flask app (webhook server)
    webhook_app.run(
        host='0.0.0.0',
        port=8000,
        debug=False,
        threaded=True
    )

if __name__ == "__main__":
    serve()
