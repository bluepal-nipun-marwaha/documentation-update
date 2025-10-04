"""
ArangoDB client for managing document relationships and graph operations.
"""
import os
import json
from typing import Dict, List, Optional, Any
from arango import ArangoClient
from arango.database import StandardDatabase
from arango.graph import Graph
import structlog

logger = structlog.get_logger(__name__)

class ArangoDBClient:
    """Client for ArangoDB operations."""
    
    def __init__(self):
        self.host = os.getenv('ARANGO_HOST', 'localhost')
        self.port = int(os.getenv('ARANGO_PORT', '8529'))
        self.username = os.getenv('ARANGO_USERNAME', 'root')
        self.password = os.getenv('ARANGO_PASSWORD', 'openSesame')  # Use your actual password
        self.database_name = os.getenv('ARANGO_DATABASE', 'rework_docs')
        
        self.client = None
        self.db = None
        self.graph = None
        
    def connect(self) -> bool:
        """Connect to ArangoDB and initialize database."""
        try:
            print(f"🔗 Connecting to ArangoDB at {self.host}:{self.port}")
            
            # Initialize client
            self.client = ArangoClient(hosts=f'http://{self.host}:{self.port}')
            
            # Connect to system database
            sys_db = self.client.db('_system', username=self.username, password=self.password)
            
            # Create database if it doesn't exist
            if not sys_db.has_database(self.database_name):
                print(f"📊 Creating database: {self.database_name}")
                sys_db.create_database(self.database_name)
            else:
                print(f"📊 Database {self.database_name} already exists")
            
            # Connect to our database
            self.db = self.client.db(self.database_name, username=self.username, password=self.password)
            
            # Initialize collections and graph
            self._setup_collections()
            self._setup_graph()
            
            print("✅ ArangoDB connection established successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to connect to ArangoDB: {str(e)}")
            logger.error("ArangoDB connection failed", error=str(e))
            return False
    
    def _setup_collections(self):
        """Create necessary collections."""
        # Document collections
        document_collections = ['commits', 'documents', 'files', 'authors', 'repositories']
        # Edge collections
        edge_collections = ['commit_to_document', 'document_to_file', 'author_to_commit', 'repo_to_commit']
        
        # Create document collections
        for name in document_collections:
            if not self.db.has_collection(name):
                print(f"📁 Creating document collection: {name}")
                self.db.create_collection(name)
            else:
                print(f"📁 Document collection {name} already exists")
        
        # Create edge collections
        for name in edge_collections:
            if not self.db.has_collection(name):
                print(f"📁 Creating edge collection: {name}")
                self.db.create_collection(name, edge=True)
            else:
                print(f"📁 Edge collection {name} already exists")
    
    def _setup_graph(self):
        """Create the document relationship graph."""
        graph_name = 'document_graph'
        
        if not self.db.has_graph(graph_name):
            print(f"🕸️ Creating graph: {graph_name}")
            
            # Define edge definitions
            edge_definitions = [
                {
                    'edge_collection': 'commit_to_document',
                    'from_vertex_collections': ['commits'],
                    'to_vertex_collections': ['documents']
                },
                {
                    'edge_collection': 'document_to_file',
                    'from_vertex_collections': ['documents'],
                    'to_vertex_collections': ['files']
                },
                {
                    'edge_collection': 'author_to_commit',
                    'from_vertex_collections': ['authors'],
                    'to_vertex_collections': ['commits']
                },
                {
                    'edge_collection': 'repo_to_commit',
                    'from_vertex_collections': ['repositories'],
                    'to_vertex_collections': ['commits']
                }
            ]
            
            self.graph = self.db.create_graph(graph_name, edge_definitions)
            print(f"✅ Graph {graph_name} created successfully")
        else:
            print(f"🕸️ Graph {graph_name} already exists")
            self.graph = self.db.graph(graph_name)
    
    def store_commit(self, commit_data: Dict[str, Any]) -> str:
        """Store a commit in the database."""
        try:
            # Store commit - ArangoDB keys must be alphanumeric and start with letter
            commit_key = commit_data['commit_hash'][:20].replace('-', '').replace('_', '')
            if not commit_key[0].isalpha():
                commit_key = 'c' + commit_key[1:]  # Ensure starts with letter
            
            commit_doc = {
                '_key': commit_key,
                'hash': commit_data['commit_hash'],
                'message': commit_data['commit_message'],
                'author': commit_data['author'],
                'timestamp': commit_data['timestamp'],
                'branch': commit_data['branch'],
                'repository_url': commit_data['repository_url']
            }
            
            result = self.db.collection('commits').insert(commit_doc)
            print(f"💾 Stored commit: {commit_data['commit_hash'][:8]}")
            
            # Store author if not exists
            self._store_author(commit_data['author'])
            
            # Store repository if not exists
            self._store_repository(commit_data['repository_url'])
            
            # Create relationships
            self._create_commit_relationships(commit_doc['_key'], commit_data)
            
            return result['_key']
            
        except Exception as e:
            print(f"❌ Failed to store commit: {str(e)}")
            logger.error("Failed to store commit", error=str(e), commit_hash=commit_data.get('commit_hash'))
            raise
    
    def _store_author(self, author_name: str):
        """Store author if not exists."""
        # Create safe key for author
        author_key = author_name.replace(' ', '_').replace('-', '_').replace('.', '_')
        if not author_key[0].isalpha():
            author_key = 'a' + author_key[1:]
        
        if not self.db.collection('authors').find({'_key': author_key}):
            author_doc = {
                '_key': author_key,
                'name': author_name
            }
            self.db.collection('authors').insert(author_doc)
            print(f"👤 Stored author: {author_name}")
    
    def _store_repository(self, repo_url: str):
        """Store repository if not exists."""
        # Create safe key for repository
        repo_key = repo_url.split('/')[-1].replace('-', '_').replace('.', '_')
        if not repo_key[0].isalpha():
            repo_key = 'r' + repo_key[1:]
        
        if not self.db.collection('repositories').find({'_key': repo_key}):
            repo_doc = {
                '_key': repo_key,
                'url': repo_url,
                'name': repo_url.split('/')[-1]
            }
            self.db.collection('repositories').insert(repo_doc)
            print(f"📦 Stored repository: {repo_key}")
    
    def _create_commit_relationships(self, commit_key: str, commit_data: Dict[str, Any]):
        """Create relationships between commit and other entities."""
        try:
            # Create safe keys for relationships
            author_key = commit_data["author"].replace(' ', '_').replace('-', '_').replace('.', '_')
            if not author_key[0].isalpha():
                author_key = 'a' + author_key[1:]
            
            repo_key = commit_data['repository_url'].split('/')[-1].replace('-', '_').replace('.', '_')
            if not repo_key[0].isalpha():
                repo_key = 'r' + repo_key[1:]
            
            # Author to commit relationship
            author_edge = {
                '_from': f'authors/{author_key}',
                '_to': f'commits/{commit_key}',
                'relationship': 'authored'
            }
            self.db.collection('author_to_commit').insert(author_edge)
            
            # Repository to commit relationship
            repo_edge = {
                '_from': f'repositories/{repo_key}',
                '_to': f'commits/{commit_key}',
                'relationship': 'contains'
            }
            self.db.collection('repo_to_commit').insert(repo_edge)
            
            print(f"🔗 Created relationships for commit: {commit_key}")
            
        except Exception as e:
            print(f"❌ Failed to create relationships: {str(e)}")
            logger.error("Failed to create relationships", error=str(e))
    
    def store_document_mapping(self, commit_key: str, files_to_update: List[str]):
        """Store document mapping for a commit."""
        try:
            for file_path in files_to_update:
                # Store document if not exists
                doc_key = file_path.replace('/', '_').replace('.', '_')
                if not self.db.collection('documents').find({'_key': doc_key}):
                    doc_doc = {
                        '_key': doc_key,
                        'path': file_path,
                        'type': 'documentation'
                    }
                    self.db.collection('documents').insert(doc_doc)
                
                # Store file if not exists
                file_key = file_path.split('/')[-1]
                if not self.db.collection('files').find({'_key': file_key}):
                    file_doc = {
                        '_key': file_key,
                        'path': file_path,
                        'type': 'file'
                    }
                    self.db.collection('files').insert(file_doc)
                
                # Create commit to document relationship
                commit_to_doc_edge = {
                    '_from': f'commits/{commit_key}',
                    '_to': f'documents/{doc_key}',
                    'relationship': 'affects',
                    'file_path': file_path
                }
                self.db.collection('commit_to_document').insert(commit_to_doc_edge)
                
                # Create document to file relationship
                doc_to_file_edge = {
                    '_from': f'documents/{doc_key}',
                    '_to': f'files/{file_key}',
                    'relationship': 'contains'
                }
                self.db.collection('document_to_file').insert(doc_to_file_edge)
            
            print(f"📄 Stored document mapping for {len(files_to_update)} files")
            
        except Exception as e:
            print(f"❌ Failed to store document mapping: {str(e)}")
            logger.error("Failed to store document mapping", error=str(e))
    
    def get_commit_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent commit history."""
        try:
            query = f"""
            FOR commit IN commits
            SORT commit.timestamp DESC
            LIMIT {limit}
            RETURN commit
            """
            cursor = self.db.aql.execute(query)
            return list(cursor)
        except Exception as e:
            print(f"❌ Failed to get commit history: {str(e)}")
            return []
    
    def get_documents_affected_by_commit(self, commit_hash: str) -> List[str]:
        """Get documents affected by a specific commit."""
        try:
            query = """
            FOR commit IN commits
            FILTER commit.hash == @commit_hash
            FOR v, e, p IN 1..1 OUTBOUND commit commit_to_document
            RETURN v.path
            """
            cursor = self.db.aql.execute(query, bind_vars={'commit_hash': commit_hash})
            return list(cursor)
        except Exception as e:
            print(f"❌ Failed to get affected documents: {str(e)}")
            return []
    
    def close(self):
        """Close the database connection."""
        if self.client:
            self.client.close()
            print("🔌 ArangoDB connection closed")

# Global instance
arango_client = ArangoDBClient()
