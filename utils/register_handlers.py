"""
Register all document handlers with the factory.
This module should be imported to register handlers.
"""

from utils.document_handlers import DocumentHandlerFactory
from utils.docx_handler import DOCXHandler
from utils.excel_handler import ExcelSectionHandler
from utils.csv_handler import CSVSectionHandler
from utils.markdown_section_handler import MarkdownSectionHandler

# Register all handlers
DocumentHandlerFactory.register_handler(ExcelSectionHandler())
DocumentHandlerFactory.register_handler(CSVSectionHandler())
DocumentHandlerFactory.register_handler(MarkdownSectionHandler())

# Note: DOCXHandler is registered separately since it's not a BaseDocumentHandler
# We'll handle DOCX files specially in the workflow

