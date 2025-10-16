"""
DOCX Paragraph Inserter Module - Fixed Version

Handles adding new paragraphs to DOCX documents by duplicating existing paragraphs.
Based on the proven approach from double_para.py
"""

import io
import logging
from typing import List, Dict, Any, Optional
from docx import Document
from docx.shared import RGBColor

logger = logging.getLogger(__name__)


class DocxParagraphInserter:
    """Handles adding new paragraphs to DOCX documents by duplicating existing ones."""
    
    def __init__(self):
        self.logger = logger
    
    def insert_new_paragraphs(self, docx_bytes: bytes, commit_context: Dict[str, Any], llm_service) -> bytes:
        """
        Main entry point for inserting new paragraphs into a DOCX document.
        
        Args:
            docx_bytes: Original DOCX file as bytes
            commit_context: Commit information and context
            llm_service: LLM service for generating content
            
        Returns:
            Updated DOCX file as bytes
        """
        try:
            # Load document from bytes
            doc = Document(io.BytesIO(docx_bytes))
            self.logger.info(f"Loaded DOCX document with {len(doc.paragraphs)} paragraphs")
            
            # Find insertion point
            insertion_point = self._find_insertion_point(doc)
            if insertion_point is None:
                self.logger.warning("No suitable insertion point found")
                return docx_bytes
            
            # Generate new content
            new_content = self._generate_new_content(commit_context, llm_service)
            if not new_content:
                self.logger.warning("No new content generated")
                return docx_bytes
            
            # Insert new paragraphs
            self._insert_paragraphs_at_point(doc, insertion_point, new_content)
            
            self.logger.info(f"Inserted new content at position {insertion_point}")
            
            # Save document to bytes
            output_stream = io.BytesIO()
            doc.save(output_stream)
            return output_stream.getvalue()
            
        except Exception as e:
            self.logger.error(f"Error inserting paragraphs into DOCX document: {e}")
            return docx_bytes
    
    def _find_insertion_point(self, doc: Document) -> Optional[int]:
        """
        Find the best insertion point for new content.
        
        Args:
            doc: Document object
            
        Returns:
            Index where to insert new content, or None if no suitable point found
        """
        # Look for a good insertion point (e.g., before the last paragraph)
        if len(doc.paragraphs) > 0:
            # Insert before the last paragraph
            return len(doc.paragraphs) - 1
        
        return None
    
    def _generate_new_content(self, commit_context: Dict[str, Any], llm_service) -> List[str]:
        """
        Generate new content based on commit context.
        
        Args:
            commit_context: Commit information
            llm_service: LLM service
            
        Returns:
            List of paragraph texts to insert
        """
        try:
            commit_message = commit_context.get('message', '')
            files_changed = commit_context.get('files_changed', [])
            
            prompt = f"""
Based on the commit information, generate new documentation content:

COMMIT INFORMATION:
- Message: {commit_message}
- Files Changed: {', '.join(files_changed)}

INSTRUCTIONS:
- Generate 2-3 paragraphs of documentation content that explains the changes
- Write in a professional, technical documentation style
- Focus on what was added or changed and why it's important
- Each paragraph should be on a separate line
- Do not include markdown formatting - this is for a Word document

NEW DOCUMENTATION CONTENT:
"""
            
            response = llm_service.generate_response(prompt, temperature=0.3, max_tokens=800)
            
            # Split into paragraphs
            paragraphs = [p.strip() for p in response.split('\n') if p.strip()]
            
            # Filter out any markdown-like content
            clean_paragraphs = []
            for para in paragraphs:
                # Remove markdown formatting
                clean_para = para.replace('**', '').replace('*', '').replace('#', '').replace('`', '')
                if clean_para and len(clean_para) > 20:  # Only keep substantial paragraphs
                    clean_paragraphs.append(clean_para)
            
            self.logger.info(f"Generated {len(clean_paragraphs)} new paragraphs")
            return clean_paragraphs
            
        except Exception as e:
            self.logger.error(f"Error generating new content: {e}")
            return []
    
    def _insert_paragraphs_at_point(self, doc: Document, insertion_point: int, new_content: List[str]) -> None:
        """
        Insert new paragraphs at the specified point.
        
        Args:
            doc: Document object
            insertion_point: Index where to insert
            new_content: List of paragraph texts to insert
        """
        try:
            # Find a template paragraph to copy formatting from
            template_paragraph = None
            if insertion_point < len(doc.paragraphs):
                template_paragraph = doc.paragraphs[insertion_point]
            elif len(doc.paragraphs) > 0:
                template_paragraph = doc.paragraphs[-1]
            
            # Insert each new paragraph
            for i, content in enumerate(new_content):
                if template_paragraph:
                    # Duplicate the template paragraph
                    new_para = self._duplicate_paragraph(doc, template_paragraph, content)
                else:
                    # Create a new paragraph
                    new_para = doc.add_paragraph(content)
                
                # Insert at the correct position
                if insertion_point + i < len(doc.paragraphs):
                    # Insert before existing paragraph
                    para_element = new_para._element
                    ref_para = doc.paragraphs[insertion_point + i]
                    ref_para._element.addprevious(para_element)
                else:
                    # Add to end
                    pass  # Already added by add_paragraph
                
                self.logger.info(f"Inserted paragraph {i+1}: '{content[:50]}...'")
                
        except Exception as e:
            self.logger.error(f"Error inserting paragraphs: {e}")
    
    def _duplicate_paragraph(self, doc, template_paragraph, new_text: str):
        """
        Duplicate a paragraph with new text while preserving formatting.
        
        Args:
            doc: Document object
            template_paragraph: Paragraph to duplicate
            new_text: New text for the duplicated paragraph
            
        Returns:
            New paragraph with duplicated formatting
        """
        try:
            # Create new paragraph using the document
            new_para = doc.add_paragraph()
            
            # Copy paragraph-level formatting
            if template_paragraph.style:
                new_para.style = template_paragraph.style
            
            # Copy alignment
            if template_paragraph.alignment:
                new_para.alignment = template_paragraph.alignment
            
            # Copy formatting from template runs
            if template_paragraph.runs:
                # Use the first run's formatting as template
                template_run = template_paragraph.runs[0]
                
                # Add new text with template formatting
                new_run = new_para.add_run(new_text)
                new_run.font.bold = template_run.font.bold
                new_run.font.italic = template_run.font.italic
                new_run.font.underline = template_run.font.underline
                new_run.font.size = template_run.font.size
                
                if template_run.font.color and template_run.font.color.rgb:
                    new_run.font.color.rgb = template_run.font.color.rgb
            else:
                # No runs in template, add plain text
                new_para.add_run(new_text)
            
            return new_para
            
        except Exception as e:
            self.logger.error(f"Error duplicating paragraph: {e}")
            # Fallback: create simple paragraph
            return doc.add_paragraph(new_text)
