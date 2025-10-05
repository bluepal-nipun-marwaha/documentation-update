"""
Markdown file handler for section-level editing with header-based sections.
Supports .md and .markdown files with section identification by headers.
"""

import io
import re
from typing import Dict, List, Any, Optional
import structlog

from .document_handlers import BaseDocumentHandler, UpdateSection

logger = structlog.get_logger(__name__)

class MarkdownSectionHandler(BaseDocumentHandler):
    """Handler for Markdown files with section-level editing."""
    
    def __init__(self):
        super().__init__()
        self.logger = logger
    
    def get_file_extensions(self) -> List[str]:
        """Get supported Markdown file extensions."""
        return ['md', 'markdown']
    
    def extract_content_with_structure(self, file_bytes: bytes) -> Dict[str, Any]:
        """
        Extract Markdown content with structural information.
        
        Args:
            file_bytes: Raw Markdown file content as bytes
            
        Returns:
            Dictionary containing Markdown structure with sections, headers, and content
        """
        try:
            # Decode bytes to text
            markdown_text = file_bytes.decode('utf-8')
            
            # Parse markdown structure
            sections = self._parse_markdown_sections(markdown_text)
            
            structure = {
                'file_type': 'markdown',
                'sections': sections,
                'total_sections': len(sections),
                'raw_content': markdown_text
            }
            
            self.logger.info(f"[SUCCESS] Extracted Markdown structure: {len(sections)} sections")
            return structure
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to extract Markdown structure: {str(e)}")
            return {'file_type': 'markdown', 'error': str(e)}
    
    def _parse_markdown_sections(self, markdown_text: str) -> List[Dict[str, Any]]:
        """Parse markdown text into sections based on headers."""
        try:
            lines = markdown_text.split('\n')
            sections = []
            current_section = None
            
            for line_num, line in enumerate(lines):
                # Check if line is a header
                header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
                
                if header_match:
                    # Save previous section if exists
                    if current_section:
                        sections.append(current_section)
                    
                    # Start new section
                    header_level = len(header_match.group(1))
                    header_text = header_match.group(2).strip()
                    
                    current_section = {
                        'section_id': f"section_{len(sections)}",
                        'header_level': header_level,
                        'header_text': header_text,
                        'header_line': line_num,
                        'content_lines': [],
                        'content': '',
                        'subsections': [],
                        'tables': [],
                        'lists': [],
                        'code_blocks': []
                    }
                else:
                    # Add line to current section
                    if current_section:
                        current_section['content_lines'].append(line)
                        
                        # Detect special content types
                        self._detect_content_types(line, line_num, current_section)
                    else:
                        # Content before first header
                        if not sections:
                            sections.append({
                                'section_id': 'intro',
                                'header_level': 0,
                                'header_text': 'Introduction',
                                'header_line': 0,
                                'content_lines': [line],
                                'content': '',
                                'subsections': [],
                                'tables': [],
                                'lists': [],
                                'code_blocks': []
                            })
                        else:
                            sections[0]['content_lines'].append(line)
            
            # Save last section
            if current_section:
                sections.append(current_section)
            
            # Process content for each section
            for section in sections:
                section['content'] = '\n'.join(section['content_lines'])
            
            return sections
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to parse markdown sections: {str(e)}")
            return []
    
    def _detect_content_types(self, line: str, line_num: int, section: Dict[str, Any]):
        """Detect special content types within a section."""
        try:
            # Detect tables
            if '|' in line and line.count('|') >= 2:
                section['tables'].append({
                    'line': line_num,
                    'content': line
                })
            
            # Detect lists
            if re.match(r'^\s*[-*+]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
                section['lists'].append({
                    'line': line_num,
                    'content': line
                })
            
            # Detect code blocks
            if line.strip().startswith('```'):
                section['code_blocks'].append({
                    'line': line_num,
                    'content': line
                })
                
        except Exception as e:
            self.logger.warning(f"[WARNING] Error detecting content types: {str(e)}")
    
    def apply_section_updates(self, file_bytes: bytes, updates: List[UpdateSection]) -> bytes:
        """
        Apply targeted updates to Markdown file while preserving structure.
        
        Args:
            file_bytes: Original Markdown file content as bytes
            updates: List of UpdateSection objects describing what to update
            
        Returns:
            Updated Markdown file content as bytes
        """
        try:
            # Decode bytes to text
            markdown_text = file_bytes.decode('utf-8')
            lines = markdown_text.split('\n')
            
            # Apply updates
            for update in updates:
                self._apply_single_update(lines, update)
            
            # Reconstruct markdown
            result_text = '\n'.join(lines)
            result_bytes = result_text.encode('utf-8')
            
            self.logger.info(f"[SUCCESS] Applied {len(updates)} Markdown updates: {len(result_bytes)} bytes")
            return result_bytes
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to apply Markdown updates: {str(e)}")
            return file_bytes  # Return original if update fails
    
    def _apply_single_update(self, lines: List[str], update: UpdateSection):
        """Apply a single update to the markdown lines."""
        try:
            if update.section_type == 'section':
                self._update_section(lines, update)
            elif update.section_type == 'paragraph':
                self._update_paragraph(lines, update)
            elif update.section_type == 'header':
                self._update_header(lines, update)
            elif update.section_type == 'table':
                self._update_table(lines, update)
            else:
                self.logger.warning(f"[WARNING] Unknown Markdown update type: {update.section_type}")
                
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to apply Markdown update {update.section_id}: {str(e)}")
    
    def _update_section(self, lines: List[str], update: UpdateSection):
        """Update an entire section."""
        try:
            target = update.target_location
            section_id = target.get('section_id')
            start_line = target.get('start_line')
            end_line = target.get('end_line')
            
            if start_line is None or end_line is None:
                self.logger.error(f"[ERROR] Invalid section target: {target}")
                return
            
            if update.action == 'replace':
                # Replace section content
                new_lines = update.new_content.split('\n')
                lines[start_line:end_line+1] = new_lines
            
            elif update.action == 'append':
                # Append to section
                new_lines = update.new_content.split('\n')
                lines.insert(end_line + 1, '')  # Add blank line
                lines[end_line + 2:end_line + 2] = new_lines
            
            self.logger.info(f"[SUCCESS] Updated Markdown section {section_id}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to update Markdown section: {str(e)}")
    
    def _update_paragraph(self, lines: List[str], update: UpdateSection):
        """Update a specific paragraph."""
        try:
            target = update.target_location
            line_index = target.get('line')
            
            if line_index is None or line_index >= len(lines):
                self.logger.error(f"[ERROR] Invalid paragraph target: {target}")
                return
            
            if update.action == 'replace':
                lines[line_index] = update.new_content
            
            self.logger.info(f"[SUCCESS] Updated Markdown paragraph at line {line_index}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to update Markdown paragraph: {str(e)}")
    
    def _update_header(self, lines: List[str], update: UpdateSection):
        """Update a header."""
        try:
            target = update.target_location
            line_index = target.get('line')
            header_level = target.get('level', 1)
            
            if line_index is None or line_index >= len(lines):
                self.logger.error(f"[ERROR] Invalid header target: {target}")
                return
            
            if update.action == 'replace':
                new_header = '#' * header_level + ' ' + update.new_content
                lines[line_index] = new_header
            
            self.logger.info(f"[SUCCESS] Updated Markdown header at line {line_index}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to update Markdown header: {str(e)}")
    
    def _update_table(self, lines: List[str], update: UpdateSection):
        """Update a table."""
        try:
            target = update.target_location
            start_line = target.get('start_line')
            end_line = target.get('end_line')
            
            if start_line is None or end_line is None:
                self.logger.error(f"[ERROR] Invalid table target: {target}")
                return
            
            if update.action == 'replace':
                new_lines = update.new_content.split('\n')
                lines[start_line:end_line+1] = new_lines
            
            self.logger.info(f"[SUCCESS] Updated Markdown table")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to update Markdown table: {str(e)}")
    
    def create_section_update(self, section_id: str, start_line: int, end_line: int,
                             new_content: str, reason: str = None) -> UpdateSection:
        """Create a section update."""
        return UpdateSection(
            section_id=section_id,
            section_type="section",
            action="replace",
            new_content=new_content,
            target_location={
                'section_id': section_id,
                'start_line': start_line,
                'end_line': end_line
            },
            reason=reason
        )
    
    def create_paragraph_update(self, line_index: int, new_content: str, 
                               reason: str = None) -> UpdateSection:
        """Create a paragraph update."""
        return UpdateSection(
            section_id=f"para_{line_index}",
            section_type="paragraph",
            action="replace",
            new_content=new_content,
            target_location={'line': line_index},
            reason=reason
        )
    
    def create_header_update(self, line_index: int, header_level: int, 
                           new_content: str, reason: str = None) -> UpdateSection:
        """Create a header update."""
        return UpdateSection(
            section_id=f"header_{line_index}",
            section_type="header",
            action="replace",
            new_content=new_content,
            target_location={'line': line_index, 'level': header_level},
            reason=reason
        )

