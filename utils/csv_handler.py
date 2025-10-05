"""
CSV file handler for section-level editing with dialect preservation.
Supports .csv files with automatic dialect detection.
"""

import io
import csv
from typing import Dict, List, Any, Optional
import structlog

from .document_handlers import BaseDocumentHandler, UpdateSection

logger = structlog.get_logger(__name__)

class CSVSectionHandler(BaseDocumentHandler):
    """Handler for CSV files with section-level editing."""
    
    def __init__(self):
        super().__init__()
        self.logger = logger
    
    def get_file_extensions(self) -> List[str]:
        """Get supported CSV file extensions."""
        return ['csv']
    
    def extract_content_with_structure(self, file_bytes: bytes) -> Dict[str, Any]:
        """
        Extract CSV content with structural information.
        
        Args:
            file_bytes: Raw CSV file content as bytes
            
        Returns:
            Dictionary containing CSV structure, headers, rows, and dialect info
        """
        try:
            # Decode bytes to text
            csv_text = file_bytes.decode('utf-8')
            
            # Detect CSV dialect
            try:
                dialect = csv.Sniffer().sniff(csv_text, delimiters=',;\t|')
            except csv.Error:
                # Fallback to default dialect
                dialect = csv.excel()
            
            # Parse CSV content
            reader = csv.reader(io.StringIO(csv_text), dialect=dialect)
            rows = list(reader)
            
            if not rows:
                return {
                    'file_type': 'csv',
                    'dialect': self._dialect_to_dict(dialect),
                    'headers': [],
                    'data_rows': [],
                    'total_rows': 0,
                    'total_columns': 0
                }
            
            # Extract headers (first row)
            headers = []
            for col_idx, header_value in enumerate(rows[0]):
                headers.append({
                    'column': col_idx,
                    'value': header_value,
                    'formatted_value': header_value
                })
            
            # Extract data rows (skip header)
            data_rows = []
            for row_idx, row_data in enumerate(rows[1:], 1):
                row_info = {
                    'row': row_idx,
                    'values': row_data,
                    'formatted_values': row_data.copy(),
                    'row_id': f"row_{row_idx}"
                }
                data_rows.append(row_info)
            
            structure = {
                'file_type': 'csv',
                'dialect': self._dialect_to_dict(dialect),
                'headers': headers,
                'data_rows': data_rows,
                'total_rows': len(rows),
                'total_columns': len(headers),
                'raw_rows': rows  # Keep original for reference
            }
            
            self.logger.info(f"[SUCCESS] Extracted CSV structure: {len(rows)} rows, {len(headers)} columns")
            return structure
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to extract CSV structure: {str(e)}")
            return {'file_type': 'csv', 'error': str(e)}
    
    def _dialect_to_dict(self, dialect) -> Dict[str, Any]:
        """Convert CSV dialect to dictionary for serialization."""
        return {
            'delimiter': dialect.delimiter,
            'quotechar': dialect.quotechar,
            'escapechar': dialect.escapechar,
            'doublequote': dialect.doublequote,
            'skipinitialspace': dialect.skipinitialspace,
            'lineterminator': dialect.lineterminator,
            'quoting': dialect.quoting
        }
    
    def _dict_to_dialect(self, dialect_dict: Dict[str, Any]) -> csv.Dialect:
        """Convert dictionary back to CSV dialect."""
        class CustomDialect(csv.Dialect):
            delimiter = dialect_dict.get('delimiter', ',')
            quotechar = dialect_dict.get('quotechar', '"')
            escapechar = dialect_dict.get('escapechar', None)
            doublequote = dialect_dict.get('doublequote', True)
            skipinitialspace = dialect_dict.get('skipinitialspace', False)
            lineterminator = dialect_dict.get('lineterminator', '\r\n')
            quoting = dialect_dict.get('quoting', csv.QUOTE_MINIMAL)
        
        return CustomDialect()
    
    def apply_section_updates(self, file_bytes: bytes, updates: List[UpdateSection]) -> bytes:
        """
        Apply targeted updates to CSV file while preserving dialect.
        
        Args:
            file_bytes: Original CSV file content as bytes
            updates: List of UpdateSection objects describing what to update
            
        Returns:
            Updated CSV file content as bytes
        """
        try:
            # Decode bytes to text
            csv_text = file_bytes.decode('utf-8')
            
            # Detect original dialect
            try:
                dialect = csv.Sniffer().sniff(csv_text, delimiters=',;\t|')
            except csv.Error:
                dialect = csv.excel()
            
            # Parse CSV content
            reader = csv.reader(io.StringIO(csv_text), dialect=dialect)
            rows = list(reader)
            
            # Apply updates
            for update in updates:
                self._apply_single_update(rows, update)
            
            # Write back with same dialect
            output = io.StringIO()
            writer = csv.writer(output, dialect=dialect)
            writer.writerows(rows)
            
            result_text = output.getvalue()
            result_bytes = result_text.encode('utf-8')
            
            self.logger.info(f"[SUCCESS] Applied {len(updates)} CSV updates: {len(result_bytes)} bytes")
            return result_bytes
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to apply CSV updates: {str(e)}")
            return file_bytes  # Return original if update fails
    
    def _apply_single_update(self, rows: List[List[str]], update: UpdateSection):
        """Apply a single update to the CSV rows."""
        try:
            if update.section_type == 'row':
                self._update_row(rows, update)
            elif update.section_type == 'cell':
                self._update_cell(rows, update)
            elif update.section_type == 'header':
                self._update_header(rows, update)
            else:
                self.logger.warning(f"[WARNING] Unknown CSV update type: {update.section_type}")
                
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to apply CSV update {update.section_id}: {str(e)}")
    
    def _update_row(self, rows: List[List[str]], update: UpdateSection):
        """Update a specific row."""
        try:
            target = update.target_location
            row_index = target.get('row')
            
            if row_index is None or row_index >= len(rows):
                self.logger.error(f"[ERROR] Invalid row index: {row_index}")
                return
            
            if update.action == 'replace_row' and update.new_values:
                # Ensure row has enough columns
                while len(rows[row_index]) < len(update.new_values):
                    rows[row_index].append('')
                
                # Update row values
                for col_idx, new_value in enumerate(update.new_values):
                    if col_idx < len(rows[row_index]):
                        rows[row_index][col_idx] = str(new_value)
            
            elif update.action == 'insert_row' and update.new_values:
                # Insert new row
                rows.insert(row_index, [str(val) for val in update.new_values])
            
            elif update.action == 'delete_row':
                # Delete row
                if row_index < len(rows):
                    rows.pop(row_index)
            
            self.logger.info(f"[SUCCESS] Updated CSV row {row_index}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to update CSV row: {str(e)}")
    
    def _update_cell(self, rows: List[List[str]], update: UpdateSection):
        """Update a specific cell."""
        try:
            target = update.target_location
            row_index = target.get('row')
            col_index = target.get('column')
            
            if row_index is None or col_index is None:
                self.logger.error(f"[ERROR] Invalid cell target: {target}")
                return
            
            if row_index >= len(rows):
                self.logger.error(f"[ERROR] Row index out of range: {row_index}")
                return
            
            # Ensure row has enough columns
            while len(rows[row_index]) <= col_index:
                rows[row_index].append('')
            
            # Update cell value
            if update.action == 'replace':
                rows[row_index][col_index] = str(update.new_content)
            
            self.logger.info(f"[SUCCESS] Updated CSV cell [{row_index}][{col_index}]")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to update CSV cell: {str(e)}")
    
    def _update_header(self, rows: List[List[str]], update: UpdateSection):
        """Update header row."""
        try:
            if not rows:
                self.logger.error(f"[ERROR] Cannot update header: no rows")
                return
            
            target = update.target_location
            col_index = target.get('column')
            
            if col_index is None:
                self.logger.error(f"[ERROR] Invalid header target: {target}")
                return
            
            # Ensure header row has enough columns
            while len(rows[0]) <= col_index:
                rows[0].append('')
            
            # Update header value
            if update.action == 'replace':
                rows[0][col_index] = str(update.new_content)
            
            self.logger.info(f"[SUCCESS] Updated CSV header column {col_index}")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to update CSV header: {str(e)}")
    
    def create_row_update(self, row_index: int, new_values: List[str], 
                         reason: str = None) -> UpdateSection:
        """Create a row update section."""
        return UpdateSection(
            section_id=f"csv_row_{row_index}",
            section_type="row",
            action="replace_row",
            new_values=new_values,
            target_location={'row': row_index},
            reason=reason
        )
    
    def create_cell_update(self, row_index: int, col_index: int, 
                          new_value: str, reason: str = None) -> UpdateSection:
        """Create a cell update section."""
        return UpdateSection(
            section_id=f"csv_cell_{row_index}_{col_index}",
            section_type="cell",
            action="replace",
            new_content=new_value,
            target_location={'row': row_index, 'column': col_index},
            reason=reason
        )
    
    def create_header_update(self, col_index: int, new_value: str, 
                           reason: str = None) -> UpdateSection:
        """Create a header update section."""
        return UpdateSection(
            section_id=f"csv_header_{col_index}",
            section_type="header",
            action="replace",
            new_content=new_value,
            target_location={'column': col_index},
            reason=reason
        )

