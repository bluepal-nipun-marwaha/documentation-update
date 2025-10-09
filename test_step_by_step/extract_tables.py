#!/usr/bin/env python3
"""
Table Extraction Script
Extracts tables from the original DOCX file for Step 3 processing.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any
from docx import Document

class TableExtractor:
    """Extracts tables from DOCX files."""
    
    def __init__(self, input_dir=None, output_dir=None):
        """Initialize the table extractor."""
        if input_dir:
            self.input_dir = Path(input_dir)
        else:
            self.input_dir = Path(__file__).parent / "input"
            
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            self.output_dir = Path(__file__).parent / "output"
        
        # Ensure directories exist
        self.input_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
    
    def run_extraction(self):
        """Run table extraction."""
        try:
            print("Table Extraction")
            print("=" * 50)
            print(f"Input Directory: {self.input_dir}")
            print(f"Output Directory: {self.output_dir}")
            print("Starting Table Extraction")
            print("=" * 50)
            
            # Load original document
            original_doc = self.load_original_documentation()
            if not original_doc:
                print("Failed to load original document")
                return False
            
            # Extract tables
            tables_data = self.extract_tables(original_doc)
            
            # Save extracted tables
            self.save_extracted_tables(tables_data)
            
            print("=" * 50)
            print("Table Extraction Completed Successfully!")
            print(f"Check the output directory: {self.output_dir}")
            print("=" * 50)
            
            return True
            
        except Exception as e:
            print(f"Table extraction failed: {str(e)}")
            return False
    
    def load_original_documentation(self) -> Document:
        """Load the original documentation."""
        try:
            print("Loading original documentation...")
            docx_files = list(self.input_dir.glob("*.docx"))
            if not docx_files:
                print("No DOCX files found in input directory")
                return None
            
            original_file = docx_files[0]  # Take the first DOCX file
            doc = Document(original_file)
            print(f"Original documentation loaded: {original_file.name}")
            return doc
            
        except Exception as e:
            print(f"Error loading original documentation: {str(e)}")
            return None
    
    def extract_tables(self, doc: Document) -> List[Dict[str, Any]]:
        """Extract all tables from the document."""
        try:
            print("Extracting tables...")
            
            tables_data = []
            
            for i, table in enumerate(doc.tables):
                print(f"  Processing table {i + 1}: {len(table.rows)} rows x {len(table.columns)} columns")
                
                table_data = {
                    "id": f"table_{i}",
                    "index": i,
                    "rows": len(table.rows),
                    "columns": len(table.columns),
                    "alignment": "left",
                    "columns_data": [],
                    "rows_data": []
                }
                
                # Extract data by rows
                for row_idx, row in enumerate(table.rows):
                    row_data = {
                        "index": row_idx,
                        "is_header": row_idx == 0,  # First row is header
                        "cells": []
                    }
                    
                    for col_idx, cell in enumerate(row.cells):
                        cell_text = cell.text.strip()
                        
                        # Add to columns_data
                        column_data = {
                            "index": col_idx,
                            "content": cell_text,
                            "is_header": row_idx == 0,
                            "formatting": {
                                "font": {
                                    "name": "Calibri",
                                    "size": 10.5,
                                    "bold": row_idx == 0,  # Headers are bold
                                    "italic": False,
                                    "color": "FFFFFF" if row_idx == 0 else "333333"
                                },
                                "alignment": {
                                    "horizontal": "center" if row_idx == 0 else "left",
                                    "vertical": "top"
                                },
                                "background": {
                                    "color": "34495E" if row_idx == 0 else "2D2D2D"
                                },
                                "borders": {}
                            }
                        }
                        table_data["columns_data"].append(column_data)
                        
                        # Add to row cells
                        cell_data = {
                            "index": col_idx,
                            "content": cell_text,
                            "is_header": row_idx == 0,
                            "formatting": column_data["formatting"]
                        }
                        row_data["cells"].append(cell_data)
                    
                    table_data["rows_data"].append(row_data)
                
                tables_data.append(table_data)
            
            print(f"Extracted {len(tables_data)} tables")
            return tables_data
            
        except Exception as e:
            print(f"Error extracting tables: {str(e)}")
            return []
    
    def save_extracted_tables(self, tables_data: List[Dict[str, Any]]):
        """Save extracted tables to JSON file."""
        try:
            print("Saving extracted tables...")
            
            output_path = self.output_dir / "extracted_tables.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(tables_data, f, indent=2, ensure_ascii=False)
            
            print(f"Extracted tables saved: {output_path}")
            
            # Also create a summary
            summary_path = self.output_dir / "table_summary.txt"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("Table Extraction Summary\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Total tables extracted: {len(tables_data)}\n\n")
                
                for i, table in enumerate(tables_data):
                    f.write(f"Table {i + 1}:\n")
                    f.write(f"  Dimensions: {table['rows']} rows x {table['columns']} columns\n")
                    f.write(f"  ID: {table['id']}\n")
                    
                    # Show first few cells
                    if table['columns_data']:
                        f.write("  Sample content:\n")
                        for j, col in enumerate(table['columns_data'][:6]):  # First 6 cells
                            f.write(f"    [{j}] {col['content'][:50]}...\n")
                    f.write("\n")
            
            print(f"Table summary saved: {summary_path}")
            
        except Exception as e:
            print(f"Error saving extracted tables: {str(e)}")

def main():
    """Main function to run table extraction."""
    extractor = TableExtractor()
    success = extractor.run_extraction()
    
    if success:
        print("Table extraction completed successfully!")
        print("Check the 'test_step_by_step/output' folder for results")
        print("\nGenerated files:")
        print("- extracted_tables.json: Extracted table data")
        print("- table_summary.txt: Human-readable summary")
    else:
        print("Table extraction failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
