#!/usr/bin/env python3
"""
Test script to check which headings actually have paragraphs and can be updated.
"""

import sys
from docx import Document

def check_headings_with_content(docx_path):
    """Check which headings have paragraphs under them."""
    try:
        doc = Document(docx_path)
        
        headings_with_content = []
        current_section = None
        
        for i, paragraph in enumerate(doc.paragraphs):
            text = paragraph.text.strip()
            
            if not text:
                continue
                
            # Check if this is a heading
            is_heading = False
            heading_level = 0
            
            if paragraph.style.name.startswith('Heading'):
                is_heading = True
                heading_level = int(paragraph.style.name.split()[-1]) if paragraph.style.name.split()[-1].isdigit() else 1
            elif paragraph.runs and any(run.bold for run in paragraph.runs):
                if len(text) < 100 and not text.endswith('.'):
                    is_heading = True
                    heading_level = 1
            
            if is_heading:
                # Save previous section if it exists
                if current_section and current_section['paragraphs']:
                    headings_with_content.append(current_section)
                
                # Start new section
                current_section = {
                    'heading': text,
                    'level': heading_level,
                    'paragraph_index': i,
                    'paragraphs': []
                }
            else:
                # Add paragraph to current section
                if current_section:
                    current_section['paragraphs'].append({
                        'text': text,
                        'index': i
                    })
        
        # Don't forget the last section
        if current_section and current_section['paragraphs']:
            headings_with_content.append(current_section)
        
        print(f"Found {len(headings_with_content)} headings with content:")
        print("=" * 80)
        
        for section in headings_with_content:
            print(f"\n[{section['level']}] {section['heading']}")
            print(f"   Paragraphs: {len(section['paragraphs'])}")
            print(f"   Start index: {section['paragraph_index']}")
            
            # Show first few paragraphs
            for j, para in enumerate(section['paragraphs'][:2]):
                preview = para['text'][:100] + "..." if len(para['text']) > 100 else para['text']
                print(f"   {j+1}. {preview}")
            if len(section['paragraphs']) > 2:
                print(f"   ... and {len(section['paragraphs']) - 2} more paragraphs")
        
        return headings_with_content
        
    except Exception as e:
        print(f"Error checking document: {e}")
        return None

def main():
    """Main function to check headings with content."""
    
    docx_path = "test_step_by_step/input/Click_Professional_Documentation.docx"
    
    print("Checking Headings with Content")
    print("=" * 80)
    
    headings = check_headings_with_content(docx_path)
    
    if headings:
        print(f"\nTotal headings with content: {len(headings)}")
        print("\nThese are the headings that can be updated with new content.")
    else:
        print("No headings with content found")

if __name__ == "__main__":
    main()
