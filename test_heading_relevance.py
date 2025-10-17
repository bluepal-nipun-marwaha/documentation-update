#!/usr/bin/env python3
"""
Test script to analyze which headings are relevant for commit content.
This will help us determine where to place new content based on commit changes.
"""

import sys
from docx import Document
import json

def analyze_heading_relevance_for_commit(docx_path, commit_message, files_changed, diff_summary=""):
    """
    Analyze which headings in a DOCX file are relevant for a specific commit.
    
    Args:
        docx_path: Path to the DOCX file
        commit_message: Commit message
        files_changed: List of changed files
        diff_summary: Optional LLM-generated diff summary
        
    Returns:
        Dictionary with relevance analysis
    """
    try:
        doc = Document(docx_path)
        
        # Parse document structure
        headings = []
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
                # Save previous section
                if current_section:
                    headings.append(current_section)
                
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
        if current_section:
            headings.append(current_section)
        
        # Analyze relevance for each heading
        relevance_analysis = []
        
        print(f"Analyzing relevance for commit: '{commit_message}'")
        print(f"Files changed: {', '.join(files_changed)}")
        if diff_summary:
            print(f"Diff summary: {diff_summary[:200]}...")
        print("=" * 80)
        
        for section in headings:
            relevance_score = 0
            reasons = []
            
            heading_text = section['heading'].lower()
            section_content = ' '.join([p['text'] for p in section['paragraphs']]).lower()
            
            # Analyze based on commit message
            commit_lower = commit_message.lower()
            
            # Check for feature-related keywords
            if any(word in commit_lower for word in ['feat', 'feature', 'add', 'new']):
                if any(word in heading_text for word in ['feature', 'example', 'usage', 'how to']):
                    relevance_score += 3
                    reasons.append("Commit adds new features - matches feature/example sections")
            
            # Check for interactive/builder keywords
            if any(word in commit_lower for word in ['interactive', 'builder', 'demo']):
                if any(word in heading_text for word in ['example', 'demo', 'usage', 'tutorial']):
                    relevance_score += 4
                    reasons.append("Commit mentions interactive/builder - matches example sections")
            
            # Check file patterns
            if any('example' in f.lower() for f in files_changed):
                if any(word in heading_text for word in ['example', 'demo', 'usage']):
                    relevance_score += 3
                    reasons.append("Files changed include examples - matches example sections")
            
            if any('interactive' in f.lower() for f in files_changed):
                if any(word in heading_text for word in ['feature', 'example', 'usage']):
                    relevance_score += 3
                    reasons.append("Files changed include interactive components - matches feature sections")
            
            # Check for architecture changes
            if any(word in commit_lower for word in ['arch', 'structure', 'core', 'class']):
                if any(word in heading_text for word in ['arch', 'structure', 'core', 'class']):
                    relevance_score += 2
                    reasons.append("Commit mentions architecture - matches architecture sections")
            
            # Check for documentation changes
            if any('readme' in f.lower() for f in files_changed):
                if any(word in heading_text for word in ['example', 'usage', 'tutorial']):
                    relevance_score += 2
                    reasons.append("README files changed - likely documentation updates")
            
            # Determine if this section should be updated
            should_update = relevance_score >= 3
            update_type = "none"
            
            if should_update:
                if any(word in heading_text for word in ['example', 'demo']):
                    update_type = "add_example"
                elif any(word in heading_text for word in ['feature']):
                    update_type = "add_feature"
                elif any(word in heading_text for word in ['usage', 'how to']):
                    update_type = "add_usage"
                else:
                    update_type = "add_content"
            
            analysis = {
                'heading': section['heading'],
                'level': section['level'],
                'relevance_score': relevance_score,
                'should_update': should_update,
                'update_type': update_type,
                'reasons': reasons,
                'paragraph_count': len(section['paragraphs']),
                'paragraph_index': section['paragraph_index']
            }
            
            relevance_analysis.append(analysis)
            
            # Display analysis
            status = "UPDATE" if should_update else "SKIP"
            print(f"\n{status} [{section['level']}] {section['heading']}")
            print(f"   Relevance Score: {relevance_score}/10")
            print(f"   Update Type: {update_type}")
            print(f"   Paragraphs: {len(section['paragraphs'])}")
            if reasons:
                print(f"   Reasons: {'; '.join(reasons)}")
            else:
                print(f"   Reasons: No specific relevance found")
        
        return {
            'commit_message': commit_message,
            'files_changed': files_changed,
            'diff_summary': diff_summary,
            'analysis': relevance_analysis,
            'total_headings': len(headings),
            'headings_to_update': len([a for a in relevance_analysis if a['should_update']])
        }
        
    except Exception as e:
        print(f"Error analyzing document: {e}")
        return None

def display_analysis_summary(result):
    """Display a summary of the relevance analysis."""
    if not result:
        return
        
    print("\n" + "=" * 80)
    print("RELEVANCE ANALYSIS SUMMARY")
    print("=" * 80)
    
    print(f"Commit: {result['commit_message']}")
    print(f"Files changed: {len(result['files_changed'])}")
    print(f"Total headings analyzed: {result['total_headings']}")
    print(f"Headings to update: {result['headings_to_update']}")
    
    print("\nRECOMMENDED UPDATES:")
    for analysis in result['analysis']:
        if analysis['should_update']:
            print(f"  - [{analysis['level']}] {analysis['heading']}")
            print(f"    Type: {analysis['update_type']}")
            print(f"    Score: {analysis['relevance_score']}/10")
            print(f"    Reason: {'; '.join(analysis['reasons'])}")
    
    print("\nNEXT STEPS:")
    print("1. Review the recommended updates above")
    print("2. For each recommended heading, duplicate a paragraph")
    print("3. Edit the duplicated paragraph with relevant commit content")
    print("4. Test the updated document")

def main():
    """Main function to test the heading relevance analyzer."""
    
    # Test with our specific commit
    docx_path = "test_step_by_step/input/Click_Professional_Documentation.docx"
    commit_message = "feat: meaningful interactive builder"
    files_changed = [
        "README.md", 
        "pyproject.toml", 
        "src/click/__init__.py",
        "examples/interactive_builder/README.md",
        "examples/interactive_builder/interactive_demo.py",
        "src/click/interactive_builder.py"
    ]
    diff_summary = "Added new interactive builder feature with demo examples and core implementation"
    
    print("Testing Heading Relevance Analyzer")
    print("=" * 80)
    
    # Analyze relevance
    result = analyze_heading_relevance_for_commit(
        docx_path, 
        commit_message, 
        files_changed, 
        diff_summary
    )
    
    if result:
        # Display summary
        display_analysis_summary(result)
        
        print("\nAnalysis completed successfully!")
        
    else:
        print("Failed to analyze document")

if __name__ == "__main__":
    main()
