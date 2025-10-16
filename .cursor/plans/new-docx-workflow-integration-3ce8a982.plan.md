<!-- 3ce8a982-ab65-4de7-adaa-102a2964348e 280c3cbb-56a0-4edb-9f0f-365281c16159 -->
# New DOCX Workflow Integration Plan

## Overview

Replace the complex 5-step DOCX workflow with three specialized, simpler workflows based on the edit operation type. Each workflow preserves formatting while making targeted updates.

## Analysis of Existing Workflows

### 1. **document_editor (fixed).py** - For Editing Existing Lines

- **Purpose**: Replace/rephrase text while preserving word-level formatting (bold, italic, underline, colors)
- **Key Functions**:
  - `capture_detailed_formatting()`: Captures font, size, bold, italic, color, highlight per run
  - `rephrase_text_with_formatting()`: Uses LLM with formatting-aware prompts and `<BOLD>`, `<ITALIC>`, `<UNDERLINE>` tags
  - `parse_formatted_response()`: Parses LLM tags into segments
  - `replace_paragraph_text()`: Safely clears runs (not removes) and applies new text with formatting
  - `apply_proportional_formatting_safe()`: Distributes formatting proportionally when no tags found
- **Safe for**: Paragraphs, list items, table cells, headers, footers

### 2. **double_para.py** - For Adding New Paragraphs

- **Purpose**: Duplicate paragraphs to create insertion points, then edit the duplicated content
- **Key Functions**:
  - Same formatting capture/apply as document_editor
  - `duplicate_text_with_formatting()`: Simple duplication with `\n` separator
  - Uses proportional formatting distribution (not LLM tags)
- **Strategy**: Duplicate nearest paragraph, then edit the duplicate to create new content

### 3. **table_editor.py** - For Editing Tables

- **Purpose**: Extract table data, recreate with professional formatting (GitHub docs style)
- **Key Functions**:
  - `extract_table_content()`: Extracts all cell text as list of lists
  - `get_most_used_font()`: Detects document's primary font
  - `replace_table_in_document()`: Replaces entire table XML element
  - **Formatting Applied**:
    - Font: Document's most-used font
    - Size: 10.5pt
    - Header row: Bold white text on dark blue-gray (#34495E)
    - Borders: Black, single line, 4pt
    - Padding: 12pt left, 6pt top/bottom/right
    - Row height: 30pt minimum
    - Vertical alignment: Center
    - Adds row of "1"s at bottom (per requirements)

## Implementation Strategy

### Module Structure

Create three new modules in `utils/`:

1. `utils/docx_line_editor.py` - Line/paragraph editing with formatting preservation
2. `utils/docx_paragraph_inserter.py` - Paragraph insertion via duplication
3. `utils/docx_table_editor.py` - Table extraction and reformatting

### Integration Points

#### In `existing_repo_workflow.py`:

**Current Flow** (lines 1905-1957 in `_update_documentation_with_llm()`):

```python
if file_path.endswith('.docx'):
    if enhanced_docx_workflow_enabled:
        # Current 5-step workflow
    else:
        # Basic fallback
```

**New Flow**:

```python
if file_path.endswith('.docx'):
    # LLM determines edit type: "edit_line", "add_paragraph", "edit_table"
    edit_type = llm_service.determine_docx_edit_type(commit_context, doc_content)
    
    if edit_type == "edit_line":
        # Use line editor workflow
        updated_content = docx_line_editor.edit_lines(original_docx, commit_context, llm_service)
    elif edit_type == "add_paragraph":
        # Use paragraph inserter workflow
        updated_content = docx_paragraph_inserter.insert_paragraph(original_docx, commit_context, llm_service)
    elif edit_type == "edit_table":
        # Use table editor workflow
        updated_content = docx_table_editor.edit_tables(original_docx, commit_context, llm_service)
    else:
        # Fallback to basic processing
```

## Implementation Steps

### Step 1: Create `utils/docx_line_editor.py`

- Extract and adapt functions from `document_editor (fixed).py`
- Key functions to implement:
  - `capture_detailed_formatting(paragraph)`
  - `rephrase_text_with_formatting(text, formats, llm_service, commit_context)`
  - `parse_formatted_response(llm_response, original_formats)`
  - `replace_paragraph_text(paragraph, new_text_or_segments, original_formats)`
  - `apply_proportional_formatting_safe(paragraph, new_text, original_formats)`
  - `edit_document_lines(docx_bytes, commit_context, llm_service)` - Main entry point
- Integrate with `LLMService` (not OpenAI directly)
- Return updated DOCX as bytes

### Step 2: Create `utils/docx_paragraph_inserter.py`

- Extract and adapt functions from `double_para.py`
- Key functions to implement:
  - Same formatting helpers as line_editor
  - `find_insertion_point(doc, commit_context, llm_service)` - LLM determines where to insert
  - `duplicate_paragraph(paragraph)` - Creates copy with formatting
  - `insert_paragraph(docx_bytes, commit_context, llm_service)` - Main entry point
- Strategy: Find insertion point → Duplicate nearby paragraph → Edit duplicate with new content
- Return updated DOCX as bytes

### Step 3: Create `utils/docx_table_editor.py`

- Extract and adapt functions from `table_editor.py`
- Key functions to implement:
  - `extract_table_content(table)` - Extract as list of lists
  - `get_most_used_font(doc)` - Detect document font
  - `replace_table_in_document(doc, old_table, new_data, font_name)` - Replace with formatting
  - `edit_tables(docx_bytes, commit_context, llm_service)` - Main entry point
- LLM updates table content (cells) based on commit
- Apply GitHub docs style formatting
- Return updated DOCX as bytes

### Step 4: Add LLM Edit Type Determination

In `services/llm_service.py`, add new method:

```python
def determine_docx_edit_type(self, commit_context, doc_content):
    """
    Determines what type of DOCX edit is needed.
    Returns: "edit_line", "add_paragraph", "edit_table"
    """
```

- Analyzes commit message, diff, and document structure
- Returns edit type classification

### Step 5: Update `existing_repo_workflow.py`

In `_update_documentation_with_llm()` method:

- Remove 5-step workflow code (lines 2768-2955)
- Add new workflow dispatcher:
  - Call `llm_service.determine_docx_edit_type()`
  - Route to appropriate module based on edit type
  - Handle errors with fallback
- Keep MD file handling unchanged (direct LLM pass-through)

### Step 6: Update Configuration

- Remove or deprecate `ENHANCED_DOCX_WORKFLOW` flag
- Add new flags if needed:
  - `DOCX_WORKFLOW_ENABLED=true` (enable new workflow)
  - `DOCX_TABLE_FORMATTING=github_docs` (formatting style)

### Step 7: Update Documentation

- Update `ENHANCED_DOCX_WORKFLOW_INTEGRATION.md` to describe new workflows
- Add workflow selection logic documentation
- Document when each workflow is used

## Key Implementation Details

### Formatting Preservation Strategy

All workflows use the **"Safe Clearing"** approach:

- Clear text from existing runs (set `run.text = ""`)
- DO NOT remove run XML elements
- Reuse cleared runs to preserve structure
- Only create new runs if needed
- Maintains compatibility with tables, lists, headers, footers

### LLM Integration

- Use existing `LLMService` (not direct OpenAI calls)
- Support all configured providers (Ollama, OpenAI, Anthropic, etc.)
- Formatting-aware prompts with `<BOLD>`, `<ITALIC>`, `<UNDERLINE>` tags
- Fallback to proportional formatting if tags not returned

### Error Handling

- Each module has try-except with logging
- Fallback to original content on failure
- Return original DOCX bytes if all workflows fail

## Testing Approach

1. Test each module independently with sample DOCX files
2. Test edit type determination accuracy
3. Test integration in main workflow
4. Test with different LLM providers
5. Test formatting preservation (bold, italic, colors, etc.)

## Markdown Files

**No changes needed** - Markdown files continue to:

- Pass through LLM directly
- Use existing `generate_documentation_update()` method
- Return updated text for direct replacement

## Benefits of New Approach

1. **Simpler**: 3 focused workflows vs 5 complex steps
2. **Faster**: No multi-step conversions (DOCX→MD→DOCX)
3. **Better Formatting**: Direct manipulation preserves all formatting
4. **More Accurate**: LLM determines appropriate edit type
5. **Maintainable**: Clear separation of concerns
6. **Professional Output**: GitHub docs style tables

### To-dos

- [ ] Create utils/docx_line_editor.py with formatting-preserving line editing workflow
- [ ] Create utils/docx_paragraph_inserter.py with paragraph duplication and insertion workflow
- [ ] Create utils/docx_table_editor.py with table extraction and GitHub docs formatting workflow
- [ ] Add determine_docx_edit_type() method to services/llm_service.py
- [ ] Update existing_repo_workflow.py to use new workflow dispatcher and remove 5-step workflow
- [ ] Update configuration flags and environment variables
- [ ] Update ENHANCED_DOCX_WORKFLOW_INTEGRATION.md with new workflow documentation