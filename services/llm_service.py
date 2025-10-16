"""
LLM Service for intelligent commit analysis and documentation updates.
Now supports multiple providers: OpenAI, Anthropic, Mistral, Gemma, Cohere, Ollama.
"""
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
import structlog
from utils.config import get_settings
from services.providers import (
    OpenAIProvider, AnthropicProvider, MistralProvider, 
    GemmaProvider, CohereProvider, OllamaProvider
)

logger = structlog.get_logger(__name__)

class LLMService:
    """Service for LLM-powered commit analysis and documentation updates with multiple providers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize LLM service with provider configuration."""
        if config is None:
            settings = get_settings()
            config = settings.ai.dict()
        
        self.config = config
        self.provider = self._create_provider(config)
        self._initialize_provider()
    
    def _create_provider(self, config: Dict[str, Any]):
        """Create the appropriate LLM provider based on configuration."""
        provider_name = config.get('llm_provider', 'ollama').lower()
        
        if provider_name == 'openai':
            return OpenAIProvider(config)
        elif provider_name == 'anthropic':
            return AnthropicProvider(config)
        elif provider_name == 'mistral':
            return MistralProvider(config)
        elif provider_name == 'gemma':
            return GemmaProvider(config)
        elif provider_name == 'cohere':
            return CohereProvider(config)
        elif provider_name == 'ollama':
            return OllamaProvider(config)
        else:
            logger.warning(f"Unknown provider {provider_name}, falling back to Ollama")
            return OllamaProvider(config)
    
    def _initialize_provider(self):
        """Initialize the selected provider."""
        try:
            if self.provider.is_available():
                model_info = self.provider.get_model_info()
                logger.info(f"[SUCCESS] LLM provider initialized: {model_info}")
            else:
                raise Exception(f"LLM provider {self.config.get('llm_provider')} is not available")
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize LLM provider: {str(e)}")
            raise
    
    def _call_llm(self, prompt: str, system_prompt: str = "", temperature: float = 0.3, max_tokens: int = 1000) -> str:
        """Call the configured LLM provider with the given prompt."""
        try:
            return self.provider.generate_response(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
        except Exception as e:
            logger.error(f"[ERROR] LLM provider call failed: {str(e)}")
            raise Exception(f"LLM provider failed: {str(e)}")
    
    def generate_response(self, prompt: str, system_prompt: str = "", temperature: float = 0.3, max_tokens: int = 1000) -> str:
        """
        Public method to generate responses from the LLM.
        This method is used by the DOCX workflow modules.
        
        Args:
            prompt: The input prompt for the LLM
            system_prompt: Optional system prompt
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
            
        Returns:
            Generated response from the LLM
        """
        return self._call_llm(prompt, system_prompt, temperature, max_tokens)
    
    def analyze_commit_impact(self, commit_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze commit impact and determine documentation needs.
        
        Args:
            commit_data: Dictionary containing commit information
            
        Returns:
            Dictionary with analysis results and recommendations
        """
        try:
            logger.info("🤖 Analyzing commit impact with LLM...")
            
            # Extract key information
            commit_message = commit_data.get('message', '')
            author = commit_data.get('author', '')
            modified_files = commit_data.get('modified', [])
            added_files = commit_data.get('added', [])
            removed_files = commit_data.get('removed', [])
            diff_content = commit_data.get('diff', '')
            
            # Create analysis prompt
            prompt = self._create_commit_analysis_prompt(
                commit_message, author, modified_files, added_files, 
                removed_files, diff_content
            )
            
            # Call LLM for analysis
            system_prompt = "You are an expert software documentation analyst. Analyze commits and determine their impact on documentation."
            llm_response = self._call_llm(prompt, system_prompt, temperature=0.3, max_tokens=1000)
            
            # Parse response
            analysis_result = self._parse_commit_analysis(llm_response)
            
            logger.info(f"[SUCCESS] Commit analysis completed: {analysis_result.get('impact_level', 'unknown')} impact")
            return analysis_result
            
        except Exception as e:
            logger.error(f"[ERROR] Commit analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "impact_level": "unknown",
                "recommended_docs": [],
                "reasoning": "Analysis failed due to error"
            }
    
    def _create_commit_analysis_prompt(self, commit_message: str, author: str, 
                                     modified_files: List[str], added_files: List[str], 
                                     removed_files: List[str], diff_content: str) -> str:
        """Create prompt for commit analysis."""
        
        prompt = f"""
Analyze this commit and determine its impact on documentation:

            COMMIT DETAILS:
            - Message: {commit_message}
            - Author: {author}
            - Modified files: {', '.join(modified_files) if modified_files else 'None'}
            - Added files: {', '.join(added_files) if added_files else 'None'}
            - Removed files: {', '.join(removed_files) if removed_files else 'None'}

            CODE CHANGES:
            {diff_content[:2000] if diff_content else 'No diff available'}

            ANALYSIS REQUIRED:
            1. Determine the impact level: low, medium, or high
            2. Identify which types of documentation might need updates
            3. Provide reasoning for your assessment

RESPOND IN JSON FORMAT:
{{
    "impact_level": "low|medium|high",
    "reasoning": "Brief explanation of why this commit affects documentation",
    "recommended_docs": [
        {{
            "doc_type": "api_reference|user_guide|installation|changelog|readme",
            "priority": "high|medium|low",
            "reason": "Why this documentation needs updating"
        }}
    ],
    "key_changes": [
        "List of key functional changes that affect users"
    ]
}}
"""
        return prompt
    
    def _parse_commit_analysis(self, llm_response: str) -> Dict[str, Any]:
        """Parse LLM response for commit analysis."""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
            if json_match:
                analysis_data = json.loads(json_match.group())
                analysis_data["success"] = True
                return analysis_data
            else:
                # Fallback if no JSON found
                return {
                    "success": False,
                    "impact_level": "medium",
                    "reasoning": "Could not parse LLM response",
                    "recommended_docs": [],
                    "key_changes": []
                }
        except json.JSONDecodeError as e:
            logger.error(f"[ERROR] Failed to parse LLM response: {str(e)}")
            return {
                "success": False,
                "impact_level": "medium",
                "reasoning": "Invalid JSON response from LLM",
                "recommended_docs": [],
                "key_changes": []
            }
    
    def select_documentation_files(self, analysis_result: Dict[str, Any], 
                                 available_docs: List[str]) -> List[str]:
        """
        Select which documentation files to update based on analysis.
        
        Args:
            analysis_result: Result from commit analysis
            available_docs: List of available documentation files
            
        Returns:
            List of documentation files to update
        """
        try:
            logger.info("🤖 Selecting documentation files with LLM...")
            
            # Create selection prompt
            prompt = self._create_doc_selection_prompt(analysis_result, available_docs)
            
            # Call LLM for selection
            system_prompt = "You are an expert at selecting relevant documentation files based on code changes."
            llm_response = self._call_llm(prompt, system_prompt, temperature=0.2, max_tokens=500)
            
            # Debug logging
            logger.info(f"🔍 LLM Document Selection Response: {llm_response}")
            
            # Parse response
            selected_files = self._parse_doc_selection(llm_response, available_docs)
            
            logger.info(f"[SUCCESS] Selected {len(selected_files)} documentation files for update")
            return selected_files
            
        except Exception as e:
            logger.error(f"[ERROR] Documentation selection failed: {str(e)}")
            # Fallback to basic selection
            return self._fallback_doc_selection(analysis_result, available_docs)
    
    def _create_doc_selection_prompt(self, analysis_result: Dict[str, Any], 
                                   available_docs: List[str]) -> str:
        """Create prompt for documentation file selection."""
        
        recommended_docs = analysis_result.get('recommended_docs', [])
        key_changes = analysis_result.get('key_changes', [])
        
        prompt = f"""
Based on this commit analysis, select which documentation files should be updated:

COMMIT ANALYSIS:
- Impact Level: {analysis_result.get('impact_level', 'unknown')}
- Reasoning: {analysis_result.get('reasoning', 'No reasoning provided')}
- Key Changes: {', '.join(key_changes) if key_changes else 'None identified'}

RECOMMENDED DOCUMENTATION TYPES:
{json.dumps(recommended_docs, indent=2) if recommended_docs else 'None'}

AVAILABLE DOCUMENTATION FILES (YOU MUST SELECT FROM THESE ONLY):
{json.dumps(available_docs, indent=2)}

CRITICAL INSTRUCTIONS:
1. You MUST ONLY select files from the "AVAILABLE DOCUMENTATION FILES" list above
2. DO NOT suggest files that are not in the available list
3. If there are available documentation files and the commit has any functional changes, select at least one file
4. Even minor changes can benefit from documentation updates
5. If only one documentation file exists, strongly consider updating it

SELECTION CRITERIA:
- Be PROACTIVE - if there are any documentation files available, consider updating them
- Prioritize comprehensive documentation files (like main guides or references)
- Documentation should be kept current and comprehensive

RESPOND WITH A JSON ARRAY containing ONLY file paths from the available list:
["file1.md", "file2.md", ...]

REMEMBER: Only select files that exist in the AVAILABLE DOCUMENTATION FILES list above!
"""
        return prompt
    
    def _parse_doc_selection(self, llm_response: str, available_docs: List[str]) -> List[str]:
        """Parse LLM response for documentation file selection."""
        try:
            import re
            json_match = re.search(r'\[.*\]', llm_response, re.DOTALL)
            if json_match:
                selected_files = json.loads(json_match.group())
                # Validate that selected files exist in available docs
                valid_files = [f for f in selected_files if f in available_docs]
                return valid_files
            else:
                return self._fallback_doc_selection({}, available_docs)
        except json.JSONDecodeError:
            return self._fallback_doc_selection({}, available_docs)
    
    def _fallback_doc_selection(self, analysis_result: Dict[str, Any], 
                               available_docs: List[str]) -> List[str]:
        """Fallback documentation selection when LLM fails."""
        logger.info("🔄 Using fallback documentation selection")
        
        # More inclusive fallback logic
        if not available_docs:
            return []
        
        # If there are available docs and the commit has any impact, select files
        impact_level = analysis_result.get('impact_level', 'unknown')
        if impact_level in ['medium', 'high'] or len(available_docs) == 1:
            # For medium/high impact or if only one doc exists, select all available docs
            return available_docs[:2]  # Limit to 2 files max
        
        # For low impact, still be selective but include comprehensive docs
        fallback_files = []
        for doc in available_docs:
            if any(keyword in doc.lower() for keyword in ['api', 'reference', 'guide', 'documentation', 'docx']):
                fallback_files.append(doc)
        
        # If no specific keywords found, still select the first available doc
        if not fallback_files and available_docs:
            fallback_files.append(available_docs[0])
        
        return fallback_files[:2]  # Limit to 2 files
    
    def generate_documentation_update(self, doc_content: str, commit_context: Dict[str, Any], 
                                    rag_context: str = "") -> str:
        """
        Generate updated documentation content using LLM with enhanced Markdown support.
        
        Args:
            doc_content: Current documentation content (may be Markdown from DOCX conversion)
            commit_context: Commit information and analysis
            rag_context: Additional context from RAG retrieval
            
        Returns:
            Updated documentation content (Markdown format for DOCX files)
        """
        try:
            logger.info("🤖 Generating documentation update with enhanced LLM processing...")
            
            # Detect if content is Markdown (from DOCX conversion)
            is_markdown_content = self._is_markdown_content(doc_content)
            
            if is_markdown_content:
                logger.info("📝 Processing Markdown content for DOCX update")
                prompt = self._create_markdown_update_prompt(doc_content, commit_context, rag_context)
                system_prompt = """You are an expert technical writer specializing in Markdown documentation. 

CRITICAL INSTRUCTIONS:
1. You will receive documentation in Markdown format
2. Update it based on the code changes provided
3. Return ONLY the complete updated Markdown document
4. Preserve all existing structure (headings, lists, tables, formatting)
5. Use proper Markdown syntax for any new content
6. Maintain document hierarchy and organization
7. Update version numbers, feature lists, and examples as needed
8. Do not add explanatory text - return only the updated document"""
            else:
                # Regular text content
                prompt = self._create_update_prompt(doc_content, commit_context, rag_context)
                system_prompt = "You are an expert technical writer. Update documentation based on code changes while maintaining clarity and accuracy. Always preserve the complete structure and content of the original document."
            
            # Call LLM for update
            updated_content = self._call_llm(prompt, system_prompt, temperature=0.3, max_tokens=4000)
            
            logger.info("[SUCCESS] Enhanced documentation update generated successfully")
            return updated_content
            
        except Exception as e:
            logger.error(f"[ERROR] Documentation update generation failed: {str(e)}")
            return doc_content  # Return original content if update fails
    
    def _is_markdown_content(self, content: str) -> bool:
        """Check if content appears to be Markdown format."""
        if not content or len(content.strip()) < 10:
            return False
            
        markdown_indicators = [
            content.count('#') > 0,  # Headers
            content.count('**') >= 2,  # Bold text (pairs)
            content.count('*') > 3,   # Italic or lists (more strict)
            content.count('|') > 3,   # Tables (more strict)
            content.count('```') > 0, # Code blocks
            content.count('- ') > 0,  # Lists
            content.count('\n## ') > 0 or content.count('\n# ') > 0,  # Clear headers
        ]
        
        # If at least 2 markdown indicators are present, treat as markdown
        return sum(markdown_indicators) >= 2
    
    def _create_markdown_update_prompt(self, doc_content: str, commit_context: Dict[str, Any], 
                                     rag_context: str) -> str:
        """Create an enhanced prompt for Markdown-based documentation updates."""
        
        commit_message = commit_context.get('message', 'No commit message')
        modified_files = commit_context.get('modified', [])
        analysis = commit_context.get('analysis', {})
        diff_info = commit_context.get('diff', '')
        
        prompt = f"""You are updating technical documentation based on recent code changes. This document will be converted from Markdown back to DOCX format, so proper formatting is critical.

CURRENT DOCUMENTATION (Markdown format):
```markdown
{doc_content[:3000]}{'...' if len(doc_content) > 3000 else ''}
```

CODE CHANGES INFORMATION:
- **Commit Message**: {commit_message}
- **Modified Files**: {', '.join(modified_files[:10])}
- **Change Analysis**: {analysis.get('summary', 'No analysis available')}

ADDITIONAL CONTEXT:
{rag_context[:1000] if rag_context else 'No additional context available'}

CODE DIFF (if available):
```
{diff_info[:1500] if diff_info else 'No diff available'}
```

TASK:
Update the documentation to reflect these code changes while maintaining perfect DOCX-compatible formatting.

CRITICAL FORMATTING REQUIREMENTS (for DOCX conversion):
1. **Tables**: Use proper Markdown table syntax with | separators and alignment
2. **Headers**: Use # ## ### hierarchy consistently 
3. **Lists**: Use proper - or * for bullets, 1. 2. 3. for numbered lists
4. **Code blocks**: Use ``` with language specification
5. **Structure**: Maintain exact document organization and section flow
6. **Spacing**: Preserve proper line breaks and paragraph spacing

CONTENT UPDATES:
1. **Accuracy**: Ensure all information matches the current codebase
2. **Completeness**: Update all relevant sections (features, API, examples, etc.)
3. **Version Info**: Update version numbers if applicable (e.g., 8.3.dev → 8.4.dev)
4. **Feature Lists**: Add new features to appropriate tables and lists
5. **Examples**: Update code examples to reflect changes

FORMATTING STANDARDS:
- Tables must have proper | Column 1 | Column 2 | format
- Headers must follow # ## ### hierarchy
- Lists must be properly formatted with consistent indentation
- Code blocks must use ``` syntax
- Preserve all existing technical details and metrics

IMPORTANT:
- Return the COMPLETE updated document in Markdown format
- Preserve ALL existing content that's still relevant
- Maintain the professional document structure and flow
- Do NOT add meta-commentary or explanations about your changes
- Ensure tables and lists format correctly for DOCX conversion

Updated Documentation:"""
        
        return prompt
    
    def _create_update_prompt(self, doc_content: str, commit_context: Dict[str, Any], 
                            rag_context: str) -> str:
        """Create prompt for documentation update."""
        
        commit_message = commit_context.get('message', '')
        diff_content = commit_context.get('diff', '')
        analysis = commit_context.get('analysis', {})
        
        prompt = f"""
Update this documentation based on the code changes:

CURRENT DOCUMENTATION (COMPLETE):
{doc_content}

COMMIT INFORMATION:
- Message: {commit_message}
- Analysis: {json.dumps(analysis, indent=2)}

CODE CHANGES:
{diff_content[:1000] if diff_content else 'No diff available'}

ADDITIONAL CONTEXT:
{rag_context[:500] if rag_context else 'No additional context'}

UPDATE INSTRUCTIONS:
1. Keep the existing structure and formatting COMPLETELY
2. Update only the sections that are affected by the code changes
3. Maintain the same tone and style
4. Be precise and accurate
5. PRESERVE ALL original content - do not truncate or remove sections
6. If no changes are needed, return the original content unchanged
7. Return the COMPLETE updated documentation with all sections intact
        """
        
        return prompt
    
    def determine_docx_edit_type(self, commit_context: Dict[str, Any], doc_content: str) -> str:
        """
        Determines what type of DOCX edit is needed based on commit context and document content.
        
        Args:
            commit_context: Commit information and context
            doc_content: Current document content
            
        Returns:
            Edit type: "edit_line", "add_paragraph", "edit_table", or "no_change"
        """
        try:
            commit_message = commit_context.get('message', '')
            files_changed = commit_context.get('files', [])
            diff_content = commit_context.get('diff', '')
            
            # Create prompt to analyze what type of edit is needed
            prompt = f"""
            Analyze this commit and document to determine what type of DOCX edit is needed:
            
            COMMIT INFORMATION:
            - Message: {commit_message}
            - Files Changed: {', '.join(files_changed)}
            - Diff: {diff_content[:500] if diff_content else 'No diff available'}
            
            DOCUMENT CONTENT (first 1000 characters):
            {doc_content[:1000]}
            
            Based on this analysis, determine what type of edit is needed:
            
            1. "edit_line" - If existing text/paragraphs need to be updated or modified
            2. "add_paragraph" - If new content needs to be added to the document
            3. "edit_table" - If tables in the document need to be updated
            4. "no_change" - If no changes are needed to the document
            
            Consider:
            - Does the commit add new features that need documentation?
            - Does it modify existing functionality that needs updating?
            - Does it change data structures that affect tables?
            - Does it require new sections or explanations?
            
            Return only one of: edit_line, add_paragraph, edit_table, no_change
            """
            
            response = self.provider.generate_response(prompt)
            
            # Parse response and validate
            edit_type = response.strip().lower()
            
            valid_types = ['edit_line', 'add_paragraph', 'edit_table', 'no_change']
            if edit_type in valid_types:
                logger.info(f"Determined DOCX edit type: {edit_type}")
                return edit_type
            else:
                logger.warning(f"Invalid edit type '{edit_type}', defaulting to 'edit_line'")
                return 'edit_line'
                
        except Exception as e:
            logger.error(f"Error determining DOCX edit type: {e}")
            return 'edit_line'  # Default fallback
