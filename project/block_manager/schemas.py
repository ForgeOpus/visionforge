"""
Pydantic schemas for input validation
"""
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional


class ChatRequest(BaseModel):
    """Schema for chat message requests"""
    message: str = Field(..., max_length=5000, description="User message")
    history: List[Dict[str, Any]] = Field(default_factory=list, description="Chat history")
    modificationMode: bool = Field(default=False, description="Whether AI can modify workflow")
    workflowState: Optional[Dict[str, Any]] = Field(None, description="Current workflow state")

    @validator('message')
    def validate_message(cls, v):
        """Ensure message is not empty"""
        if not v or not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()

    @validator('history')
    def validate_history(cls, v):
        """Limit history length to prevent excessive payload"""
        if len(v) > 100:
            raise ValueError('History too long (max 100 messages)')
        return v


class ExportRequest(BaseModel):
    """Schema for model export requests"""
    nodes: List[Dict[str, Any]] = Field(..., min_items=1, description="Model nodes")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="Model edges")
    format: str = Field(..., description="Export format (pytorch or tensorflow)")
    projectName: str = Field(..., min_length=1, max_length=100, description="Project name")
    groupDefinitions: List[Dict[str, Any]] = Field(default_factory=list, description="Group definitions")

    @validator('format')
    def validate_format(cls, v):
        """Ensure format is valid"""
        if v not in ['pytorch', 'tensorflow']:
            raise ValueError('Format must be "pytorch" or "tensorflow"')
        return v

    @validator('projectName')
    def validate_project_name(cls, v):
        """Sanitize project name"""
        if not v or not v.strip():
            raise ValueError('Project name cannot be empty')
        # Remove special characters that could cause file system issues
        sanitized = ''.join(c for c in v if c.isalnum() or c in (' ', '_', '-'))
        if not sanitized:
            raise ValueError('Project name must contain at least one alphanumeric character')
        return sanitized.strip()


class ValidateRequest(BaseModel):
    """Schema for model validation requests"""
    nodes: List[Dict[str, Any]] = Field(..., min_items=1, description="Model nodes")
    edges: List[Dict[str, Any]] = Field(default_factory=list, description="Model edges")


class NodeDefinitionRequest(BaseModel):
    """Schema for node definition requests"""
    framework: str = Field(default='pytorch', description="Framework (pytorch or tensorflow)")

    @validator('framework')
    def validate_framework(cls, v):
        """Ensure framework is valid"""
        if v not in ['pytorch', 'tensorflow']:
            raise ValueError('Framework must be "pytorch" or "tensorflow"')
        return v


class RenderNodeCodeRequest(BaseModel):
    """Schema for rendering node code"""
    node_type: str = Field(..., min_length=1, description="Node type")
    framework: str = Field(..., description="Framework (pytorch or tensorflow)")
    config: Dict[str, Any] = Field(..., description="Node configuration")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Node metadata")

    @validator('framework')
    def validate_framework(cls, v):
        """Ensure framework is valid"""
        if v not in ['pytorch', 'tensorflow']:
            raise ValueError('Framework must be "pytorch" or "tensorflow"')
        return v

    @validator('node_type')
    def validate_node_type(cls, v):
        """Ensure node type is not empty"""
        if not v or not v.strip():
            raise ValueError('Node type cannot be empty')
        return v.strip()
