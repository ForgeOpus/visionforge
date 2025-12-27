"""
JSON schema validators for block_manager models.
"""
import json
from django.core.exceptions import ValidationError


def validate_canvas_state(value):
    """
    Validate canvas_state JSON structure.
    """
    if not isinstance(value, dict):
        raise ValidationError("Canvas state must be a dictionary")

    if 'nodes' not in value or not isinstance(value.get('nodes'), list):
        raise ValidationError("Canvas state must contain a 'nodes' list")

    if 'edges' not in value or not isinstance(value.get('edges'), list):
        raise ValidationError("Canvas state must contain an 'edges' list")

    # Validate node structure
    for node in value.get('nodes', []):
        if not isinstance(node, dict):
            raise ValidationError("Each node must be a dictionary")
        if 'id' not in node:
            raise ValidationError("Each node must have an 'id' field")
        if 'data' not in node or not isinstance(node.get('data'), dict):
            raise ValidationError("Each node must have a 'data' dictionary")


def validate_block_config(value):
    """
    Validate block configuration JSON.
    """
    if not isinstance(value, dict):
        raise ValidationError("Block config must be a dictionary")

    # Max size check to prevent DoS
    json_str = json.dumps(value)
    if len(json_str) > 10000:  # 10KB limit
        raise ValidationError("Block config exceeds maximum size")


def validate_group_internal_structure(value):
    """
    Validate group block internal structure.
    """
    if not isinstance(value, dict):
        raise ValidationError("Internal structure must be a dictionary")

    if 'nodes' in value and not isinstance(value['nodes'], list):
        raise ValidationError("Internal structure 'nodes' must be a list")

    if 'edges' in value and not isinstance(value['edges'], list):
        raise ValidationError("Internal structure 'edges' must be a list")

    # Max size check
    json_str = json.dumps(value)
    if len(json_str) > 50000:  # 50KB limit for group structures
        raise ValidationError("Internal structure exceeds maximum size")


def validate_shape_data(value):
    """
    Validate shape data (input_shape, output_shape).
    """
    if value is None:
        return

    if not isinstance(value, dict):
        raise ValidationError("Shape data must be a dictionary")

    # Max size check
    json_str = json.dumps(value)
    if len(json_str) > 1000:  # 1KB limit for shape data
        raise ValidationError("Shape data exceeds maximum size")
