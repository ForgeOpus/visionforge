from __future__ import annotations

import hashlib
import json
from typing import Any, Dict

from .models import ConfigFieldSpec, ConfigOptionSpec, NodeSpec, NodeTemplateSpec


def _option_to_dict(option: ConfigOptionSpec) -> Dict[str, Any]:
    return {
        "value": option.value,
        "label": option.label,
    }


def _field_to_dict(field: ConfigFieldSpec) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "name": field.name,
        "label": field.label,
        "type": field.field_type,
    }
    if field.required:
        payload["required"] = True
    if field.default is not None:
        payload["default"] = field.default
    if field.min is not None:
        payload["min"] = field.min
    if field.max is not None:
        payload["max"] = field.max
    if field.description:
        payload["description"] = field.description
    if field.placeholder:
        payload["placeholder"] = field.placeholder
    if field.accept is not None:
        payload["accept"] = field.accept
    if field.options:
        payload["options"] = [_option_to_dict(option) for option in field.options]
    return payload


def _template_to_dict(template: NodeTemplateSpec) -> Dict[str, Any]:
    return {
        "name": template.name,
        "engine": template.engine,
        "content": template.content,
        "defaultContext": template.default_context,
    }


def spec_to_dict(spec: NodeSpec) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "type": spec.type,
        "label": spec.label,
        "category": spec.category,
        "color": spec.color,
        "icon": spec.icon,
        "description": spec.description,
        "framework": spec.framework.value,
        "allowsMultipleInputs": spec.allows_multiple_inputs,
        "configSchema": [_field_to_dict(field) for field in spec.config_schema],
        "metadata": spec.metadata,
    }
    if spec.shape_fn:
        payload["shapeFn"] = spec.shape_fn
    if spec.validation_fn:
        payload["validationFn"] = spec.validation_fn
    if spec.template:
        payload["template"] = _template_to_dict(spec.template)
    payload["hash"] = compute_spec_hash(payload)
    return payload


def compute_spec_hash(payload: Dict[str, Any]) -> str:
    """Compute a deterministic hash for the given payload."""

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
