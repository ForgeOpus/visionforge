from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from jinja2 import Environment, StrictUndefined

from ..specs.models import NodeSpec


@dataclass(frozen=True)
class RenderedTemplate:
    code: str
    context: Dict[str, Any]


def _build_environment() -> Environment:
    return Environment(undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)


def render_node_template(
    spec: NodeSpec,
    config: Optional[Dict[str, Any]] = None,
    extra_context: Optional[Dict[str, Any]] = None,
) -> RenderedTemplate:
    if not spec.template:
        raise ValueError(f"Node '{spec.type}' does not expose a template for rendering")

    env = _build_environment()
    template = env.from_string(spec.template.content)

    # Start with defaults from schema, then merge user config
    final_config = {**spec.default_config(), **(config or {})}
    
    # Fill in missing required fields with sensible placeholders
    for field_spec in spec.config_schema:
        if field_spec.required and field_spec.name not in final_config:
            # Provide type-appropriate placeholders for required fields
            if field_spec.field_type == "number":
                # Use min value if available, otherwise a sensible default (64 is common for channels/features)
                final_config[field_spec.name] = int(field_spec.min) if field_spec.min is not None else 64
            elif field_spec.field_type == "select" and field_spec.options:
                # Use first option value
                final_config[field_spec.name] = field_spec.options[0].value
            elif field_spec.field_type == "boolean":
                final_config[field_spec.name] = False
            else:
                # String or unknown type - use placeholder if available
                final_config[field_spec.name] = field_spec.placeholder or "placeholder"
    
    context = {
        "config": final_config,
        "metadata": {
            "type": spec.type,
            "framework": spec.framework.value,
            **spec.metadata,
        },
        "context": {**spec.template.default_context, **(extra_context or {})},
    }

    rendered = template.render(**context)
    return RenderedTemplate(code=rendered.strip(), context=context)
