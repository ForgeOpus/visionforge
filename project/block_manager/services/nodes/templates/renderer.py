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

    final_config = {**spec.default_config(), **(config or {})}
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
