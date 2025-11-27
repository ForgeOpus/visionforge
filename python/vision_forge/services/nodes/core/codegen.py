"""
Code generation utilities for rendering node templates.
Provides Jinja2-based template rendering for generating framework code.
"""

from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound

from .base import NodeDefinition
from .types import Framework, TensorShape


@dataclass
class RenderedTemplate:
    """Result of rendering a node template"""
    code: str
    context: Dict[str, Any]


class TemplateRenderer:
    """
    Renderer for node code templates using Jinja2.

    Loads templates from framework-specific directories and renders them
    with node configuration and computed context values.

    Template locations:
    - PyTorch: block_manager/services/nodes/frameworks/pytorch/templates/
    - TensorFlow: block_manager/services/nodes/frameworks/tensorflow/templates/

    Usage:
        renderer = TemplateRenderer()

        # Render a node's template
        result = renderer.render(
            node=linear_node,
            config={"out_features": 128},
            input_shape=TensorShape(dims=[32, 512])
        )
        print(result.code)  # "nn.Linear(512, 128, bias=True)"
    """

    def __init__(self, base_path: Optional[Path] = None):
        """
        Initialize the template renderer.

        Args:
            base_path: Base path for template directories.
                      Defaults to block_manager/services/nodes/frameworks/
        """
        if base_path is None:
            # Default to frameworks directory relative to this file
            base_path = Path(__file__).parent.parent / "frameworks"

        self.base_path = base_path
        self._environments: Dict[Framework, Environment] = {}

    def _get_environment(self, framework: Framework) -> Environment:
        """Get or create Jinja2 environment for a framework"""
        if framework not in self._environments:
            template_dir = self.base_path / framework.value / "templates"

            # Create directory if it doesn't exist
            template_dir.mkdir(parents=True, exist_ok=True)

            self._environments[framework] = Environment(
                loader=FileSystemLoader(str(template_dir)),
                undefined=StrictUndefined,
                trim_blocks=True,
                lstrip_blocks=True,
                # Enable autoescape for safety (can be disabled per-template)
                autoescape=False,
            )

            # Add custom filters
            self._environments[framework].filters['lower'] = str.lower
            self._environments[framework].filters['upper'] = str.upper
            self._environments[framework].filters['bool_lower'] = lambda x: str(x).lower()

        return self._environments[framework]

    def render(
        self,
        node: NodeDefinition,
        config: Dict[str, Any],
        input_shape: Optional[TensorShape] = None,
        extra_context: Optional[Dict[str, Any]] = None
    ) -> RenderedTemplate:
        """
        Render a node's code template.

        Args:
            node: Node definition to render
            config: Node configuration dictionary
            input_shape: Optional input tensor shape
            extra_context: Additional context variables

        Returns:
            RenderedTemplate with generated code and context used

        Raises:
            TemplateNotFound: If template file doesn't exist
            jinja2.UndefinedError: If required context variable is missing
        """
        framework = node.metadata.framework
        env = self._get_environment(framework)

        # Build context from node
        context = node.get_template_context(config, input_shape)

        # Merge extra context
        if extra_context:
            context["extra"] = extra_context

        # Load and render template
        template_name = f"{node.template_name}.jinja2"

        try:
            template = env.get_template(template_name)
            rendered = template.render(**context)
            return RenderedTemplate(code=rendered.strip(), context=context)
        except TemplateNotFound:
            # Provide helpful error message
            template_path = self.base_path / framework.value / "templates" / template_name
            raise TemplateNotFound(
                f"Template not found: {template_path}. "
                f"Create a Jinja2 template file for the '{node.metadata.type}' node."
            )

    def render_string(
        self,
        template_string: str,
        context: Dict[str, Any],
        framework: Framework = Framework.PYTORCH
    ) -> str:
        """
        Render a template string directly.

        Useful for inline templates or testing.

        Args:
            template_string: Jinja2 template content
            context: Context dictionary
            framework: Framework for environment settings

        Returns:
            Rendered string
        """
        env = self._get_environment(framework)
        template = env.from_string(template_string)
        return template.render(**context).strip()

    def template_exists(self, node: NodeDefinition) -> bool:
        """Check if a template file exists for a node"""
        framework = node.metadata.framework
        template_path = (
            self.base_path /
            framework.value /
            "templates" /
            f"{node.template_name}.jinja2"
        )
        return template_path.exists()


# Global renderer instance
_global_renderer = TemplateRenderer()


def render_node_template(
    node: NodeDefinition,
    config: Dict[str, Any],
    input_shape: Optional[TensorShape] = None,
    extra_context: Optional[Dict[str, Any]] = None
) -> RenderedTemplate:
    """
    Render a node's code template using the global renderer.

    Args:
        node: Node definition to render
        config: Node configuration dictionary
        input_shape: Optional input tensor shape
        extra_context: Additional context variables

    Returns:
        RenderedTemplate with generated code and context

    Example:
        from block_manager.services.nodes.core import (
            render_node_template,
            get_node_definition,
            TensorShape
        )

        linear = get_node_definition("linear")
        result = render_node_template(
            node=linear,
            config={"out_features": 128, "bias": True},
            input_shape=TensorShape(dims=[32, 512])
        )
        print(result.code)  # "nn.Linear(512, 128, bias=True)"
    """
    return _global_renderer.render(node, config, input_shape, extra_context)


def get_renderer() -> TemplateRenderer:
    """Get the global template renderer instance"""
    return _global_renderer
