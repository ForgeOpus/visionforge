"""
Template Manager for Code Generation
Handles loading and caching of Jinja2 templates
"""

from jinja2 import Environment, PackageLoader, Template
from typing import Dict, Optional


class TemplateManager:
    """
    Manages Jinja2 template loading and caching for code generation.
    Singleton pattern ensures templates are loaded once and reused.
    """

    _instance: Optional['TemplateManager'] = None
    _env: Optional[Environment] = None
    _template_cache: Dict[str, Template] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_environment(cls) -> Environment:
        """Get or create the Jinja2 environment"""
        if cls._env is None:
            cls._env = Environment(
                loader=PackageLoader(
                    'block_manager.services.nodes',
                    'templates'
                ),
                autoescape=False,  # Generating Python code, not HTML
                trim_blocks=True,
                lstrip_blocks=True,
                keep_trailing_newline=True
            )

            # Add custom filters if needed
            cls._env.filters['repr'] = repr

        return cls._env

    @classmethod
    def get_template(cls, template_path: str) -> Template:
        """
        Get a template by path, with caching.

        Args:
            template_path: Path to template relative to templates directory
                          e.g., "pytorch/layers/conv2d.py.jinja2"

        Returns:
            Compiled Jinja2 template
        """
        if template_path not in cls._template_cache:
            env = cls.get_environment()
            cls._template_cache[template_path] = env.get_template(template_path)

        return cls._template_cache[template_path]

    @classmethod
    def render(cls, template_path: str, context: Dict) -> str:
        """
        Render a template with the given context.

        Args:
            template_path: Path to template
            context: Template variables

        Returns:
            Rendered template as string
        """
        template = cls.get_template(template_path)
        return template.render(**context)

    @classmethod
    def clear_cache(cls):
        """Clear the template cache (useful for testing)"""
        cls._template_cache.clear()
        cls._env = None
