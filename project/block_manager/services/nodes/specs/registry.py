from __future__ import annotations

import importlib
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

from .models import Framework, NodeSpec


_SPEC_MODULES = (
    "block_manager.services.nodes.specs.pytorch",
    "block_manager.services.nodes.specs.tensorflow",
)


@lru_cache(maxsize=1)
def _load_spec_map() -> Dict[Framework, Dict[str, NodeSpec]]:
    """Load and cache node specifications for all frameworks."""

    spec_map: Dict[Framework, Dict[str, NodeSpec]] = {
        Framework.PYTORCH: {},
        Framework.TENSORFLOW: {},
    }

    for module_path in _SPEC_MODULES:
        module = importlib.import_module(module_path)
        specs: Iterable[NodeSpec] = getattr(module, "NODE_SPECS")
        for spec in specs:
            bucket = spec_map.setdefault(spec.framework, {})
            bucket[spec.type] = spec
    return spec_map


def list_node_specs(framework: Framework) -> List[NodeSpec]:
    """Return all node specifications for the given framework."""

    return list(_load_spec_map().get(framework, {}).values())


def get_node_spec(node_type: str, framework: Framework) -> Optional[NodeSpec]:
    """Retrieve a specific node specification."""

    return _load_spec_map().get(framework, {}).get(node_type)


def iter_all_specs() -> Iterable[NodeSpec]:
    """Iterate over every registered node specification."""

    for framework_map in _load_spec_map().values():
        yield from framework_map.values()


def reset_spec_cache() -> None:
    """Reset the cached specification map (primarily for testing)."""

    _load_spec_map.cache_clear()
