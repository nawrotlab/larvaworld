"""Model configuration JSON export utilities."""

from __future__ import annotations

import json
from typing import Any

__all__ = ["export_model_config_to_json"]


def export_model_config_to_json(
    model_id: str,
    config: dict[str, Any],
    compare_model_id: str | None = None,
    compare_config: dict[str, Any] | None = None,
) -> str:
    """Generate JSON export for model configuration(s).

    Parameters
    ----------
    model_id : str
        Primary model ID
    config : dict
        Primary model configuration dictionary
    compare_model_id : str, optional
        Secondary model ID for comparison
    compare_config : dict, optional
        Secondary model configuration dictionary

    Returns
    -------
    str
        Formatted JSON string with configuration(s)
    """

    export_data = {
        "export": {
            "timestamp": _get_timestamp(),
            "format": "larvaworld-model-config-1.0",
        },
        "primary_model": {
            "id": model_id,
            "config": config,
        },
    }

    if compare_model_id and compare_config:
        export_data["comparison_model"] = {
            "id": compare_model_id,
            "config": compare_config,
        }
        export_data["diff"] = _compute_config_diff(config, compare_config)

    return json.dumps(export_data, indent=2, default=str)


def _get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    from datetime import datetime

    return datetime.now().isoformat()


def _compute_config_diff(
    config1: dict[str, Any],
    config2: dict[str, Any],
) -> dict[str, Any]:
    """Compute differences between two configurations.

    Returns only keys that differ between configs.
    """
    diff = {
        "different_keys": {},
        "only_in_primary": {},
        "only_in_comparison": {},
    }

    all_keys = set(config1.keys()) | set(config2.keys())

    for key in all_keys:
        val1 = config1.get(key)
        val2 = config2.get(key)

        if key not in config1:
            diff["only_in_comparison"][key] = val2
        elif key not in config2:
            diff["only_in_primary"][key] = val1
        elif val1 != val2:
            diff["different_keys"][key] = {
                "primary": val1,
                "comparison": val2,
            }

    return diff


def format_export_filename(model_id: str, compare_model_id: str | None = None) -> str:
    """Generate a filename for the exported configuration.

    Parameters
    ----------
    model_id : str
        Primary model ID
    compare_model_id : str, optional
        Comparison model ID

    Returns
    -------
    str
        Suggested filename (without path)
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if compare_model_id:
        return f"model_comparison_{model_id}_vs_{compare_model_id}_{timestamp}.json"
    else:
        return f"model_config_{model_id}_{timestamp}.json"
