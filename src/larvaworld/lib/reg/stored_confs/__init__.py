"""
This module sets up and stores configuration parameter sets for diverse elements of the platform
 including experiments, environments, larva models and tracker-specific data formats
 used to import experimental datasets from different labs.
"""

from __future__ import annotations

__displayname__ = "Available configurations"

__all__: list[str] = ["data_conf", "essay_conf", "sim_conf"]

from . import data_conf, essay_conf, sim_conf
