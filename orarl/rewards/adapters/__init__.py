"""Dependency-light reward adapters shipped with OraRL.

Modules are intentionally not imported here so the reward router can load each
task family lazily.
"""

__all__ = [
    "segmentation",
    "spatial_grounding",
    "spatial_intelligence",
    "spatial_temporal_grounding",
    "temporal_grounding",
    "tracking",
    "video_qa",
]
