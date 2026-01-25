"""Configuration classes for simulation"""

from weon_garment_code.config.dataset_properties import (
    DatasetProperties,
    RenderProperties,
    SimulationProperties,
)
from weon_garment_code.config.garment_properties import GarmentMaterialProperties
from weon_garment_code.config.paths_config import PathConfig
from weon_garment_code.config.render_config import RenderConfig, UVTextureConfig
from weon_garment_code.config.simulation_control import SimulationControl
from weon_garment_code.config.simulation_options import (
    AttachmentConstraint,
    SimulationOptions,
)

__all__ = [
    "SimulationControl",
    "GarmentMaterialProperties",
    "SimulationOptions",
    "AttachmentConstraint",
    "PathConfig",
    "RenderConfig",
    "UVTextureConfig",
    "DatasetProperties",
    "SimulationProperties",
    "RenderProperties",
]
