"""Paquete publico del simulador de interferometro de Michelson.

Importar desde `michelson` da acceso directo a las clases y funciones mas
usadas sin tener que recordar en que archivo viven.
"""

from .simulation import (
    Camera,
    MichelsonConfig,
    SimulationResult,
    Spectrum,
    estimate_visibility_from_profile,
    fringe_spacing_m,
    microdegrees_to_rad,
    microradians_to_rad,
    monochromatic_spectrum,
    profile_along_fringe_normal,
    rad_to_microdegrees,
    rad_to_microradians,
    rectangular_spectrum,
    simulate_camera,
    spectrum_from_csv,
    theoretical_rectangular_coherence_length_fwhm_m,
    theoretical_rectangular_first_zero_m,
)

# Lista explicita de simbolos publicos que se exportan con `from michelson import *`.
__all__ = [
    "Camera",
    "MichelsonConfig",
    "SimulationResult",
    "Spectrum",
    "estimate_visibility_from_profile",
    "fringe_spacing_m",
    "microdegrees_to_rad",
    "microradians_to_rad",
    "monochromatic_spectrum",
    "profile_along_fringe_normal",
    "rad_to_microdegrees",
    "rad_to_microradians",
    "rectangular_spectrum",
    "simulate_camera",
    "spectrum_from_csv",
    "theoretical_rectangular_coherence_length_fwhm_m",
    "theoretical_rectangular_first_zero_m",
]
