"""Interfaz de linea de comandos para el simulador de Michelson.

Este archivo no contiene la fisica principal; solo traduce parametros escritos
por el usuario en consola hacia objetos de `simulation.py`, ejecuta el calculo
y guarda resultados faciles de abrir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .render import save_pattern_png, save_profile_plot_png
from .simulation import (
    Camera,
    MichelsonConfig,
    fringe_spacing_m,
    fwhm_from_curve,
    microdegrees_to_rad,
    microradians_to_rad,
    monochromatic_spectrum,
    profile_along_fringe_normal,
    rad_to_microdegrees,
    rectangular_spectrum,
    save_npz,
    save_profile_csv,
    simulate_camera,
    spectrum_from_csv,
    theoretical_rectangular_coherence_length_fwhm_m,
    theoretical_rectangular_first_zero_m,
)

DEFAULT_MIRROR_2_AZIMUTH_UDEG = float(rad_to_microdegrees(2500e-6))
DEFAULT_MIRROR_2_CENITAL_UDEG = float(rad_to_microdegrees(500e-6))


def parse_args() -> argparse.Namespace:
    """Define las variables manipulables desde la terminal."""

    parser = argparse.ArgumentParser(
        description="Simula el patron de interferencia de un interferometro de Michelson.",
    )

    # Parametros de la fuente luminosa.
    parser.add_argument("--source", choices=["mono", "rectangular", "csv"], default="rectangular")
    parser.add_argument("--wavelength-nm", type=float, default=632.8, help="Longitud de onda central.")
    parser.add_argument("--width-nm", type=float, default=20.0, help="Ancho espectral para fuente rectangular.")
    parser.add_argument("--spectrum-samples", type=int, default=801)
    parser.add_argument("--spectrum-csv", type=Path, help="CSV con columnas wavelength_nm,weight.")

    # Inclinaciones absolutas de los dos espejos. La diferencia entre espejo 2 y
    # espejo 1 es la que produce la separacion y orientacion de las franjas.
    parser.add_argument("--mirror-1-azimuth-udeg", type=float, default=0.0)
    parser.add_argument("--mirror-1-cenital-udeg", type=float, default=0.0)
    parser.add_argument("--mirror-2-azimuth-udeg", type=float, default=DEFAULT_MIRROR_2_AZIMUTH_UDEG)
    parser.add_argument("--mirror-2-cenital-udeg", type=float, default=DEFAULT_MIRROR_2_CENITAL_UDEG)
    parser.add_argument("--mirror-1-azimuth-urad", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mirror-1-cenital-urad", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mirror-2-azimuth-urad", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--mirror-2-cenital-urad", type=float, default=None, help=argparse.SUPPRESS)

    # Alias de la version anterior: permiten especificar directamente la
    # inclinacion relativa sin pensar en cada espejo por separado.
    parser.add_argument(
        "--tilt-azimuth-udeg",
        type=float,
        default=None,
        help="Alias: inclinacion azimutal relativa espejo2-espejo1.",
    )
    parser.add_argument(
        "--tilt-cenital-udeg",
        "--tilt-cenith-udeg",
        dest="tilt_cenital_udeg",
        type=float,
        default=None,
        help="Alias: inclinacion cenital relativa espejo2-espejo1.",
    )
    parser.add_argument(
        "--tilt-azimuth-urad",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--tilt-cenital-urad",
        "--tilt-cenith-urad",
        dest="tilt_cenital_urad",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )

    # OPD central: opd0 es una diferencia fija; el desplazamiento del espejo 2
    # aporta 2*d por el recorrido ida-vuelta.
    parser.add_argument("--opd0-um", type=float, default=0.0, help="OPD fija central en micrometros.")
    parser.add_argument(
        "--mirror-2-displacement-um",
        "--movable-mirror-displacement-um",
        dest="mirror_2_displacement_um",
        type=float,
        default=0.0,
        help="Desplazamiento axial del espejo 2; cambia la OPD central en 2*d.",
    )

    # Parametros de muestreo y visualizacion.
    parser.add_argument("--pixels-x", type=int, default=900)
    parser.add_argument("--pixels-y", type=int, default=650)
    parser.add_argument("--camera-width-mm", type=float, default=8.0)
    parser.add_argument("--camera-height-mm", type=float, default=6.0)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--profile-zoom", type=float, default=1.0, help="Zoom visual del perfil perpendicular.")
    parser.add_argument("--profile-samples", type=int, default=2500)

    # Parametros experimentales de los dos haces.
    parser.add_argument("--intensity-1", type=float, default=1.0)
    parser.add_argument("--intensity-2", type=float, default=1.0)
    parser.add_argument("--contrast", type=float, default=1.0)

    # Carpeta donde se guardan imagenes, CSV y NPZ.
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def build_spectrum(args: argparse.Namespace):
    """Construye la fuente luminosa elegida por el usuario."""

    if args.source == "mono":
        return monochromatic_spectrum(args.wavelength_nm)
    if args.source == "rectangular":
        return rectangular_spectrum(args.wavelength_nm, args.width_nm, samples=args.spectrum_samples)
    if args.spectrum_csv is None:
        raise ValueError("--spectrum-csv is required when --source csv")
    return spectrum_from_csv(args.spectrum_csv)


def resolve_mirror_tilts(args: argparse.Namespace) -> tuple[float, float, float, float]:
    """Devuelve inclinaciones absolutas de espejo 1 y 2 en radianes."""

    mirror_1_azimuth = float(microdegrees_to_rad(args.mirror_1_azimuth_udeg))
    mirror_1_cenital = float(microdegrees_to_rad(args.mirror_1_cenital_udeg))
    mirror_2_azimuth = float(microdegrees_to_rad(args.mirror_2_azimuth_udeg))
    mirror_2_cenital = float(microdegrees_to_rad(args.mirror_2_cenital_udeg))

    if args.mirror_1_azimuth_urad is not None:
        mirror_1_azimuth = float(microradians_to_rad(args.mirror_1_azimuth_urad))
    if args.mirror_1_cenital_urad is not None:
        mirror_1_cenital = float(microradians_to_rad(args.mirror_1_cenital_urad))
    if args.mirror_2_azimuth_urad is not None:
        mirror_2_azimuth = float(microradians_to_rad(args.mirror_2_azimuth_urad))
    if args.mirror_2_cenital_urad is not None:
        mirror_2_cenital = float(microradians_to_rad(args.mirror_2_cenital_urad))

    if args.tilt_azimuth_udeg is not None:
        mirror_2_azimuth = mirror_1_azimuth + float(microdegrees_to_rad(args.tilt_azimuth_udeg))
    if args.tilt_cenital_udeg is not None:
        mirror_2_cenital = mirror_1_cenital + float(microdegrees_to_rad(args.tilt_cenital_udeg))
    if args.tilt_azimuth_urad is not None:
        mirror_2_azimuth = mirror_1_azimuth + float(microradians_to_rad(args.tilt_azimuth_urad))
    if args.tilt_cenital_urad is not None:
        mirror_2_cenital = mirror_1_cenital + float(microradians_to_rad(args.tilt_cenital_urad))

    return mirror_1_azimuth, mirror_1_cenital, mirror_2_azimuth, mirror_2_cenital


def make_summary(config: MichelsonConfig, spectrum, profile: dict[str, np.ndarray]) -> str:
    """Crea un resumen de parametros y resultados principales."""

    lambda0 = spectrum.representative_wavelength_m
    spacing = fringe_spacing_m(lambda0, config.relative_azimuth_rad, config.relative_cenital_rad)
    numeric_lc = fwhm_from_curve(profile["opd_m"], profile["visibility"])

    lines = [
        "Michelson interference simulation",
        "",
        f"Source: {spectrum.name}",
        f"Representative wavelength: {lambda0 * 1e9:.6g} nm",
        f"Mirror 1 tilt: azimuth={float(rad_to_microdegrees(config.mirror_1_azimuth_rad)):.6g} u°, cenital={float(rad_to_microdegrees(config.mirror_1_cenital_rad)):.6g} u°",
        f"Mirror 2 tilt: azimuth={float(rad_to_microdegrees(config.mirror_2_azimuth_rad)):.6g} u°, cenital={float(rad_to_microdegrees(config.mirror_2_cenital_rad)):.6g} u°",
        f"Relative tilt mirror2-mirror1: azimuth={float(rad_to_microdegrees(config.relative_azimuth_rad)):.6g} u°, cenital={float(rad_to_microdegrees(config.relative_cenital_rad)):.6g} u°",
        f"Relative tilt magnitude: {float(rad_to_microdegrees(config.tilt_magnitude_rad)):.6g} u°",
        f"Fixed OPD offset: {config.opd0_m * 1e6:.6g} um",
        f"Mirror 2 displacement: {config.mirror_2_displacement_m * 1e6:.6g} um",
        f"Central OPD including displacement: {config.center_opd_m * 1e6:.6g} um",
        f"Camera visible field: {config.camera.visible_width_m * 1e3:.6g} mm x {config.camera.visible_height_m * 1e3:.6g} mm",
        f"Camera zoom: {config.camera.zoom:.6g}",
        f"Fringe spacing along profile: {spacing * 1e3:.6g} mm",
    ]

    if np.isfinite(numeric_lc):
        lines.append(f"Coherence length from simulated visibility FWHM: {numeric_lc * 1e6:.6g} um OPD")
    else:
        lines.append("Coherence length from simulated visibility FWHM: not covered by current field/tilt")

    if spectrum.shape == "rectangular":
        width_m = float(spectrum.wavelengths_m[-1] - spectrum.wavelengths_m[0])
        theoretical = theoretical_rectangular_coherence_length_fwhm_m(lambda0, width_m)
        first_zero = theoretical_rectangular_first_zero_m(lambda0, width_m)
        lines.append(f"Rectangular theoretical coherence FWHM: {theoretical * 1e6:.6g} um OPD")
        lines.append(f"Rectangular first zero estimate: {first_zero * 1e6:.6g} um OPD")
        if np.isfinite(numeric_lc):
            error = 100.0 * (numeric_lc - theoretical) / theoretical
            lines.append(f"Difference simulated vs theoretical: {error:.3g} %")
    elif spectrum.is_monochromatic:
        lines.append("Theoretical coherence length: infinite for ideal monochromatic source")
    else:
        lines.append("Theoretical coherence length: use numeric visibility for custom spectra")

    lines.extend(
        [
            "",
            "Generated files:",
            "  pattern.png: normalized camera intensity",
            "  profile.csv: line profile perpendicular to fringes",
            "  profile.png: quick plot of intensity and visibility",
            "  simulation.npz: arrays for further analysis",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Punto de entrada cuando se ejecuta `python -m michelson.cli`."""

    args = parse_args()
    spectrum = build_spectrum(args)

    mirror_1_azimuth, mirror_1_cenital, mirror_2_azimuth, mirror_2_cenital = resolve_mirror_tilts(args)

    camera = Camera(
        pixels_x=args.pixels_x,
        pixels_y=args.pixels_y,
        width_m=args.camera_width_mm * 1e-3,
        height_m=args.camera_height_mm * 1e-3,
        zoom=args.zoom,
    )
    config = MichelsonConfig(
        mirror_1_azimuth_rad=mirror_1_azimuth,
        mirror_1_cenital_rad=mirror_1_cenital,
        mirror_2_azimuth_rad=mirror_2_azimuth,
        mirror_2_cenital_rad=mirror_2_cenital,
        mirror_2_displacement_m=args.mirror_2_displacement_um * 1e-6,
        opd0_m=args.opd0_um * 1e-6,
        intensity_1=args.intensity_1,
        intensity_2=args.intensity_2,
        contrast=args.contrast,
        camera=camera,
    )

    # Ejecutar simulacion y guardar artefactos.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = simulate_camera(config, spectrum)
    profile = profile_along_fringe_normal(config, spectrum, samples=args.profile_samples)

    save_pattern_png(args.output_dir / "pattern.png", result.intensity)
    save_profile_csv(args.output_dir / "profile.csv", profile)
    save_profile_plot_png(args.output_dir / "profile.png", profile, profile_zoom=args.profile_zoom)
    save_npz(args.output_dir / "simulation.npz", result, profile)

    summary = make_summary(config, spectrum, profile)
    (args.output_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
