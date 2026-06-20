"""Pruebas numericas basicas del simulador.

Estas pruebas no intentan cubrir una interfaz de usuario; solo verifican que
las ecuaciones principales conserven los factores fisicos importantes.
"""

import math
import unittest

import numpy as np

from michelson.simulation import (
    Camera,
    MichelsonConfig,
    fwhm_from_curve,
    fringe_spacing_m,
    intensity_from_opd,
    microdegrees_to_rad,
    microradians_to_rad,
    monochromatic_spectrum,
    profile_along_fringe_normal,
    rad_to_microdegrees,
    rectangular_spectrum,
    theoretical_rectangular_coherence_length_fwhm_m,
)


class SimulationTests(unittest.TestCase):
    """Casos pequenos que corren rapido y detectan cambios peligrosos."""

    def test_fringe_spacing_uses_double_mirror_tilt(self):
        """El periodo debe incluir el factor 2 por reflexion del espejo."""

        wavelength = 632.8e-9
        spacing = fringe_spacing_m(wavelength, 100e-6, 0.0)
        self.assertAlmostEqual(spacing, wavelength / (2.0 * 100e-6))

    def test_legacy_relative_tilt_alias_still_works(self):
        """tilt_x_rad/tilt_y_rad siguen funcionando como alias relativos."""

        config = MichelsonConfig(tilt_x_rad=50e-6, tilt_y_rad=10e-6)
        self.assertAlmostEqual(config.relative_azimuth_rad, 50e-6)
        self.assertAlmostEqual(config.relative_cenital_rad, 10e-6)
        self.assertAlmostEqual(config.mirror_2_azimuth_rad, 50e-6)
        self.assertAlmostEqual(config.mirror_2_cenital_rad, 10e-6)

    def test_microdegree_conversions_match_radians(self):
        """La interfaz usa microgrados, pero el nucleo conserva radianes."""

        self.assertAlmostEqual(microdegrees_to_rad(180.0e6), math.pi)
        self.assertAlmostEqual(rad_to_microdegrees(math.pi), 180.0e6)
        self.assertAlmostEqual(microradians_to_rad(100.0), 100e-6)

    def test_two_absolute_mirror_tilts_reduce_to_relative_tilt(self):
        """Si ambos espejos se inclinan, las franjas dependen de la diferencia."""

        config = MichelsonConfig(
            mirror_1_azimuth_rad=300e-6,
            mirror_1_cenital_rad=100e-6,
            mirror_2_azimuth_rad=2800e-6,
            mirror_2_cenital_rad=600e-6,
        )
        self.assertAlmostEqual(config.relative_azimuth_rad, 2500e-6)
        self.assertAlmostEqual(config.relative_cenital_rad, 500e-6)

    def test_movable_mirror_displacement_doubles_center_opd(self):
        """Mover un espejo d cambia la OPD central en 2d."""

        config = MichelsonConfig(
            mirror_2_displacement_m=5e-6,
            mirror_2_azimuth_rad=0.0,
            mirror_2_cenital_rad=0.0,
        )
        self.assertAlmostEqual(config.center_opd_m, 10e-6)
        self.assertAlmostEqual(float(config.opd(np.array([0.0]), np.array([0.0]))[0]), 10e-6)

    def test_monochromatic_equal_beams_are_bounded_between_zero_and_two(self):
        """Dos haces iguales oscilan entre interferencia constructiva y destructiva."""

        config = MichelsonConfig(tilt_x_rad=0.0, tilt_y_rad=0.0)
        spectrum = monochromatic_spectrum(632.8)
        opd = np.array([0.0, 632.8e-9 / 2.0])
        intensity, visibility = intensity_from_opd(config, spectrum, opd)
        self.assertAlmostEqual(float(intensity[0]), 2.0, places=12)
        self.assertAlmostEqual(float(intensity[1]), 0.0, places=12)
        self.assertTrue(np.allclose(visibility, 1.0))

    def test_rectangular_spectrum_is_uniform_over_requested_width(self):
        """La fuente rectangular cubre una banda uniforme de longitud de onda."""

        spectrum = rectangular_spectrum(center_nm=632.8, width_nm=10.0, samples=101)
        self.assertEqual(spectrum.shape, "rectangular")
        self.assertAlmostEqual(float(spectrum.wavelengths_m[0]), 627.8e-9)
        self.assertAlmostEqual(float(spectrum.wavelengths_m[-1]), 637.8e-9)
        self.assertTrue(np.allclose(spectrum.weights, np.full(101, 1.0 / 101)))
        self.assertTrue(np.all(np.isfinite(spectrum.visibility_envelope(np.linspace(-1e-6, 1e-6, 5)))))

    def test_rectangular_numeric_coherence_matches_sinc_fwhm_approximation(self):
        """El FWHM numerico rectangular debe acercarse a la teoria de sinc."""

        spectrum = rectangular_spectrum(center_nm=632.8, width_nm=10.0, samples=2401)
        theoretical = theoretical_rectangular_coherence_length_fwhm_m(632.8e-9, 10.0e-9)
        opd = np.linspace(-2.0 * theoretical, 2.0 * theoretical, 3500)
        numeric = fwhm_from_curve(opd, spectrum.visibility_envelope(opd))
        self.assertTrue(math.isfinite(numeric))
        self.assertLess(abs(numeric - theoretical) / theoretical, 0.05)

    def test_profile_runs_through_camera_center_and_reports_opd(self):
        """El punto central del perfil debe coincidir con la OPD central."""

        camera = Camera(pixels_x=100, pixels_y=80, width_m=8e-3, height_m=6e-3)
        config = MichelsonConfig(tilt_x_rad=50e-6, tilt_y_rad=50e-6, opd0_m=3e-6, camera=camera)
        profile = profile_along_fringe_normal(config, monochromatic_spectrum(), samples=101)
        center = 50
        self.assertAlmostEqual(float(profile["x_m"][center]), 0.0, places=15)
        self.assertAlmostEqual(float(profile["y_m"][center]), 0.0, places=15)
        self.assertAlmostEqual(float(profile["opd_m"][center]), 3e-6, places=15)

    def test_camera_zoom_does_not_shrink_profile_width(self):
        """El zoom de camara no debe cambiar el ancho fisico del perfil."""

        camera = Camera(pixels_x=100, pixels_y=80, width_m=8e-3, height_m=6e-3, zoom=4.0)
        config = MichelsonConfig(tilt_x_rad=50e-6, tilt_y_rad=0.0, camera=camera)
        profile = profile_along_fringe_normal(config, monochromatic_spectrum(), samples=11)
        self.assertAlmostEqual(float(profile["s_m"][0]), -4e-3)
        self.assertAlmostEqual(float(profile["s_m"][-1]), 4e-3)


if __name__ == "__main__":
    unittest.main()
