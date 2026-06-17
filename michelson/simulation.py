"""Ecuaciones base para simular un interferometro de Michelson.

El programa modela lo que ve una camara cuando dos haces colimados se
recombinan. Cada espejo puede inclinarse en dos grados de libertad:

- azimutal: componente horizontal, asociada al eje x de la camara.
- cenital: componente vertical, asociada al eje y de la camara.

Para angulos pequenos, el patron depende de la inclinacion relativa entre los
dos espejos. Si el espejo 2 esta mas inclinado que el espejo 1, la diferencia
de camino optico (OPD) cambia linealmente sobre la camara:

    OPD(x, y) = OPD_central
                + 2 * (tilt_rel_azimutal * x + tilt_rel_cenital * y)

El factor 2 aparece porque la reflexion duplica el cambio angular del haz. Si
un espejo se desplaza axialmente una distancia d, la OPD cambia 2d porque la
luz recorre ese cambio en ida y vuelta.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class Camera:
    """Define cuantos pixeles se calculan y que area fisica ve la camara."""

    # Numero de muestras horizontales y verticales del patron.
    pixels_x: int = 900
    pixels_y: int = 650

    # Campo de vision fisico de la camara cuando zoom = 1.
    width_m: float = 8.0e-3
    height_m: float = 6.0e-3

    # Zoom numerico: aumenta el patron sin cambiar la fisica del interferometro.
    zoom: float = 1.0

    def __post_init__(self) -> None:
        """Valida que la camara tenga una malla fisicamente util."""

        if self.pixels_x < 2 or self.pixels_y < 2:
            raise ValueError("pixels_x and pixels_y must be at least 2")
        if self.width_m <= 0 or self.height_m <= 0:
            raise ValueError("camera width and height must be positive")
        if self.zoom <= 0:
            raise ValueError("zoom must be positive")

    @property
    def visible_width_m(self) -> float:
        """Ancho real que queda visible despues de aplicar zoom."""

        return self.width_m / self.zoom

    @property
    def visible_height_m(self) -> float:
        """Alto real que queda visible despues de aplicar zoom."""

        return self.height_m / self.zoom

    def grid(self) -> tuple[np.ndarray, np.ndarray]:
        """Construye las coordenadas x,y de cada pixel en metros."""

        # La camara se centra en (0, 0), por eso los limites son simetricos.
        x = np.linspace(
            -0.5 * self.visible_width_m,
            0.5 * self.visible_width_m,
            self.pixels_x,
        )
        y = np.linspace(
            -0.5 * self.visible_height_m,
            0.5 * self.visible_height_m,
            self.pixels_y,
        )
        return np.meshgrid(x, y)


@dataclass(frozen=True)
class Spectrum:
    """Espectro discreto de la fuente, muestreado en longitud de onda."""

    # Longitudes de onda en metros y pesos relativos de intensidad.
    wavelengths_m: np.ndarray
    weights: np.ndarray

    # Metadatos usados para reportes y aproximaciones analiticas.
    name: str = "custom"
    center_wavelength_m: float | None = None
    fwhm_m: float | None = None
    shape: str = "custom"

    def __post_init__(self) -> None:
        """Convierte entradas a arreglos de numpy y normaliza los pesos."""

        wavelengths = np.asarray(self.wavelengths_m, dtype=float)
        weights = np.asarray(self.weights, dtype=float)

        if wavelengths.ndim != 1 or weights.ndim != 1:
            raise ValueError("wavelengths_m and weights must be 1-D arrays")
        if wavelengths.size != weights.size:
            raise ValueError("wavelengths_m and weights must have same length")
        if wavelengths.size == 0:
            raise ValueError("spectrum must contain at least one wavelength")
        if np.any(wavelengths <= 0):
            raise ValueError("wavelengths must be positive")
        if np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("weights must be non-negative and not all zero")

        # Normalizar hace que sum(weights) = 1 y facilita interpretar gamma.
        normalized = weights / np.sum(weights)
        object.__setattr__(self, "wavelengths_m", wavelengths)
        object.__setattr__(self, "weights", normalized)

    @property
    def is_monochromatic(self) -> bool:
        """Indica si la fuente ideal contiene una sola longitud de onda."""

        return self.wavelengths_m.size == 1

    @property
    def representative_wavelength_m(self) -> float:
        """Longitud de onda usada para escalas de franjas y reportes."""

        if self.center_wavelength_m is not None:
            return self.center_wavelength_m
        return float(np.sum(self.wavelengths_m * self.weights))

    def complex_coherence(self, opd_m: np.ndarray, chunk_size: int = 64) -> np.ndarray:
        """Calcula la coherencia compleja gamma(OPD).

        gamma(OPD) contiene dos cosas a la vez:

        - su fase genera las franjas claras y oscuras;
        - su magnitud es la envolvente de visibilidad.

        Para espectros gaussianos se usa la aproximacion analitica de banda
        angosta. Para espectros CSV personalizados se integra numericamente.
        """

        opd = np.asarray(opd_m, dtype=float)

        # Caso gaussiano: expresion cerrada para la envolvente de coherencia.
        if self.shape == "gaussian" and self.center_wavelength_m and self.fwhm_m:
            lc_fwhm = theoretical_gaussian_coherence_length_fwhm_m(
                self.center_wavelength_m,
                self.fwhm_m,
            )
            envelope = np.exp(-4.0 * np.log(2.0) * (opd / lc_fwhm) ** 2)
            phase = 2.0 * np.pi * opd / self.center_wavelength_m
            return envelope * np.exp(1j * phase)

        # Caso monocromatico: no hay perdida de visibilidad con OPD.
        if self.is_monochromatic:
            phase = 2.0 * np.pi * opd / self.wavelengths_m[0]
            return np.exp(1j * phase)

        # Caso general: suma ponderada de muchas ondas monocromaticas.
        gamma = np.zeros(opd.shape, dtype=np.complex128)
        flat = opd.ravel()
        gamma_flat = gamma.ravel()

        # Se procesa en bloques para no crear matrices gigantes si la camara o
        # el espectro tienen muchas muestras.
        for start in range(0, self.wavelengths_m.size, chunk_size):
            stop = min(start + chunk_size, self.wavelengths_m.size)
            lambdas = self.wavelengths_m[start:stop]
            weights = self.weights[start:stop]
            phase = (2.0 * np.pi * flat[None, :]) / lambdas[:, None]
            gamma_flat += np.sum(weights[:, None] * np.exp(1j * phase), axis=0)

        return gamma

    def visibility_envelope(self, opd_m: np.ndarray, chunk_size: int = 64) -> np.ndarray:
        """Devuelve |gamma|, que es el contraste maximo posible local."""

        return np.abs(self.complex_coherence(opd_m, chunk_size=chunk_size))


@dataclass(frozen=True)
class MichelsonConfig:
    """Parametros fisicos de los dos brazos del interferometro."""

    # Inclinacion del espejo 1. Por defecto se deja como referencia sin tilt.
    mirror_1_azimuth_rad: float = 0.0
    mirror_1_cenital_rad: float = 0.0

    # Inclinacion del espejo 2. Estos valores por defecto generan franjas
    # visibles en una camara de pocos milimetros.
    mirror_2_azimuth_rad: float = 2500e-6
    mirror_2_cenital_rad: float = 500e-6

    # Desplazamiento axial del espejo 2. Si se mueve d, la OPD cambia 2d.
    mirror_2_displacement_m: float = 0.0

    # OPD adicional fija. Sirve para modelar una diferencia inicial entre brazos.
    opd0_m: float = 0.0

    # Intensidades relativas de los dos haces que llegan a la camara.
    intensity_1: float = 1.0
    intensity_2: float = 1.0

    # Factor experimental extra: 1 ideal, menor que 1 si hay perdidas de contraste.
    contrast: float = 1.0
    camera: Camera = Camera()

    # Alias heredados de la primera version. Si se pasan, se interpretan como
    # inclinacion relativa: espejo 2 = espejo 1 + tilt relativo.
    tilt_x_rad: float | None = None
    tilt_y_rad: float | None = None

    def __post_init__(self) -> None:
        """Valida parametros y resuelve alias de compatibilidad."""

        # Compatibilidad: permite seguir usando MichelsonConfig(tilt_x_rad=...).
        if self.tilt_x_rad is not None:
            object.__setattr__(
                self,
                "mirror_2_azimuth_rad",
                self.mirror_1_azimuth_rad + float(self.tilt_x_rad),
            )
        if self.tilt_y_rad is not None:
            object.__setattr__(
                self,
                "mirror_2_cenital_rad",
                self.mirror_1_cenital_rad + float(self.tilt_y_rad),
            )

        if self.intensity_1 < 0 or self.intensity_2 < 0:
            raise ValueError("beam intensities must be non-negative")
        if self.intensity_1 + self.intensity_2 <= 0:
            raise ValueError("at least one beam intensity must be positive")
        if not 0 <= self.contrast <= 1:
            raise ValueError("contrast must be between 0 and 1")

        # Despues de resolver los alias, tilt_x_rad y tilt_y_rad quedan como
        # campos de lectura con la inclinacion relativa efectiva.
        object.__setattr__(self, "tilt_x_rad", self.relative_azimuth_rad)
        object.__setattr__(self, "tilt_y_rad", self.relative_cenital_rad)

    @property
    def relative_azimuth_rad(self) -> float:
        """Inclinacion azimutal relativa: espejo 2 menos espejo 1."""

        return self.mirror_2_azimuth_rad - self.mirror_1_azimuth_rad

    @property
    def relative_cenital_rad(self) -> float:
        """Inclinacion cenital relativa: espejo 2 menos espejo 1."""

        return self.mirror_2_cenital_rad - self.mirror_1_cenital_rad

    @property
    def tilt_magnitude_rad(self) -> float:
        """Magnitud de la inclinacion relativa entre los dos espejos."""

        return float(np.hypot(self.relative_azimuth_rad, self.relative_cenital_rad))

    @property
    def center_opd_m(self) -> float:
        """OPD en el centro de la camara, incluyendo desplazamiento del espejo."""

        return self.opd0_m + 2.0 * self.mirror_2_displacement_m

    def opd(self, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
        """Calcula la OPD para cada punto (x, y) de la camara."""

        return self.center_opd_m + 2.0 * (
            self.relative_azimuth_rad * x_m + self.relative_cenital_rad * y_m
        )


@dataclass(frozen=True)
class SimulationResult:
    """Arreglos principales que produce una simulacion de camara."""

    x_m: np.ndarray
    y_m: np.ndarray
    opd_m: np.ndarray
    intensity: np.ndarray
    visibility: np.ndarray


def monochromatic_spectrum(wavelength_nm: float = 632.8) -> Spectrum:
    """Crea una fuente ideal de una sola longitud de onda."""

    return Spectrum(
        wavelengths_m=np.array([wavelength_nm * 1e-9]),
        weights=np.array([1.0]),
        name=f"monochromatic {wavelength_nm:g} nm",
        center_wavelength_m=wavelength_nm * 1e-9,
        fwhm_m=0.0,
        shape="monochromatic",
    )


def gaussian_spectrum(
    center_nm: float = 632.8,
    fwhm_nm: float = 10.0,
    samples: int = 801,
    span_fwhm: float = 6.0,
) -> Spectrum:
    """Crea una fuente con espectro gaussiano.

    El arreglo discreto queda disponible para inspeccion, pero la coherencia de
    esta forma espectral se evalua con la formula analitica de banda angosta.
    """

    if fwhm_nm <= 0:
        raise ValueError("fwhm_nm must be positive for a Gaussian spectrum")
    if samples < 3:
        raise ValueError("samples must be at least 3")

    # Se muestrea una ventana centrada en lambda0 para poder guardar el espectro.
    half_span_nm = 0.5 * span_fwhm * fwhm_nm
    start_nm = max(1e-6, center_nm - half_span_nm)
    stop_nm = center_nm + half_span_nm
    wavelengths_nm = np.linspace(start_nm, stop_nm, samples)

    # Conversion entre FWHM y desviacion estandar de una gaussiana.
    sigma_nm = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    weights = np.exp(-0.5 * ((wavelengths_nm - center_nm) / sigma_nm) ** 2)

    return Spectrum(
        wavelengths_m=wavelengths_nm * 1e-9,
        weights=weights,
        name=f"Gaussian {center_nm:g} nm FWHM {fwhm_nm:g} nm",
        center_wavelength_m=center_nm * 1e-9,
        fwhm_m=fwhm_nm * 1e-9,
        shape="gaussian",
    )


def spectrum_from_csv(path: str | Path) -> Spectrum:
    """Carga un espectro desde CSV con columnas wavelength_nm, weight."""

    data = np.loadtxt(path, delimiter=",", comments="#", ndmin=2)
    if data.shape[1] < 2:
        raise ValueError("CSV spectrum must have at least two columns")

    wavelengths_nm = data[:, 0]
    weights = data[:, 1]
    center_m = float(np.sum(wavelengths_nm * weights) / np.sum(weights)) * 1e-9

    return Spectrum(
        wavelengths_m=wavelengths_nm * 1e-9,
        weights=weights,
        name=f"CSV spectrum {Path(path).name}",
        center_wavelength_m=center_m,
        fwhm_m=None,
    )


def intensity_from_opd(
    config: MichelsonConfig,
    spectrum: Spectrum,
    opd_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convierte OPD en intensidad normalizada y visibilidad."""

    # Intensidad media sin interferencia.
    base = config.intensity_1 + config.intensity_2

    # Amplitud maxima del termino interferente para dos haces de intensidades I1,I2.
    interference_scale = 2.0 * np.sqrt(config.intensity_1 * config.intensity_2)

    # La parte real da las oscilaciones; la magnitud da la envolvente.
    gamma = spectrum.complex_coherence(opd_m)
    fringe_term = np.real(gamma)
    intensity = base + config.contrast * interference_scale * fringe_term

    # Normalizar hace que dos haces iguales oscilen idealmente entre 0 y 2.
    normalized = intensity / base
    visibility = config.contrast * interference_scale * np.abs(gamma) / base
    return normalized, visibility


def simulate_camera(config: MichelsonConfig, spectrum: Spectrum) -> SimulationResult:
    """Calcula el patron completo sobre todos los pixeles de la camara."""

    x, y = config.camera.grid()
    opd = config.opd(x, y)
    intensity, visibility = intensity_from_opd(config, spectrum, opd)
    return SimulationResult(x_m=x, y_m=y, opd_m=opd, intensity=intensity, visibility=visibility)


def fringe_spacing_m(wavelength_m: float, tilt_x_rad: float, tilt_y_rad: float) -> float:
    """Periodo espacial de las franjas en la direccion perpendicular a ellas."""

    # Aqui tilt_x_rad y tilt_y_rad son componentes relativas entre los espejos.
    tilt = float(np.hypot(tilt_x_rad, tilt_y_rad))
    if tilt == 0:
        return np.inf
    return wavelength_m / (2.0 * tilt)


def theoretical_gaussian_coherence_length_fwhm_m(center_wavelength_m: float, fwhm_m: float) -> float:
    """Longitud de coherencia teorica FWHM para espectro gaussiano.

    Se reporta como ancho completo a media altura de la visibilidad en OPD:

        Lc = (4 ln 2 / pi) * lambda0^2 / Delta_lambda

    usando la aproximacion de banda angosta Delta_nu ~= c Delta_lambda/lambda0^2.
    """

    if fwhm_m <= 0:
        return np.inf
    return (4.0 * np.log(2.0) / np.pi) * (center_wavelength_m**2 / fwhm_m)


def fwhm_from_curve(x: np.ndarray, y: np.ndarray) -> float:
    """Estima el FWHM de una curva alrededor de su maximo."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1-D arrays of same length")
    if x.size < 3:
        return np.nan

    # Ordenar por x evita errores si el perfil se entrega invertido.
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    peak_idx = int(np.argmax(y))
    half = 0.5 * y[peak_idx]

    # Buscar el cruce izquierdo con media altura e interpolar entre muestras.
    left = np.nan
    for idx in range(peak_idx, 0, -1):
        if y[idx - 1] <= half <= y[idx] or y[idx] <= half <= y[idx - 1]:
            left = float(np.interp(half, [y[idx - 1], y[idx]], [x[idx - 1], x[idx]]))
            break

    # Buscar el cruce derecho con media altura e interpolar entre muestras.
    right = np.nan
    for idx in range(peak_idx, x.size - 1):
        if y[idx] >= half >= y[idx + 1] or y[idx] <= half <= y[idx + 1]:
            right = float(np.interp(half, [y[idx], y[idx + 1]], [x[idx], x[idx + 1]]))
            break

    if np.isnan(left) or np.isnan(right):
        return np.nan
    return right - left


def profile_along_fringe_normal(
    config: MichelsonConfig,
    spectrum: Spectrum,
    samples: int = 2500,
) -> dict[str, np.ndarray]:
    """Muestrea una linea perpendicular a las franjas claras/oscuras."""

    if samples < 2:
        raise ValueError("samples must be at least 2")

    # Las franjas son lineas de OPD constante. Su normal apunta en la direccion
    # del gradiente de OPD, o sea la inclinacion relativa entre espejos.
    tilt = config.tilt_magnitude_rad
    if tilt == 0:
        direction = np.array([1.0, 0.0])
    else:
        direction = np.array([config.relative_azimuth_rad, config.relative_cenital_rad]) / tilt

    # Se calcula la linea mas larga posible que cabe completa dentro del sensor.
    width = config.camera.visible_width_m
    height = config.camera.visible_height_m
    limits: list[float] = []
    if abs(direction[0]) > 1e-15:
        limits.append(0.5 * width / abs(direction[0]))
    if abs(direction[1]) > 1e-15:
        limits.append(0.5 * height / abs(direction[1]))
    half_length = min(limits) if limits else 0.5 * width

    # s es la coordenada sobre la linea del perfil. x,y son sus coordenadas de
    # camara correspondientes.
    s = np.linspace(-half_length, half_length, samples)
    x = s * direction[0]
    y = s * direction[1]

    opd = config.opd(x, y)
    intensity, visibility = intensity_from_opd(config, spectrum, opd)

    # Esta estimacion imita lo que se haria midiendo maximos/minimos locales.
    estimated_visibility = estimate_visibility_from_profile(
        opd,
        intensity,
        spectrum.representative_wavelength_m,
    )

    return {
        "s_m": s,
        "x_m": x,
        "y_m": y,
        "opd_m": opd,
        "intensity": intensity,
        "visibility": visibility,
        "estimated_visibility": estimated_visibility,
    }


def estimate_visibility_from_profile(
    opd_m: np.ndarray,
    intensity: np.ndarray,
    representative_wavelength_m: float,
    window_periods: float = 4.0,
) -> np.ndarray:
    """Estima visibilidad local con maximos y minimos en una ventana movil."""

    opd = np.asarray(opd_m, dtype=float)
    values = np.asarray(intensity, dtype=float)
    if opd.ndim != 1 or values.ndim != 1 or opd.size != values.size:
        raise ValueError("opd_m and intensity must be 1-D arrays of same length")
    if representative_wavelength_m <= 0 or opd.size < 3:
        return np.full(opd.shape, np.nan)

    # Ordenar por OPD simplifica barrer ventanas de tamano fijo.
    order = np.argsort(opd)
    sorted_opd = opd[order]
    sorted_values = values[order]

    # Una ventana de varios periodos evita estimar visibilidad con un solo punto.
    half_window = 0.5 * window_periods * representative_wavelength_m
    result = np.full(sorted_opd.shape, np.nan)

    left = 0
    right = 0
    for idx, center in enumerate(sorted_opd):
        while left < sorted_opd.size and sorted_opd[left] < center - half_window:
            left += 1
        while right < sorted_opd.size and sorted_opd[right] <= center + half_window:
            right += 1

        if right - left >= 3:
            segment = sorted_values[left:right]
            local_max = float(np.max(segment))
            local_min = float(np.min(segment))
            denom = local_max + local_min
            if denom > 0:
                result[idx] = (local_max - local_min) / denom

    # Regresar al orden original del perfil.
    unsorted = np.full(result.shape, np.nan)
    unsorted[order] = result
    return unsorted


def save_profile_csv(path: str | Path, profile: dict[str, np.ndarray]) -> None:
    """Guarda el perfil en CSV con unidades comodas para leer."""

    columns = [
        profile["s_m"] * 1e3,
        profile["x_m"] * 1e3,
        profile["y_m"] * 1e3,
        profile["opd_m"] * 1e6,
        profile["intensity"],
        profile["visibility"],
        profile["estimated_visibility"],
    ]
    header = "s_mm,x_mm,y_mm,opd_um,intensity,visibility,estimated_visibility"
    data = np.column_stack(columns)
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def save_npz(path: str | Path, result: SimulationResult, profile: dict[str, np.ndarray]) -> None:
    """Guarda arreglos completos de camara y perfil para analisis posterior."""

    np.savez_compressed(
        path,
        x_m=result.x_m,
        y_m=result.y_m,
        opd_m=result.opd_m,
        intensity=result.intensity,
        visibility=result.visibility,
        **{f"profile_{key}": value for key, value in profile.items()},
    )


def load_spectrum_from_rows(rows: Iterable[tuple[float, float]]) -> Spectrum:
    """Crea un Spectrum desde pares (wavelength_nm, weight)."""

    wavelengths_nm = []
    weights = []
    for wavelength_nm, weight in rows:
        wavelengths_nm.append(wavelength_nm)
        weights.append(weight)

    return Spectrum(
        wavelengths_m=np.asarray(wavelengths_nm, dtype=float) * 1e-9,
        weights=np.asarray(weights, dtype=float),
        name="custom rows",
    )
