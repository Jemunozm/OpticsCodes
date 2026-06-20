"""Funciones pequenas para guardar imagenes sin depender de matplotlib."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _normalize_to_uint8(values: np.ndarray, vmin: float | None = None, vmax: float | None = None) -> np.ndarray:
    """Convierte cualquier arreglo numerico en una imagen de 8 bits."""

    array = np.asarray(values, dtype=float)

    # Si no se especifican limites, se usa el rango real de los datos.
    if vmin is None:
        vmin = float(np.nanmin(array))
    if vmax is None:
        vmax = float(np.nanmax(array))

    # Si el arreglo es plano o invalido, se evita dividir por cero.
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        return np.zeros(array.shape, dtype=np.uint8)

    normalized = np.clip((array - vmin) / (vmax - vmin), 0.0, 1.0)
    return np.round(255.0 * normalized).astype(np.uint8)


def save_pattern_png(path: str | Path, intensity: np.ndarray) -> None:
    """Guarda el patron de intensidad como PNG en escala de grises."""

    image = Image.fromarray(_normalize_to_uint8(intensity))
    image.save(path)


def _profile_zoom_mask(x_values: np.ndarray, profile_zoom: float) -> np.ndarray:
    """Selecciona la region visible del perfil alrededor de s=0."""

    if profile_zoom <= 1:
        return np.ones(x_values.shape, dtype=bool)

    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    center = 0.5 * (x_min + x_max)
    half_range = 0.5 * (x_max - x_min) / profile_zoom
    return (x_values >= center - half_range) & (x_values <= center + half_range)


def save_profile_plot_png(
    path: str | Path,
    profile: dict[str, np.ndarray],
    width: int = 1000,
    height: int = 520,
    profile_zoom: float = 1.0,
) -> None:
    """Dibuja una grafica simple del perfil sin usar librerias pesadas."""

    # Margenes del area de grafica dentro del PNG.
    margin_left = 82
    margin_right = 28
    margin_top = 38
    margin_bottom = 62
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    # Pillow dibuja sobre un lienzo RGB.
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    # El perfil usa s en milimetros para que el eje sea legible.
    x_values = profile["s_m"] * 1e3
    intensity = np.asarray(profile["intensity"], dtype=float)
    visibility = np.asarray(profile["visibility"], dtype=float)
    mask = _profile_zoom_mask(x_values, profile_zoom)
    if np.count_nonzero(mask) < 2:
        mask = np.ones(x_values.shape, dtype=bool)
    x_values = x_values[mask]
    intensity = intensity[mask]
    visibility = visibility[mask]

    # Escalas de los ejes. La visibilidad siempre se dibuja entre 0 y 1.
    x_min = float(np.nanmin(x_values))
    x_max = float(np.nanmax(x_values))
    i_min = float(np.nanmin(intensity))
    i_max = float(np.nanmax(intensity))
    if i_max <= i_min:
        i_max = i_min + 1.0

    def x_to_px(x: np.ndarray) -> np.ndarray:
        """Convierte coordenada fisica s a pixel horizontal."""

        return margin_left + (x - x_min) / (x_max - x_min) * plot_w

    def y_to_px_intensity(y: np.ndarray) -> np.ndarray:
        """Convierte intensidad normalizada a pixel vertical."""

        return margin_top + (i_max - y) / (i_max - i_min) * plot_h

    def y_to_px_visibility(y: np.ndarray) -> np.ndarray:
        """Convierte visibilidad 0..1 a pixel vertical."""

        return margin_top + (1.0 - np.clip(y, 0.0, 1.0)) * plot_h

    # Cuadricula y marco de referencia.
    axis_color = (40, 40, 40)
    grid_color = (226, 226, 226)
    for frac in np.linspace(0.0, 1.0, 6):
        x = int(margin_left + frac * plot_w)
        y = int(margin_top + frac * plot_h)
        draw.line([(x, margin_top), (x, margin_top + plot_h)], fill=grid_color)
        draw.line([(margin_left, y), (margin_left + plot_w, y)], fill=grid_color)
    draw.rectangle(
        [margin_left, margin_top, margin_left + plot_w, margin_top + plot_h],
        outline=axis_color,
        width=1,
    )

    # Curva negra: intensidad medida. Curva roja: visibilidad/envolvente teorica.
    intensity_points = list(zip(x_to_px(x_values), y_to_px_intensity(intensity)))
    visibility_points = list(zip(x_to_px(x_values), y_to_px_visibility(visibility)))
    if len(intensity_points) > 1:
        draw.line(intensity_points, fill=(20, 20, 20), width=2)
        draw.line(visibility_points, fill=(200, 54, 54), width=2)

    # Etiquetas principales de la grafica.
    draw.text((margin_left, 12), "Perfil perpendicular a las franjas", fill=(20, 20, 20), font=font)
    draw.text((margin_left + 8, height - 36), "s sobre la linea [mm]", fill=(20, 20, 20), font=font)
    draw.text((10, margin_top), "intensidad", fill=(20, 20, 20), font=font)
    draw.text((width - 162, margin_top + 8), "visibilidad teorica", fill=(200, 54, 54), font=font)

    # Marcas numericas del eje horizontal.
    for frac in np.linspace(0.0, 1.0, 5):
        x_val = x_min + frac * (x_max - x_min)
        x_px = int(margin_left + frac * plot_w)
        draw.text((x_px - 22, margin_top + plot_h + 8), f"{x_val:.2f}", fill=(40, 40, 40), font=font)

    image.save(path)
