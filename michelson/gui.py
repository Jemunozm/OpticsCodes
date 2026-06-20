"""Interfaz grafica del simulador de interferometro de Michelson.

La GUI usa tkinter, que viene incluido con Python en Windows. El objetivo es
mantener una interfaz ligera: los controles modifican los parametros fisicos,
el nucleo de `simulation.py` calcula el patron, y esta ventana solo se encarga
de dibujar el resultado y guardar archivos si el usuario lo pide.
"""

from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from PIL import Image, ImageTk

from .render import _normalize_to_uint8, save_pattern_png, save_profile_plot_png
from .simulation import (
    Camera,
    MichelsonConfig,
    Spectrum,
    fwhm_from_curve,
    fringe_spacing_m,
    microdegrees_to_rad,
    monochromatic_spectrum,
    profile_along_fringe_normal,
    rad_to_microdegrees,
    rectangular_spectrum,
    save_npz,
    save_profile_csv,
    simulate_camera,
    spectrum_from_csv,
    theoretical_rectangular_coherence_length_fwhm_m,
)

DEFAULT_MIRROR_2_AZIMUTH_UDEG = float(rad_to_microdegrees(2500e-6))
DEFAULT_MIRROR_2_CENITAL_UDEG = float(rad_to_microdegrees(500e-6))


class MichelsonApp(tk.Tk):
    """Ventana principal de la interfaz grafica."""

    def __init__(self) -> None:
        super().__init__()

        self.title("Simulador Michelson")
        self.geometry("1240x820")
        self.minsize(1040, 680)

        # Referencias al ultimo resultado calculado. Se usan para redibujar y
        # guardar sin repetir trabajo innecesariamente.
        self.result = None
        self.profile = None
        self.config: MichelsonConfig | None = None
        self.spectrum: Spectrum | None = None

        # PhotoImage debe guardarse como atributo; si no, tkinter libera la
        # imagen y el canvas queda en blanco.
        self.pattern_photo: ImageTk.PhotoImage | None = None
        self._update_job: str | None = None
        self._is_updating = False

        self._create_variables()
        self._build_layout()
        self._connect_variable_traces()
        self.after(100, self.run_simulation)

    def _create_variables(self) -> None:
        """Crea las variables enlazadas a controles de tkinter."""

        # Fuente luminosa.
        self.source_var = tk.StringVar(value="rectangular")
        self.wavelength_nm_var = tk.DoubleVar(value=632.8)
        self.width_nm_var = tk.DoubleVar(value=20.0)
        self.spectrum_samples_var = tk.IntVar(value=801)
        self.spectrum_csv_var = tk.StringVar(value="")

        # Inclinaciones de los dos espejos en microgrados.
        self.m1_azimuth_var = tk.DoubleVar(value=0.0)
        self.m1_cenital_var = tk.DoubleVar(value=0.0)
        self.m2_azimuth_var = tk.DoubleVar(value=DEFAULT_MIRROR_2_AZIMUTH_UDEG)
        self.m2_cenital_var = tk.DoubleVar(value=DEFAULT_MIRROR_2_CENITAL_UDEG)

        # Diferencia fija de camino optico y desplazamiento del espejo 2.
        self.opd0_um_var = tk.DoubleVar(value=0.0)
        self.m2_displacement_um_var = tk.DoubleVar(value=0.0)

        # Camara y muestreo numerico.
        self.pixels_x_var = tk.IntVar(value=640)
        self.pixels_y_var = tk.IntVar(value=460)
        self.camera_width_mm_var = tk.DoubleVar(value=8.0)
        self.camera_height_mm_var = tk.DoubleVar(value=6.0)
        self.zoom_var = tk.DoubleVar(value=1.0)
        self.profile_zoom_var = tk.DoubleVar(value=1.0)
        self.profile_samples_var = tk.IntVar(value=1800)

        # Intensidades y contraste experimental.
        self.intensity_1_var = tk.DoubleVar(value=1.0)
        self.intensity_2_var = tk.DoubleVar(value=1.0)
        self.contrast_var = tk.DoubleVar(value=1.0)

        # Control de actualizacion automatica y texto de estado.
        self.auto_update_var = tk.BooleanVar(value=True)
        self.summary_var = tk.StringVar(value="Preparando simulacion...")
        self.status_var = tk.StringVar(value="Listo")

    def _build_layout(self) -> None:
        """Construye la distribucion general de la ventana."""

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        control_panel = ttk.Frame(self, padding=10)
        control_panel.grid(row=0, column=0, sticky="ns")

        display_panel = ttk.Frame(self, padding=(0, 10, 10, 10))
        display_panel.grid(row=0, column=1, sticky="nsew")
        display_panel.columnconfigure(0, weight=1)
        display_panel.rowconfigure(0, weight=3)
        display_panel.rowconfigure(1, weight=2)
        display_panel.rowconfigure(2, weight=0)

        self._build_controls(control_panel)
        self._build_display(display_panel)

    def _build_controls(self, parent: ttk.Frame) -> None:
        """Crea las pestanas de controles del lado izquierdo."""

        title = ttk.Label(parent, text="Michelson", font=("Segoe UI", 16, "bold"))
        title.pack(anchor="w", pady=(0, 8))

        button_row = ttk.Frame(parent)
        button_row.pack(fill="x", pady=(0, 8))
        ttk.Button(button_row, text="Simular", command=self.run_simulation).pack(side="left", fill="x", expand=True)
        ttk.Button(button_row, text="Guardar", command=self.save_outputs).pack(side="left", fill="x", expand=True, padx=(6, 0))

        ttk.Checkbutton(
            parent,
            text="Actualizar automaticamente",
            variable=self.auto_update_var,
            command=self.schedule_update,
        ).pack(anchor="w", pady=(0, 8))

        notebook = ttk.Notebook(parent)
        notebook.pack(fill="both", expand=True)

        source_tab = ttk.Frame(notebook, padding=8)
        mirrors_tab = ttk.Frame(notebook, padding=8)
        camera_tab = ttk.Frame(notebook, padding=8)
        beams_tab = ttk.Frame(notebook, padding=8)
        notebook.add(source_tab, text="Fuente")
        notebook.add(mirrors_tab, text="Espejos")
        notebook.add(camera_tab, text="Camara")
        notebook.add(beams_tab, text="Haces")

        self._build_source_controls(source_tab)
        self._build_mirror_controls(mirrors_tab)
        self._build_camera_controls(camera_tab)
        self._build_beam_controls(beams_tab)

        summary = ttk.Label(
            parent,
            textvariable=self.summary_var,
            justify="left",
            anchor="nw",
            padding=(0, 10, 0, 0),
            width=42,
        )
        summary.pack(fill="x", pady=(8, 0))

    def _build_source_controls(self, parent: ttk.Frame) -> None:
        """Controles asociados al espectro de la fuente."""

        self._combo(parent, "Tipo", self.source_var, ("rectangular", "mono", "csv"), row=0)
        self._entry(parent, "Lambda central [nm]", self.wavelength_nm_var, row=1)
        self._entry(parent, "Ancho espectral [nm]", self.width_nm_var, row=2)
        self._entry(parent, "Muestras espectro", self.spectrum_samples_var, row=3)

        csv_frame = ttk.Frame(parent)
        csv_frame.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        csv_frame.columnconfigure(0, weight=1)
        ttk.Entry(csv_frame, textvariable=self.spectrum_csv_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(csv_frame, text="CSV", command=self.choose_spectrum_csv).grid(row=0, column=1, padx=(6, 0))

    def _build_mirror_controls(self, parent: ttk.Frame) -> None:
        """Controles de inclinacion y desplazamiento de espejos."""

        ttk.Label(parent, text="Espejo 1", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w")
        self._slider(parent, "Azimutal [u°]", self.m1_azimuth_var, -300000, 300000, row=1)
        self._slider(parent, "Cenital [u°]", self.m1_cenital_var, -300000, 300000, row=2)

        ttk.Separator(parent).grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)

        ttk.Label(parent, text="Espejo 2", font=("Segoe UI", 10, "bold")).grid(row=4, column=0, columnspan=2, sticky="w")
        self._slider(parent, "Azimutal [u°]", self.m2_azimuth_var, -300000, 300000, row=5)
        self._slider(parent, "Cenital [u°]", self.m2_cenital_var, -300000, 300000, row=6)
        self._slider(parent, "Desplazamiento [um]", self.m2_displacement_um_var, -50, 50, row=7)
        self._slider(parent, "OPD fija [um]", self.opd0_um_var, -50, 50, row=8)

    def _build_camera_controls(self, parent: ttk.Frame) -> None:
        """Controles de campo de vision, resolucion y zoom."""

        self._entry(parent, "Pixeles X", self.pixels_x_var, row=0)
        self._entry(parent, "Pixeles Y", self.pixels_y_var, row=1)
        self._entry(parent, "Ancho [mm]", self.camera_width_mm_var, row=2)
        self._entry(parent, "Alto [mm]", self.camera_height_mm_var, row=3)
        self._slider(parent, "Zoom camara", self.zoom_var, 0.5, 8.0, row=4)
        self._slider(parent, "Zoom perfil", self.profile_zoom_var, 1.0, 20.0, row=5)
        self._entry(parent, "Puntos perfil", self.profile_samples_var, row=6)

    def _build_beam_controls(self, parent: ttk.Frame) -> None:
        """Controles de intensidades relativas y contraste adicional."""

        self._slider(parent, "Intensidad haz 1", self.intensity_1_var, 0.0, 2.0, row=0)
        self._slider(parent, "Intensidad haz 2", self.intensity_2_var, 0.0, 2.0, row=1)
        self._slider(parent, "Contraste", self.contrast_var, 0.0, 1.0, row=2)

    def _build_display(self, parent: ttk.Frame) -> None:
        """Crea los lienzos de patron, perfil y barra de estado."""

        pattern_frame = ttk.LabelFrame(parent, text="Camara")
        pattern_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        pattern_frame.rowconfigure(0, weight=1)
        pattern_frame.columnconfigure(0, weight=1)
        self.pattern_canvas = tk.Canvas(pattern_frame, background="#171717", highlightthickness=0)
        self.pattern_canvas.grid(row=0, column=0, sticky="nsew")
        self.pattern_canvas.bind("<Configure>", lambda _event: self.draw_pattern())

        profile_frame = ttk.LabelFrame(parent, text="Perfil perpendicular")
        profile_frame.grid(row=1, column=0, sticky="nsew")
        profile_frame.rowconfigure(0, weight=1)
        profile_frame.columnconfigure(0, weight=1)
        self.profile_canvas = tk.Canvas(profile_frame, background="white", highlightthickness=0)
        self.profile_canvas.grid(row=0, column=0, sticky="nsew")
        self.profile_canvas.bind("<Configure>", lambda _event: self.draw_profile())

        status = ttk.Label(parent, textvariable=self.status_var, anchor="w")
        status.grid(row=2, column=0, sticky="ew", pady=(6, 0))

    def _combo(self, parent: ttk.Frame, label: str, variable: tk.StringVar, values: tuple[str, ...], row: int) -> None:
        """Agrega un combobox con etiqueta."""

        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly", width=18)
        combo.grid(row=row, column=1, sticky="ew", pady=4)
        parent.columnconfigure(1, weight=1)

    def _entry(self, parent: ttk.Frame, label: str, variable: tk.Variable, row: int) -> None:
        """Agrega una entrada numerica con etiqueta."""

        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable, width=16).grid(row=row, column=1, sticky="ew", pady=4)
        parent.columnconfigure(1, weight=1)

    def _slider(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.DoubleVar,
        start: float,
        stop: float,
        row: int,
    ) -> None:
        """Agrega un slider con una caja numerica sincronizada."""

        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        row_frame = ttk.Frame(parent)
        row_frame.grid(row=row, column=1, sticky="ew", pady=5)
        row_frame.columnconfigure(0, weight=1)
        ttk.Scale(
            row_frame,
            variable=variable,
            from_=start,
            to=stop,
            orient="horizontal",
            command=lambda _value: self.schedule_update(),
        ).grid(row=0, column=0, sticky="ew")
        ttk.Entry(row_frame, textvariable=variable, width=9).grid(row=0, column=1, padx=(6, 0))
        parent.columnconfigure(1, weight=1)

    def _connect_variable_traces(self) -> None:
        """Programa actualizacion automatica cuando cambian los controles."""

        variables: list[tk.Variable] = [
            self.source_var,
            self.wavelength_nm_var,
            self.width_nm_var,
            self.spectrum_samples_var,
            self.spectrum_csv_var,
            self.m1_azimuth_var,
            self.m1_cenital_var,
            self.m2_azimuth_var,
            self.m2_cenital_var,
            self.m2_displacement_um_var,
            self.opd0_um_var,
            self.pixels_x_var,
            self.pixels_y_var,
            self.camera_width_mm_var,
            self.camera_height_mm_var,
            self.zoom_var,
            self.profile_zoom_var,
            self.profile_samples_var,
            self.intensity_1_var,
            self.intensity_2_var,
            self.contrast_var,
        ]
        for variable in variables:
            variable.trace_add("write", lambda *_args: self.schedule_update())

    def choose_spectrum_csv(self) -> None:
        """Abre un selector de archivo para espectros personalizados."""

        filename = filedialog.askopenfilename(
            title="Seleccionar espectro CSV",
            filetypes=(("CSV", "*.csv"), ("Texto", "*.txt"), ("Todos", "*.*")),
        )
        if filename:
            self.spectrum_csv_var.set(filename)
            self.source_var.set("csv")

    def schedule_update(self) -> None:
        """Agenda una simulacion, agrupando cambios rapidos de sliders."""

        if self._is_updating or not self.auto_update_var.get():
            return
        if self._update_job is not None:
            self.after_cancel(self._update_job)
        self._update_job = self.after(180, self.run_simulation)

    def build_spectrum(self) -> Spectrum:
        """Crea el espectro seleccionado en la GUI."""

        source = self.source_var.get()
        wavelength_nm = float(self.wavelength_nm_var.get())

        if source == "mono":
            return monochromatic_spectrum(wavelength_nm)
        if source == "rectangular":
            return rectangular_spectrum(
                center_nm=wavelength_nm,
                width_nm=float(self.width_nm_var.get()),
                samples=max(3, int(self.spectrum_samples_var.get())),
            )

        path = self.spectrum_csv_var.get().strip()
        if not path:
            raise ValueError("Selecciona un archivo CSV para la fuente.")
        return spectrum_from_csv(path)

    def build_config(self) -> MichelsonConfig:
        """Lee controles y crea la configuracion fisica."""

        camera = Camera(
            pixels_x=max(2, int(self.pixels_x_var.get())),
            pixels_y=max(2, int(self.pixels_y_var.get())),
            width_m=float(self.camera_width_mm_var.get()) * 1e-3,
            height_m=float(self.camera_height_mm_var.get()) * 1e-3,
            zoom=float(self.zoom_var.get()),
        )
        return MichelsonConfig(
            mirror_1_azimuth_rad=float(microdegrees_to_rad(self.m1_azimuth_var.get())),
            mirror_1_cenital_rad=float(microdegrees_to_rad(self.m1_cenital_var.get())),
            mirror_2_azimuth_rad=float(microdegrees_to_rad(self.m2_azimuth_var.get())),
            mirror_2_cenital_rad=float(microdegrees_to_rad(self.m2_cenital_var.get())),
            mirror_2_displacement_m=float(self.m2_displacement_um_var.get()) * 1e-6,
            opd0_m=float(self.opd0_um_var.get()) * 1e-6,
            intensity_1=float(self.intensity_1_var.get()),
            intensity_2=float(self.intensity_2_var.get()),
            contrast=float(self.contrast_var.get()),
            camera=camera,
        )

    def run_simulation(self) -> None:
        """Ejecuta el modelo y refresca patron, perfil y resumen."""

        self._update_job = None
        self._is_updating = True
        try:
            self.spectrum = self.build_spectrum()
            self.config = self.build_config()
            self.result = simulate_camera(self.config, self.spectrum)
            self.profile = profile_along_fringe_normal(
                self.config,
                self.spectrum,
                samples=max(2, int(self.profile_samples_var.get())),
            )
            self.draw_pattern()
            self.draw_profile()
            self.update_summary()
            self.status_var.set("Simulacion actualizada")
        except Exception as exc:
            self.status_var.set(f"Error: {exc}")
        finally:
            self._is_updating = False

    def draw_pattern(self) -> None:
        """Dibuja el patron de camara y la linea donde se toma el perfil."""

        canvas = self.pattern_canvas
        canvas.delete("all")
        if self.result is None or self.config is None:
            return

        canvas_w = max(1, canvas.winfo_width())
        canvas_h = max(1, canvas.winfo_height())

        image = Image.fromarray(_normalize_to_uint8(self.result.intensity))
        image_w, image_h = self._fit_size(image.width, image.height, canvas_w, canvas_h)
        image = image.resize((image_w, image_h), Image.Resampling.BILINEAR)
        self.pattern_photo = ImageTk.PhotoImage(image)

        left = (canvas_w - image_w) // 2
        top = (canvas_h - image_h) // 2
        canvas.create_image(left, top, image=self.pattern_photo, anchor="nw")

        # La linea cyan marca la direccion perpendicular a las franjas.
        self._draw_profile_line_on_pattern(canvas, left, top, image_w, image_h)

    def _draw_profile_line_on_pattern(
        self,
        canvas: tk.Canvas,
        left: int,
        top: int,
        image_w: int,
        image_h: int,
    ) -> None:
        """Superpone sobre el patron la linea usada para extraer el perfil."""

        if self.profile is None or self.config is None:
            return

        width = self.config.camera.visible_width_m
        height = self.config.camera.visible_height_m
        tilt = self.config.tilt_magnitude_rad
        if tilt == 0:
            direction = np.array([1.0, 0.0])
        else:
            direction = np.array([self.config.relative_azimuth_rad, self.config.relative_cenital_rad]) / tilt

        # El perfil completo puede cubrir mas campo que la imagen cuando hay
        # zoom de camara. Aqui solo se dibuja el tramo visible de esa misma
        # recta sobre el recorte mostrado.
        limits: list[float] = []
        if abs(direction[0]) > 1e-15:
            limits.append(0.5 * width / abs(direction[0]))
        if abs(direction[1]) > 1e-15:
            limits.append(0.5 * height / abs(direction[1]))
        half_length = min(limits) if limits else 0.5 * width

        x0_m = -half_length * direction[0]
        y0_m = -half_length * direction[1]
        x1_m = half_length * direction[0]
        y1_m = half_length * direction[1]

        x0 = left + (x0_m + 0.5 * width) / width * image_w
        y0 = top + (0.5 * height - y0_m) / height * image_h
        x1 = left + (x1_m + 0.5 * width) / width * image_w
        y1 = top + (0.5 * height - y1_m) / height * image_h
        canvas.create_line(x0, y0, x1, y1, fill="#48d7ff", width=2)

    def draw_profile(self) -> None:
        """Dibuja intensidad y visibilidad sobre la linea perpendicular."""

        canvas = self.profile_canvas
        canvas.delete("all")
        if self.profile is None:
            return

        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        margin_left = 70
        margin_right = 28
        margin_top = 24
        margin_bottom = 46
        plot_w = max(1, width - margin_left - margin_right)
        plot_h = max(1, height - margin_top - margin_bottom)

        x_values = self.profile["s_m"] * 1e3
        intensity = np.asarray(self.profile["intensity"], dtype=float)
        visibility = np.asarray(self.profile["visibility"], dtype=float)
        estimated = np.asarray(self.profile["estimated_visibility"], dtype=float)

        profile_zoom = max(1.0, float(self.profile_zoom_var.get()))
        if profile_zoom > 1:
            full_min = float(np.nanmin(x_values))
            full_max = float(np.nanmax(x_values))
            center = 0.5 * (full_min + full_max)
            half_range = 0.5 * (full_max - full_min) / profile_zoom
            mask = (x_values >= center - half_range) & (x_values <= center + half_range)
            if np.any(mask):
                x_values = x_values[mask]
                intensity = intensity[mask]
                visibility = visibility[mask]
                estimated = estimated[mask]

        x_min = float(np.nanmin(x_values))
        x_max = float(np.nanmax(x_values))
        i_min = float(np.nanmin(intensity))
        i_max = float(np.nanmax(intensity))
        if not np.isfinite(i_min) or not np.isfinite(i_max) or i_max <= i_min:
            i_min, i_max = 0.0, 1.0

        def x_to_px(x: np.ndarray) -> np.ndarray:
            return margin_left + (x - x_min) / (x_max - x_min) * plot_w

        def y_to_px_intensity(y: np.ndarray) -> np.ndarray:
            return margin_top + (i_max - y) / (i_max - i_min) * plot_h

        def y_to_px_visibility(y: np.ndarray) -> np.ndarray:
            return margin_top + (1.0 - np.clip(y, 0.0, 1.0)) * plot_h

        # Cuadricula.
        for frac in np.linspace(0.0, 1.0, 6):
            x = margin_left + frac * plot_w
            y = margin_top + frac * plot_h
            canvas.create_line(x, margin_top, x, margin_top + plot_h, fill="#e5e5e5")
            canvas.create_line(margin_left, y, margin_left + plot_w, y, fill="#e5e5e5")
        canvas.create_rectangle(
            margin_left,
            margin_top,
            margin_left + plot_w,
            margin_top + plot_h,
            outline="#303030",
        )

        self._create_curve(canvas, x_to_px(x_values), y_to_px_intensity(intensity), "#1f1f1f", 2)
        self._create_curve(canvas, x_to_px(x_values), y_to_px_visibility(visibility), "#c83737", 2)

        # La visibilidad estimada puede tener NaN en los bordes; se dibuja solo
        # donde hay datos validos.
        valid = np.isfinite(estimated)
        if np.any(valid):
            self._create_curve(canvas, x_to_px(x_values[valid]), y_to_px_visibility(estimated[valid]), "#2f6fc0", 1)

        canvas.create_text(margin_left, 12, text="Intensidad", anchor="w", fill="#1f1f1f")
        canvas.create_text(width - 210, 12, text="Visibilidad teorica", anchor="w", fill="#c83737")
        canvas.create_text(width - 210, 30, text="Visibilidad estimada", anchor="w", fill="#2f6fc0")
        canvas.create_text(margin_left + plot_w / 2, height - 18, text="s sobre la linea [mm]", anchor="center")

        for frac in np.linspace(0.0, 1.0, 5):
            x_val = x_min + frac * (x_max - x_min)
            x_px = margin_left + frac * plot_w
            canvas.create_text(x_px, margin_top + plot_h + 14, text=f"{x_val:.2f}", anchor="n", fill="#303030")

    def _create_curve(
        self,
        canvas: tk.Canvas,
        x_pixels: np.ndarray,
        y_pixels: np.ndarray,
        color: str,
        width: int,
    ) -> None:
        """Dibuja una curva en el canvas a partir de arreglos de pixeles."""

        if x_pixels.size < 2:
            return

        # Reducir puntos si hay demasiados evita que tkinter se vuelva lento.
        max_points = 2200
        step = max(1, int(math.ceil(x_pixels.size / max_points)))
        points = np.column_stack((x_pixels[::step], y_pixels[::step])).ravel().tolist()
        if len(points) >= 4:
            canvas.create_line(*points, fill=color, width=width)

    def update_summary(self) -> None:
        """Actualiza el bloque numerico del panel izquierdo."""

        if self.config is None or self.spectrum is None or self.profile is None:
            return

        lambda0 = self.spectrum.representative_wavelength_m
        spacing = fringe_spacing_m(lambda0, self.config.relative_azimuth_rad, self.config.relative_cenital_rad)
        numeric_lc = fwhm_from_curve(self.profile["opd_m"], self.profile["visibility"])

        lines = [
            f"Tilt relativo az: {float(rad_to_microdegrees(self.config.relative_azimuth_rad)):.3g} u°",
            f"Tilt relativo cen: {float(rad_to_microdegrees(self.config.relative_cenital_rad)):.3g} u°",
            f"Tilt relativo total: {float(rad_to_microdegrees(self.config.tilt_magnitude_rad)):.3g} u°",
            f"OPD central: {self.config.center_opd_m * 1e6:.3g} um",
            f"Separacion franjas: {spacing * 1e3:.3g} mm",
            f"Zoom perfil: {float(self.profile_zoom_var.get()):.3g}",
        ]

        if np.isfinite(numeric_lc):
            lines.append(f"Lc simulada: {numeric_lc * 1e6:.3g} um OPD")
        else:
            lines.append("Lc simulada: fuera del campo")

        if self.spectrum.shape == "rectangular":
            width_m = float(self.spectrum.wavelengths_m[-1] - self.spectrum.wavelengths_m[0])
            theoretical = theoretical_rectangular_coherence_length_fwhm_m(lambda0, width_m)
            lines.append(f"Lc teorica rect.: {theoretical * 1e6:.3g} um OPD")

        self.summary_var.set("\n".join(lines))

    def save_outputs(self) -> None:
        """Guarda PNG, CSV, NPZ y resumen en la carpeta elegida."""

        if self.result is None or self.profile is None or self.config is None or self.spectrum is None:
            messagebox.showwarning("Sin datos", "Primero ejecuta una simulacion.")
            return

        directory = filedialog.askdirectory(title="Carpeta de salida")
        if not directory:
            return

        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_pattern_png(output_dir / "pattern.png", self.result.intensity)
        save_profile_csv(output_dir / "profile.csv", self.profile)
        save_profile_plot_png(output_dir / "profile.png", self.profile, profile_zoom=float(self.profile_zoom_var.get()))
        save_npz(output_dir / "simulation.npz", self.result, self.profile)
        (output_dir / "summary_gui.txt").write_text(self.summary_var.get(), encoding="utf-8")
        self.status_var.set(f"Resultados guardados en {output_dir}")

    @staticmethod
    def _fit_size(source_w: int, source_h: int, target_w: int, target_h: int) -> tuple[int, int]:
        """Calcula un tamano que cabe en el canvas conservando proporcion."""

        if source_w <= 0 or source_h <= 0:
            return 1, 1
        scale = min(target_w / source_w, target_h / source_h)
        scale = max(scale, 1e-6)
        return max(1, int(source_w * scale)), max(1, int(source_h * scale))


def main() -> None:
    """Arranca la aplicacion grafica."""

    app = MichelsonApp()
    app.mainloop()


if __name__ == "__main__":
    main()
