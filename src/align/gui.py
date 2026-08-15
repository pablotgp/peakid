"""Ventana tkinter de alineamiento manual: foto + silueta calculada encima.

tkinter es stdlib (cero dependencias nuevas) y mantiene TODA la matemática
en Python: la alternativa web habría duplicado la proyección en JS, dos
copias de matemática sensible a convenios. Capa fina: la proyección y la
tira viven en src/align y están testeadas; esta ventana solo dibuja y recoge
eventos. NO hay detección automática de cresta: segmentar el skyline es el
objetivo del modelo de la fase 2, no de heurísticas — el juicio del encaje
lo hace el usuario mirando la tira.

Verificación: la tira y el canvas se pueden volcar a disco con la tecla D
(mismo array que se dibuja, no una réplica); el resto, manual.

Controles:
    arrastrar arriba marcar el sector del panorama que cubre la foto
    arrastrar foto   acotar el ajuste a la zona de cresta limpia
                     (clic suelto para quitar el recuadro)
    rueda            zoom sobre el cursor
    botón central    desplazar (también espacio + arrastrar con el izquierdo)
    0                volver al encuadre completo
    flechas          azimut / inclinación (fino; con Shift, grueso)
    Q / E            giro
    + / -            campo de visión
    S                buscar (en el sector marcado, o los 360° si no hay)
    1 .. 5           saltar entre los candidatos de la última búsqueda
    L                mostrar / ocultar los topónimos
    D                volcar tira + canvas + estado a ficheros de depuración
    Ctrl+S           guardar JSON y cerrar
    Escape           salir sin guardar
"""

import math
import tkinter as tk
from pathlib import Path
from tkinter import font as tkfont

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from src.align import (
    AlignmentParams,
    alignment_json_path,
    build_strip,
    load_oriented_photo,
    project_profile,
    save_alignment,
    seed_entry,
)
from src.align.search import (
    detect_photo_skyline,
    search_alignment,
    sector_to_ranges,
    solve_pitch_roll,
)
from src.horizon import HorizonProfile, Visibility

_LINE_OK = "#ff3b30"      # silueta fiable: rojo, contraste alto sobre foto
_LINE_TRUNC = "#ff9500"   # sector truncado: ámbar y discontinuo, como render/
_STRIP_H = 200            # alto de la tira; con ±3° da un estirado de ~4-5x
_BAND_DEG = 3.0           # semiancho vertical de la tira, en grados
_ZOOM_STEP = 1.25
_ZOOM_MAX = 12.0
_SECTOR_H = 110           # alto del panel de sector (perfil de 360°)
_SECTOR_FILL = "#3d4a63"  # misma silueta que render/
_SECTOR_MARK = "#ff9500"
_LABEL_OK = "#ffffff"     # topónimo de pico visible
_LABEL_UNKNOWN = "#ffc46b"  # sin confirmar, como el ámbar de render/
_ROI_COLOR = "#00e5c0"    # recuadro que acota el ajuste
_MIN_SKYLINE_COLUMNS = 50  # por debajo, el modo automático no se sostiene


def run_align(photo_path: str, profile: HorizonProfile, lat_deg: float,
              lon_deg: float, eye_m: float, seed_params: AlignmentParams,
              seed_sources: dict[str, str], observer_source: str,
              resume: dict | None = None, peaks: list | None = None,
              ) -> str | None:
    """Abre la ventana; devuelve la ruta del JSON si se guardó, None si no.

    `resume` es la sesión anterior (el dict de load_session) si existe: los
    parámetros iniciales ya vienen de ella vía seed_params, y su bloque
    `seed` original se conserva TAL CUAL al reguardar — la procedencia EXIF
    de la primera vez es la medida del error de brújula y una recarga no
    debe sobrescribirla.
    """
    app = _AlignApp(photo_path, profile, lat_deg, lon_deg, eye_m,
                    seed_params, seed_sources, observer_source, resume,
                    peaks)
    app.root.mainloop()
    return app.saved_path


class _AlignApp:
    def __init__(self, photo_path, profile, lat_deg, lon_deg, eye_m,
                 seed_params, seed_sources, observer_source, resume=None,
                 peaks=None):
        self.photo_path = photo_path
        self.profile = profile
        self.lat_deg, self.lon_deg, self.eye_m = lat_deg, lon_deg, eye_m
        self.seed_params = seed_params
        self.seed_sources = seed_sources
        self.observer_source = observer_source
        self.resume = resume
        self.params = AlignmentParams(**vars(seed_params).copy())
        self.saved_path: str | None = None

        # ÚNICO origen de píxeles, ya orientado: el canvas principal usa
        # self.img y la tira self.photo_np, ambos de la misma llamada
        self.img, self.photo_np = load_oriented_photo(photo_path)
        self.full_w, self.full_h = self.img.size

        # cresta de la foto: se detecta UNA vez y alimenta el modo automático
        # de pitch/roll. Es la heurística asistente de search.py, no el
        # detector engañoso que se quitó en su día: aquí no muestra un número
        # de calidad, resuelve dos parámetros que el usuario puede congelar.
        cols, rows, valid = detect_photo_skyline(self.photo_np)
        self.skyline_cols = cols[valid]
        self.skyline_rows = rows[valid]
        self.has_skyline = self.skyline_cols.size >= _MIN_SKYLINE_COLUMNS

        # Topónimos: la validación que de verdad zanja un alineamiento. Ver
        # qué NOMBRE y qué ALTITUD caen sobre cada bulto distingue hipótesis
        # que el encaje de la línea no separa — medido: tres candidatos a 25°
        # unos de otros con errores en píxeles decrecientes, y el de MENOR
        # error ponía un cerro de 708 m sobre el macizo dominante y dejaba
        # fuera de cuadro una cima de 2069 m que la foto muestra clarísima.
        self.peaks = list(peaks or [])
        self.peak_az = np.array([p.azimuth_deg for p in self.peaks], dtype=float)
        self.peak_el = np.array([p.elevation_deg for p in self.peaks],
                                dtype=float)
        self.show_labels = True
        self.candidates: list = []      # última búsqueda, para saltar con 1..5
        self._search_note = ""
        # recuadro (en píxeles de la FOTO) al que se restringe el ajuste;
        # None = toda la cresta detectada
        self.roi: tuple[float, float, float, float] | None = None

        self.sector = None  # (centro, ancho) del arco marcado; None = sin marcar
        # Guard de reentrada: Scale.set() DISPARA su command, así que sin esto
        # _solve_auto -> set() -> _from_slider -> _redraw -> _solve_auto sería
        # recursión infinita. Arranca en True para que la construcción de los
        # sliders tampoco dibuje antes de que existan los lienzos.
        self._updating = True

        self.root = tk.Tk()
        self.root.title(f"peakid — alineamiento: {Path(photo_path).name}")
        # fuente medible: hace falta para truncar los topónimos al ancho de
        # la banda en vez de dejar que se salgan del lienzo
        self._label_font = tkfont.Font(family="TkDefaultFont", size=8)
        self._build_controls()
        if resume is not None and resume.get("notes"):
            self.notes.insert(0, resume["notes"])

        # la foto se escala a lo que sobra en pantalla DESPUÉS de reservar
        # controles, tira y panel de sector
        self.root.update_idletasks()
        reserved = (self.controls.winfo_reqheight() + _STRIP_H + _SECTOR_H
                    + 160)
        max_w = max(self.root.winfo_screenwidth() - 80, 320)
        max_h = max(self.root.winfo_screenheight() - reserved, 240)
        self.fit_scale = min(max_w / self.full_w, max_h / self.full_h, 1.0)
        self.cw = int(self.full_w * self.fit_scale)
        self.ch = int(self.full_h * self.fit_scale)

        self.sector_canvas = tk.Canvas(self.root, width=self.cw,
                                       height=_SECTOR_H, highlightthickness=0,
                                       bg="#ffffff")
        self.sector_canvas.pack(side="top", pady=(0, 2))
        self.strip = tk.Canvas(self.root, width=self.cw, height=_STRIP_H,
                               highlightthickness=0, bg="#101010")
        self.strip.pack(side="bottom", pady=(2, 0))
        self.canvas = tk.Canvas(self.root, width=self.cw, height=self.ch,
                                highlightthickness=0, bg="#101010")
        self.canvas.pack(side="top")

        self._reset_view()
        self._bind_events()
        self._draw_sector_panel()
        if not self.has_skyline:
            for var in self.auto_vars.values():
                var.set(False)
        self._sync_auto_state()
        self._updating = False
        self._redraw()

    # --- construcción de la interfaz ---------------------------------

    def _build_controls(self) -> None:
        # LOS CONTROLES SE EMPAQUETAN PRIMERO, con side="bottom": en tkinter
        # el orden de pack decide quién se queda sin sitio, y una foto
        # vertical hacía una ventana más alta que la pantalla que cortaba por
        # abajo la mitad de los sliders.
        self.controls = tk.Frame(self.root)
        self.controls.pack(side="bottom", fill="x", padx=6, pady=4)
        self.sliders = {}
        self.slider_labels = {}
        self.auto_vars = {}
        self._slider_len = 330
        # dos por fila: cuatro apilados gastaban el doble de alto útil
        for i, (name, lo, hi, res) in enumerate((
                ("azimuth_deg", 0.0, 360.0, 0.1),
                ("pitch_deg", -25.0, 25.0, 0.05),
                ("hfov_deg", 20.0, 120.0, 0.1),
                ("roll_deg", -20.0, 20.0, 0.05))):
            row, col = divmod(i, 2)
            label = tk.Label(self.controls, text=name, width=17, anchor="w")
            label.grid(row=row, column=col * 3, sticky="w")
            self.slider_labels[name] = label
            slider = tk.Scale(self.controls, from_=lo, to=hi, resolution=res,
                              orient="horizontal", length=self._slider_len,
                              showvalue=True,
                              command=lambda _v, n=name: self._from_slider(n))
            slider.set(getattr(self.params, name))
            slider.grid(row=row, column=col * 3 + 1, sticky="we")
            self.sliders[name] = slider
            # pitch y roll se resuelven solos; sus casillas permiten
            # congelarlos para retoque manual
            if name in ("pitch_deg", "roll_deg"):
                var = tk.BooleanVar(value=True)
                self.auto_vars[name] = var
                tk.Checkbutton(self.controls, text="auto", variable=var,
                               command=self._on_auto_toggle).grid(
                    row=row, column=col * 3 + 2, sticky="w", padx=(2, 12))
            else:
                tk.Label(self.controls, text="").grid(row=row,
                                                      column=col * 3 + 2,
                                                      padx=(2, 12))
        self._refresh_slider_labels()

        tk.Label(self.controls, text="notes", width=17, anchor="w").grid(
            row=2, column=0, sticky="w")
        self.notes = tk.Entry(self.controls)
        self.notes.grid(row=2, column=1, columnspan=5, sticky="we")
        # Lectura del ajuste automático en su PROPIA etiqueta, no en la barra
        # de estado: se recalcula en cada redibujado y se comía los mensajes
        # de la búsqueda y de las teclas de candidato, que son respuestas a
        # una acción del usuario y no deben desaparecer solas.
        self.auto_label = tk.Label(self.controls, anchor="w", fg="#2e7d32")
        self.auto_label.grid(row=3, column=0, columnspan=6, sticky="we")
        self.status = tk.Label(self.controls, anchor="w", fg="#555555",
                               text="arrastra arriba = sector  S=buscar  "
                                    "1..5=candidatos  L=topónimos  "
                                    "rueda=zoom  0=encuadre  Ctrl+S=guardar")
        self.status.grid(row=4, column=0, columnspan=6, sticky="we")

    def _refresh_slider_labels(self) -> None:
        """La etiqueta lleva la resolución conseguida: es el dato que motivó
        acotar por sector (360° en ~330 px son 1.1°/px, inservible para
        ajustar décimas; un sector de 30° baja a 0.09°/px)."""
        for name, slider in self.sliders.items():
            span = abs(float(slider.cget("to")) - float(slider.cget("from")))
            self.slider_labels[name].config(
                text=f"{name}  {span / self._slider_len:.2f}°/px")

    def _on_auto_toggle(self) -> None:
        self._sync_auto_state()
        self._redraw()

    def _sync_auto_state(self) -> None:
        """Un slider en automático se deshabilita: su valor lo pone el
        solver, y dejarlo arrastrable invitaría a pelearse con él."""
        for name, var in self.auto_vars.items():
            enabled = not var.get() or not self.has_skyline
            self.sliders[name].config(state="normal" if enabled else "disabled")

    def _bind_events(self) -> None:
        for key, attr, step in (("<Left>", "azimuth_deg", -0.1),
                                ("<Right>", "azimuth_deg", 0.1),
                                ("<Shift-Left>", "azimuth_deg", -2.0),
                                ("<Shift-Right>", "azimuth_deg", 2.0),
                                ("<Up>", "pitch_deg", 0.05),
                                ("<Down>", "pitch_deg", -0.05),
                                ("<Shift-Up>", "pitch_deg", 1.0),
                                ("<Shift-Down>", "pitch_deg", -1.0),
                                ("<q>", "roll_deg", -0.05),
                                ("<e>", "roll_deg", 0.05),
                                ("<plus>", "hfov_deg", 0.5),
                                ("<KP_Add>", "hfov_deg", 0.5),
                                ("<minus>", "hfov_deg", -0.5),
                                ("<KP_Subtract>", "hfov_deg", -0.5)):
            self.root.bind(key,
                           lambda e, a=attr, st=step: self._nudge(e, a, st))
        self.root.bind("<Control-s>", lambda _e: self._save())
        self.root.bind("<Escape>", lambda _e: self.root.destroy())
        self.root.bind("<d>", self._dump_debug)
        # <Control-s> es más específico que <s>: Tk resuelve por
        # especificidad, así que guardar no dispara además la búsqueda
        self.root.bind("<s>", self._run_search)
        self.root.bind("<l>", self._toggle_labels)
        # OJO: en Tk un detalle NUMÉRICO en un binding es el número de BOTÓN
        # del ratón, no una tecla. bind("<1>") registra <Button-1>, así que
        # los dígitos no hacían nada Y, peor, cada clic en la foto aplicaba
        # el candidato 1 y se llevaba por delante el ajuste manual en curso.
        # Con dígitos hay que escribir <Key-N> siempre.
        for n in range(1, 6):   # 1..5: saltar entre candidatos de la búsqueda
            self.root.bind(f"<Key-{n}>",
                           lambda e, i=n - 1: self._apply_candidate(i, e))
        self.sector_canvas.bind("<Button-1>", self._sector_press)
        self.sector_canvas.bind("<B1-Motion>", self._sector_drag)
        self.sector_canvas.bind("<ButtonRelease-1>", self._sector_drag)
        # <Key-0> y no <0>: hoy <0> funcionaría por accidente (0 no es un
        # número de botón válido, así que Tk lo trata como tecla), y depender
        # de eso es justo lo que se acaba rompiendo solo
        self.root.bind("<Key-0>",
                       lambda e: self._typing(e) or self._reset_zoom())

        # zoom: rueda de Windows/macOS (<MouseWheel>) y de X11 (Button-4/5)
        for widget in (self.canvas, self.strip):
            widget.bind("<MouseWheel>",
                        lambda e: self._zoom_at(e.x, e.y,
                                                _ZOOM_STEP if e.delta > 0
                                                else 1 / _ZOOM_STEP))
            widget.bind("<Button-4>",
                        lambda e: self._zoom_at(e.x, e.y, _ZOOM_STEP))
            widget.bind("<Button-5>",
                        lambda e: self._zoom_at(e.x, e.y, 1 / _ZOOM_STEP))
        self.canvas.bind("<Button-2>", self._pan_start)
        self.canvas.bind("<B2-Motion>", self._pan_move)
        # el botón izquierdo SIN espacio dibuja el recuadro de ajuste: hasta
        # ahora no hacía nada (desplazar pide espacio o botón central)
        self.canvas.bind("<Button-1>", self._roi_press)
        self.canvas.bind("<B1-Motion>", self._roi_drag)
        self.canvas.bind("<ButtonRelease-1>", self._roi_release)
        self.root.bind("<KeyPress-space>", self._space_down)
        self.root.bind("<KeyRelease-space>", self._space_up)
        self.space_held = False

    # --- vista (zoom y desplazamiento) --------------------------------

    def _reset_view(self) -> None:
        self.view_scale = self.fit_scale
        self.view_x = 0.0
        self.view_y = 0.0
        self._photo_cache_key = None

    def _reset_zoom(self) -> None:
        self._reset_view()
        self._redraw()

    def _clamp_view(self) -> None:
        span_x, span_y = self.cw / self.view_scale, self.ch / self.view_scale
        self.view_x = (0.0 if span_x >= self.full_w
                       else min(max(self.view_x, 0.0), self.full_w - span_x))
        self.view_y = (0.0 if span_y >= self.full_h
                       else min(max(self.view_y, 0.0), self.full_h - span_y))

    def _zoom_at(self, cx: int, cy: int, factor: float) -> None:
        """Zoom manteniendo fijo el punto de la foto bajo el cursor."""
        new_scale = min(max(self.view_scale * factor, self.fit_scale),
                        _ZOOM_MAX)
        if new_scale == self.view_scale:
            return
        px = self.view_x + cx / self.view_scale
        py = self.view_y + cy / self.view_scale
        self.view_scale = new_scale
        self.view_x = px - cx / new_scale
        self.view_y = py - cy / new_scale
        self._clamp_view()
        self._redraw()

    def _pan_start(self, event) -> None:
        if event.num == 1 and not self.space_held:
            return  # el botón izquierdo solo arrastra con espacio pulsado
        self._pan_anchor = (event.x, event.y, self.view_x, self.view_y)

    def _pan_move(self, event) -> None:
        anchor = getattr(self, "_pan_anchor", None)
        if anchor is None:
            return
        x0, y0, vx0, vy0 = anchor
        self.view_x = vx0 - (event.x - x0) / self.view_scale
        self.view_y = vy0 - (event.y - y0) / self.view_scale
        self._clamp_view()
        self._redraw()

    # --- recuadro de ajuste -------------------------------------------

    def _roi_press(self, event) -> None:
        if self.space_held:              # con espacio el izquierdo desplaza
            self._pan_start(event)
            return
        self._roi_anchor = (event.x, event.y)

    def _roi_drag(self, event) -> None:
        if self.space_held:
            self._pan_move(event)
            return
        anchor = getattr(self, "_roi_anchor", None)
        if anchor is None:
            return
        self.canvas.delete("roi")
        self.canvas.create_rectangle(anchor[0], anchor[1], event.x, event.y,
                                     outline=_ROI_COLOR, width=2, dash=(5, 3),
                                     tags="roi")

    def _roi_release(self, event) -> None:
        anchor = getattr(self, "_roi_anchor", None)
        self._roi_anchor = None
        if self.space_held or anchor is None:
            return
        if abs(event.x - anchor[0]) < 8 or abs(event.y - anchor[1]) < 8:
            # un clic suelto (o casi) LIMPIA el recuadro: es la forma
            # evidente de deshacerlo sin buscar una tecla
            self.roi = None
            self.canvas.delete("roi")
            self.status.config(text="recuadro quitado: se usa toda la cresta "
                                    "detectada")
            self._redraw()
            return
        # a píxeles de la FOTO: el recuadro debe sobrevivir al zoom
        x0 = self.view_x + min(anchor[0], event.x) / self.view_scale
        x1 = self.view_x + max(anchor[0], event.x) / self.view_scale
        y0 = self.view_y + min(anchor[1], event.y) / self.view_scale
        y1 = self.view_y + max(anchor[1], event.y) / self.view_scale
        self.roi = (x0, y0, x1, y1)
        used = self._fit_columns()[0].size
        self.status.config(
            text=f"ajuste restringido al recuadro: {used} columnas de cresta "
                 f"(clic suelto para quitarlo)")
        self._redraw()

    def _fit_columns(self):
        """Columnas de cresta que alimentan el ajuste, aplicando el recuadro.

        Con obstáculos anchos —una valla que cruza media foto— ninguna
        heurística puede separar la cresta buena de la mala; acotar a mano la
        zona limpia sí, y con media foto despejada suele bastar.
        """
        cols, rows = self.skyline_cols, self.skyline_rows
        if self.roi is None:
            return cols, rows
        x0, y0, x1, y1 = self.roi
        inside = (cols >= x0) & (cols <= x1) & (rows >= y0) & (rows <= y1)
        if inside.sum() < 20:   # recuadro inservible: mejor ignorarlo
            return cols, rows
        return cols[inside], rows[inside]

    def _draw_roi(self) -> None:
        self.canvas.delete("roi")
        if self.roi is None:
            return
        x0, y0 = self._to_canvas(self.roi[0], self.roi[1])
        x1, y1 = self._to_canvas(self.roi[2], self.roi[3])
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=_ROI_COLOR,
                                     width=2, dash=(5, 3), tags="roi")

    def _space_down(self, event) -> None:
        if not self._typing(event):
            self.space_held = True

    def _space_up(self, _event) -> None:
        self.space_held = False
        self._pan_anchor = None

    # --- parámetros ---------------------------------------------------

    def _typing(self, _event) -> bool:
        """Las teclas de ajuste no deben escribirse en el campo notes."""
        return self.root.focus_get() is self.notes

    def _from_slider(self, name: str) -> None:
        if self._updating:   # el valor lo estamos poniendo nosotros
            return
        value = float(self.sliders[name].get())
        # Segunda defensa, y la que de verdad hace falta: Tk puede ENCOLAR el
        # callback del Scale, así que llega cuando el guard ya se apagó. Si el
        # valor coincide con el que ya tenemos, es el eco de un set() nuestro y
        # no un arrastre del usuario. Para que la comparación funcione,
        # _apply_value guarda siempre el valor LEÍDO del slider (redondeado a
        # su resolución), no el que se le pidió.
        if value == getattr(self.params, name):
            return
        setattr(self.params, name, value)
        self._redraw()

    def _apply_value(self, name: str, value: float) -> None:
        """Pone un valor en el slider sin que rebote como si fuera del usuario."""
        slider = self.sliders[name]
        was_disabled = str(slider.cget("state")) == "disabled"
        previous, self._updating = self._updating, True
        try:
            clamped = min(max(value, float(slider.cget("from"))),
                          float(slider.cget("to")))
            slider.config(state="normal")
            slider.set(clamped)
            setattr(self.params, name, float(slider.get()))
        finally:
            self._updating = previous
            if was_disabled:
                slider.config(state="disabled")

    def _nudge(self, event, attr: str, step: float) -> None:
        if self._typing(event):
            return
        value = getattr(self.params, attr) + step
        if attr == "azimuth_deg":
            value %= 360.0
        setattr(self.params, attr, value)
        self.sliders[attr].set(value)  # dispara _from_slider -> _redraw

    # --- dibujo -------------------------------------------------------

    def _redraw(self) -> None:
        self._solve_auto()
        self._draw_photo()
        x_px, y_px, usable = project_profile(
            self.profile.azimuths_deg, self.profile.elevations_deg,
            self.params, self.full_w, self.full_h)
        self._draw_silhouette(x_px, y_px, usable)
        self._draw_peak_labels()
        self._draw_roi()
        self._draw_strip(x_px, y_px, usable)

    def _draw_peak_labels(self) -> None:
        """Nombres de peaks/ en su píxel sobre la foto.

        Se proyectan con la MISMA pinhole que la silueta (project_profile
        acepta los azimuts y elevaciones de los picos tal cual), no con un
        eje lineal en azimut: a 55° de campo la aproximación lineal
        desplazaría las etiquetas de los bordes ~4% del ancho, bastante para
        atribuir un bulto a la cima equivocada, que es justo lo que estas
        etiquetas existen para evitar.
        """
        self.canvas.delete("toponimo")
        if not self.show_labels or not self.peaks:
            return
        # El texto rotado 90° crece HACIA ARRIBA desde su ancla, así que el
        # ancla tiene que estar tan abajo como largo sea el texto. Con el
        # ancla en y=24 los nombres se salían por el borde superior y solo se
        # leía el arranque ("Alto…", "Nav…"). Ahora se reserva una banda y lo
        # que no cabe se trunca por el final, conservando la altitud, igual
        # que en render/.
        band_bottom = int(min(160, max(70, self.ch * 0.45)))
        max_text_px = band_bottom - 8
        for index, x, y in self._peak_label_slots():
            sight = self.peaks[index]
            unknown = sight.visibility is Visibility.UNKNOWN
            color = _LABEL_UNKNOWN if unknown else _LABEL_OK
            anchor_y = min(max(y, band_bottom), self.ch)
            self.canvas.create_line(x, band_bottom, x, anchor_y, fill=color,
                                    width=1, dash=(4, 3) if unknown else None,
                                    tags="toponimo")
            text = self._fit_label(self._peak_text(sight), max_text_px)
            # contorno barato dibujando el texto desplazado en negro: sobre
            # cielo claro y sobre terreno oscuro el mismo color no vale
            for dx, dy, shade in ((1, 1, "#000000"), (0, 0, color)):
                self.canvas.create_text(x + dx, band_bottom - 4 + dy,
                                        text=text, angle=90,
                                        anchor="w", fill=shade,
                                        font=self._label_font,
                                        tags="toponimo")

    def _fit_label(self, text: str, max_px: int) -> str:
        """Trunca el NOMBRE por el final si no cabe, conservando la altitud.

        "Cerro del Collado de la Torrec… 1524 m" sigue siendo reconocible en
        un mapa; cortar por el principio, no."""
        if self._label_font.measure(text) <= max_px:
            return text
        name, _, suffix = text.rpartition(" ")
        name, _, unit = name.rpartition(" ")   # separa "… 1524" y "m"
        suffix = f" {unit} {suffix}"
        while name and self._label_font.measure(name + "…" + suffix) > max_px:
            name = name[:-1]
        return name.rstrip() + "…" + suffix

    def _peak_label_slots(self) -> list[tuple[int, float, float]]:
        """Qué picos se etiquetan y dónde: (índice, x, y) en el lienzo.

        Colocación voraz por prioridad de elevación aparente, igual que
        render/: quien choca con una etiqueta ya puesta se descarta. Lo usan
        el lienzo Y el volcado de depuración, así que el volcado enseña
        exactamente lo mismo que la ventana, no una aproximación.
        """
        if not self.peaks:
            return []
        x_px, y_px, usable = project_profile(self.peak_az, self.peak_el,
                                             self.params, self.full_w,
                                             self.full_h)
        cx, cy = self._to_canvas(x_px, y_px)
        slot = 15  # ancho de ranura: alto de línea del texto rotado
        order = sorted(range(len(self.peaks)),
                       key=lambda i: -self.peaks[i].elevation_deg)
        placed: list[tuple[int, float, float]] = []
        for i in order:
            if not usable[i] or not (0 <= cx[i] <= self.cw):
                continue
            if any(abs(cx[i] - taken) < slot for _j, taken, _y in placed):
                continue
            placed.append((i, float(cx[i]), float(cy[i])))
        return placed

    @staticmethod
    def _peak_text(sight) -> str:
        unknown = sight.visibility is Visibility.UNKNOWN
        return (f"{sight.peak.label} {sight.h_m:.0f} m"
                + ("?" if unknown else ""))

    def _toggle_labels(self, event=None) -> None:
        if event is not None and self._typing(event):
            return
        self.show_labels = not self.show_labels
        self._draw_peak_labels()

    def _solve_auto(self) -> None:
        """Resuelve en forma cerrada los parámetros en modo automático.

        Fijados azimut y FOV, el residuo entre la cresta detectada y la línea
        proyectada es una recta en x: su ordenada da la inclinación y su
        pendiente el giro (ver solve_pitch_roll). Se recalcula en cada cambio,
        así el usuario solo maneja dos parámetros en vez de cuatro.
        """
        if not self.has_skyline or not any(v.get()
                                           for v in self.auto_vars.values()):
            return
        fit_cols, fit_rows = self._fit_columns()
        solved = solve_pitch_roll(
            self.profile.azimuths_deg, self.profile.elevations_deg,
            self.params, fit_cols, fit_rows, self.full_w, self.full_h)
        if solved is None:
            self.auto_label.config(
                fg="#8a6d3b",
                text="auto: no hay columnas suficientes en este encuadre; "
                     "mueve azimut/FOV o congela pitch y roll")
            return
        # solo se toca lo que esté en automático: un parámetro congelado es
        # una decisión del usuario, no una sugerencia
        for name, value in (("pitch_deg", solved[0]), ("roll_deg", solved[1])):
            if self.auto_vars[name].get():
                self._apply_value(name, value)
        self.auto_label.config(
            fg="#2e7d32",
            text=f"auto: pitch {self.params.pitch_deg:+.2f}°  "
                 f"roll {self.params.roll_deg:+.2f}°  sobre "
                 f"{fit_cols.size} columnas de cresta"
                 + (f" (recuadro; {self.skyline_cols.size} detectadas)"
                    if self.roi is not None else ""))

    # --- panel de sector ----------------------------------------------

    def _draw_sector_panel(self) -> None:
        """Perfil de 360° con el sector marcado. Es el mapa sobre el que el
        usuario dice 'la foto cubre más o menos esto'."""
        self.sector_canvas.delete("all")
        az = np.asarray(self.profile.azimuths_deg, dtype=float)
        elev = np.asarray(self.profile.elevations_deg, dtype=float)
        finite = elev[~np.isnan(elev)]
        lo = min(float(finite.min()), 0.0) if finite.size else -1.0
        hi = max(float(finite.max()), 0.0) if finite.size else 1.0
        pad = max((hi - lo) * 0.08, 0.05)
        lo, hi = lo - pad, hi + pad
        top, bottom = 16, _SECTOR_H - 14

        def x_of(az_deg):
            return az_deg / 360.0 * self.cw

        def y_of(elev_deg):
            return top + (hi - elev_deg) / (hi - lo) * (bottom - top)

        points = [(x_of(a), y_of(e)) for a, e in zip(az, elev)
                  if not np.isnan(e)]
        if len(points) >= 2:
            polygon = points + [(points[-1][0], bottom), (points[0][0], bottom)]
            self.sector_canvas.create_polygon(
                [c for p in polygon for c in p], fill=_SECTOR_FILL, outline="")
        for az_deg, name in ((0, "N"), (90, "E"), (180, "S"), (270, "O"),
                             (360, "N")):
            x = x_of(az_deg)
            self.sector_canvas.create_line(x, top, x, bottom, fill="#d8dade")
            self.sector_canvas.create_text(x, _SECTOR_H - 6, text=name,
                                           fill="#555555", anchor="s")
        if self.sector is not None:
            center, width = self.sector
            start = (center - width / 2.0) % 360.0
            end = start + width
            # el sector puede cruzar el norte: se dibuja en dos trozos
            spans = ([(start, min(end, 360.0))]
                     + ([(0.0, end - 360.0)] if end > 360.0 else []))
            for a, b in spans:
                self.sector_canvas.create_rectangle(
                    x_of(a), top, x_of(b), bottom,
                    outline=_SECTOR_MARK, width=2, fill=_SECTOR_MARK,
                    stipple="gray25")
            self.sector_canvas.create_text(
                6, 8, anchor="nw", fill="#8a5000",
                text=f"sector {start:.1f}°–{end % 360.0:.1f}° ({width:.1f}°)")
        else:
            self.sector_canvas.create_text(
                6, 8, anchor="nw", fill="#777777",
                text="arrastra para marcar el arco que cubre la foto")

    def _sector_press(self, event) -> None:
        self._sector_anchor = event.x

    def _sector_drag(self, event) -> None:
        if getattr(self, "_sector_anchor", None) is None:
            return
        self._apply_sector(self._sector_anchor, event.x)

    def _apply_sector(self, x_from: int, x_to: int) -> None:
        start = min(max(x_from, 0), self.cw) / self.cw * 360.0
        end = min(max(x_to, 0), self.cw) / self.cw * 360.0
        if abs(end - start) < 0.5:
            return
        center, width, _lo, _hi, fov_lo, fov_hi = sector_to_ranges(start, end)
        self.sector = (center, width)
        # el azimut se mueve DENTRO del sector; el FOV alrededor de su ancho
        self._set_slider_range("azimuth_deg", center - width / 2.0,
                               center + width / 2.0)
        self._set_slider_range("hfov_deg", fov_lo, fov_hi)
        self._refresh_slider_labels()
        self._draw_sector_panel()
        self._redraw()

    def _set_slider_range(self, name: str, lo: float, hi: float) -> None:
        slider = self.sliders[name]
        was_disabled = str(slider.cget("state")) == "disabled"
        previous, self._updating = self._updating, True
        try:
            slider.config(state="normal", from_=lo, to=hi)
        finally:
            self._updating = previous
            if was_disabled:
                slider.config(state="disabled")
        self._apply_value(name, getattr(self.params, name))

    def _run_search(self, event=None) -> None:
        """Tecla S: búsqueda acotada al sector. NUNCA automática al marcar el
        arco — arrancar una búsqueda de segundos mientras el usuario reajusta
        el sector bloquearía la ventana por sorpresa."""
        if event is not None and self._typing(event):
            return
        if not self.has_skyline:
            self.status.config(text="búsqueda: sin cresta detectable en la foto")
            return
        if self.sector is None:
            # sin sector marcado se barren los 360°: es preferible a negarse,
            # y mucho mejor que buscar en un sector arbitrario
            self.status.config(text="buscando en los 360° (marca un sector "
                                    "para acotar)…")
            kwargs = dict(center_az_deg=None)
        else:
            center, width = self.sector
            self.status.config(text=f"buscando en {width:.0f}° alrededor de "
                                    f"{center:.1f}°…")
            kwargs = dict(center_az_deg=center,
                          az_margin_deg=max(width / 2.0, 1.0),
                          fov_hint_deg=width,
                          fov_margin_deg=max(width / 2.0, 2.0))
        self.root.update_idletasks()   # el mensaje ANTES de bloquear
        fit_cols, fit_rows = self._fit_columns()
        result = search_alignment(
            self.profile.azimuths_deg, self.profile.elevations_deg,
            fit_cols, fit_rows, self.full_w, self.full_h, **kwargs)
        if not result.candidates:
            self.status.config(text="búsqueda: ningún candidato encaja")
            return
        # los candidatos quedan guardados: con 1..5 se salta entre ellos para
        # compararlos POR TOPÓNIMOS, que es el criterio que de verdad zanja
        self.candidates = result.candidates
        reservas = []
        if result.ambiguous:
            reservas.append(f"AMBIGUO (otro óptimo solo "
                            f"{result.ambiguity_margin:.0%} peor)")
        if result.fov_at_edge:
            reservas.append("FOV en el borde del rango")
        if result.az_at_edge:
            reservas.append("azimut en el borde del sector")
        self._search_note = ("  ⚠ " + "; ".join(reservas) if reservas
                             else "  sin reservas")
        self._apply_candidate(0)

    def _apply_candidate(self, index: int, event=None) -> None:
        """Carga el candidato `index` de la última búsqueda.

        Los candidatos están SEPARADOS entre sí en azimut, así que saltar de
        uno a otro cambia de hipótesis de verdad — y lo que hay que mirar al
        compararlos son los topónimos, no el encaje de la línea.
        """
        if event is not None and self._typing(event):
            return
        # sin realimentación, "no hay candidatos todavía" es indistinguible
        # de "la tecla no funciona", que es lo que costó descubrir el fallo
        # de binding: no callarse nunca
        if not self.candidates:
            self.status.config(text="aún no hay candidatos: pulsa S para "
                                    "buscar (en el sector marcado, o en los "
                                    "360° si no hay ninguno)")
            return
        if not 0 <= index < len(self.candidates):
            self.status.config(
                text=f"solo hay {len(self.candidates)} candidatos "
                     f"(1..{len(self.candidates)})")
            return
        chosen = self.candidates[index]
        for name in ("azimuth_deg", "hfov_deg", "pitch_deg", "roll_deg"):
            value = getattr(chosen.params, name)
            slider = self.sliders[name]
            lo, hi = float(slider.cget("from")), float(slider.cget("to"))
            # el rango se ensancha si hace falta: un candidato de otro sector
            # cae fuera del rango que fijó el sector marcado
            self._set_slider_range(name, min(lo, value), max(hi, value))
            self._apply_value(name, value)
        self._refresh_slider_labels()
        self._redraw()
        others = "  ".join(f"{i + 1}:az{c.params.azimuth_deg:.0f}"
                           for i, c in enumerate(self.candidates)
                           if i != index)
        self.status.config(
            text=f"candidato {index + 1}/{len(self.candidates)}: "
                 f"az={chosen.params.azimuth_deg:.2f} "
                 f"fov={chosen.params.hfov_deg:.2f} ({chosen.error_px:.1f} px)"
                 + self._search_note
                 + (f"   otros -> {others}" if others else ""))

    def _draw_photo(self) -> None:
        """La foto solo se re-renderiza al cambiar la vista: mover un slider
        no la toca, y así el arrastre de los sliders va fluido."""
        key = (round(self.view_scale, 6), round(self.view_x, 2),
               round(self.view_y, 2))
        if key == self._photo_cache_key:
            return
        self.tk_img = ImageTk.PhotoImage(self._view_image())
        self.canvas.delete("foto")
        self.canvas.create_image(0, 0, image=self.tk_img, anchor="nw",
                                 tags="foto")
        self.canvas.tag_lower("foto")
        self._photo_cache_key = key

    def _view_image(self) -> Image.Image:
        """El encuadre actual como imagen PIL — la misma llamada para el
        canvas y para el volcado de depuración."""
        box = (self.view_x, self.view_y,
               self.view_x + self.cw / self.view_scale,
               self.view_y + self.ch / self.view_scale)
        return self.img.resize((self.cw, self.ch), Image.BILINEAR, box=box)

    def _to_canvas(self, x_px, y_px):
        return ((x_px - self.view_x) * self.view_scale,
                (y_px - self.view_y) * self.view_scale)

    def _draw_silhouette(self, x_px, y_px, usable) -> None:
        truncated = ~np.isnan(self.profile.truncated_at_m)
        cx, cy = self._to_canvas(x_px, y_px)
        # margen de un lienzo a cada lado: una muestra casi perpendicular al
        # eje óptico proyecta a tan() gigante y no se le pueden dar al canvas
        # coordenadas de millones de píxeles
        drawable = (usable & (cx > -self.cw) & (cx < 2 * self.cw)
                    & (cy > -self.ch) & (cy < 2 * self.ch))
        self._last_sil = (cx, cy, drawable, truncated)
        self.canvas.delete("silueta")
        for start, end in _runs(drawable):
            points = [c for pair in zip(cx[start:end], cy[start:end])
                      for c in pair]
            if len(points) >= 4:
                is_trunc = bool(truncated[start:end].any())
                self.canvas.create_line(
                    *points, tags="silueta", width=2,
                    fill=_LINE_TRUNC if is_trunc else _LINE_OK,
                    dash=(6, 4) if is_trunc else None)

    def _draw_strip(self, x_px, y_px, usable) -> None:
        """Tira de recorte: banda de ±_BAND_DEG alrededor de la silueta,
        enderezada (la línea proyectada queda recta en el centro) y estirada
        en vertical. El juicio de si encaja lo hace el usuario mirando."""
        columns_px = self.view_x + np.arange(self.cw) / self.view_scale
        patch, _ = build_strip(self.photo_np, x_px, y_px, usable, columns_px,
                               self._band_px(), _STRIP_H)
        self._last_patch = patch
        self.strip_img = ImageTk.PhotoImage(Image.fromarray(patch))
        self.strip.delete("all")
        self.strip.create_image(0, 0, image=self.strip_img, anchor="nw")
        # la silueta proyectada cae en el centro de la tira por construcción
        self.strip.create_line(0, _STRIP_H // 2, self.cw, _STRIP_H // 2,
                               fill=_LINE_OK, width=1)

    def _band_px(self) -> float:
        focal_px = (self.full_w / 2.0) / math.tan(
            math.radians(self.params.hfov_deg) / 2.0)
        return focal_px * math.tan(math.radians(_BAND_DEG))

    # --- volcado forense ----------------------------------------------

    def _dump_debug(self, event=None) -> None:
        """Tecla D: vuelca junto a la foto EXACTAMENTE lo que se acaba de
        dibujar. La tira sale del mismo array que se convirtió en ImageTk, y
        el canvas de la misma llamada de resize más la silueta con los
        mismos puntos. Si la tira sale mal y el canvas bien, el fallo está
        en la ruta de la tira; si ambos mal, en la carga o la proyección:
        la comparación parte el espacio de búsqueda por la mitad."""
        if event is not None and self._typing(event):
            return
        base = Path(self.photo_path)
        prefix = str(base.with_name(base.stem))

        Image.fromarray(self._last_patch).save(prefix + ".debug_tira.png")

        view = self._view_image().copy()
        drawer = ImageDraw.Draw(view)
        cx, cy, drawable, _trunc = self._last_sil
        for start, end in _runs(drawable):
            points = list(zip(cx[start:end], cy[start:end]))
            if len(points) >= 2:
                drawer.line(points, fill=(255, 59, 48), width=2)
        # los topónimos van también al volcado: son el criterio con el que se
        # juzga si el alineamiento es correcto, así que un volcado sin ellos
        # no permitiría revisar la decisión después
        for index, x, y in self._peak_label_slots():
            anchor_y = min(max(y, 0), self.ch)
            drawer.line([x, 0, x, anchor_y], fill=(255, 255, 255), width=1)
            # el texto va PEGADO al pico, no en el borde superior: el volcado
            # se recorta a la banda de la línea y allí arriba se perdería
            # justo la información que se quiere revisar
            drawer.text((x + 2, max(anchor_y - 12, 0)),
                        self._peak_text(self.peaks[index]),
                        fill=(255, 255, 255))
        line_ys = cy[drawable]
        if line_ys.size:  # recorte a la zona de la línea, con contexto
            top = max(int(line_ys.min()) - 80, 0)
            bottom = min(int(line_ys.max()) + 80, self.ch)
            view = view.crop((0, top, self.cw, bottom))
        view.save(prefix + ".debug_canvas.png")

        state = "\n".join((
            f"photo={self.photo_path}",
            f"params={vars(self.params)}",
            f"full={self.full_w}x{self.full_h}",
            f"view_scale={self.view_scale:.6f} fit={self.fit_scale:.6f}",
            f"view_x={self.view_x:.2f} view_y={self.view_y:.2f}",
            f"cw={self.cw} ch={self.ch}",
            f"canvas_tk={self.canvas.winfo_width()}x{self.canvas.winfo_height()}",
            f"strip_tk={self.strip.winfo_width()}x{self.strip.winfo_height()}",
            f"patch={self._last_patch.shape}",
            f"photo_np={self.photo_np.shape}",
        ))
        Path(prefix + ".debug_estado.txt").write_text(state, encoding="utf-8")
        self.status.config(text=f"volcado: {prefix}.debug_*.png/.txt")

    # --- guardado -----------------------------------------------------

    def _save(self) -> None:
        json_path = alignment_json_path(self.photo_path)
        if self.resume is not None and "seed" in self.resume:
            # reanudación: el seed ORIGINAL sobrevive (la procedencia EXIF
            # de la primera vez es la medida del error de brújula)
            seed = self.resume["seed"]
        else:
            seed = {name: seed_entry(getattr(self.seed_params, name),
                                     self.seed_sources.get(name, "default"))
                    for name in ("azimuth_deg", "hfov_deg", "pitch_deg",
                                 "roll_deg")}
        observer = {"lat_deg": self.lat_deg, "lon_deg": self.lon_deg,
                    "eye_m": self.eye_m, "source": self.observer_source}
        save_alignment(str(json_path), Path(self.photo_path).name, observer,
                       seed, self.params, self.full_w, self.full_h,
                       notes=self.notes.get())
        self.saved_path = str(json_path)
        self.root.destroy()


def _runs(valid: np.ndarray) -> list[tuple[int, int]]:
    mask = np.concatenate(([False], valid, [False]))
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))
