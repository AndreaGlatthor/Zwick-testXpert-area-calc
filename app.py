"""Web-App für die Auswertung des T-Peel Tests mit Zwick TestXpert.

Dash + Plotly App mit Bild-Digitalisierung aus der Zwischenablage oder per Datei-Upload

Funktionen
----------
- Bild per Strg+V (Windows) / Cmd+V (macOS) direkt aus der Zwischenablage
    einfuegen ODER per Datei-Upload waehlen.
- Digitalisierung der Kurve direkt aus dem Screenshot:
        * Erkennung ueber den BLAEULICHEN Farbton (B > R und B > G), NICHT
            ueber einen exakten Farbwert -> robust gegen Anti-Aliasing.
        * Extraktion ueber zusammenhaengende Kurven-Komponente, um die
            relevante Linie robust von anderen Linien zu trennen.
- X-Achse: Dehnung in mm
- Y-Achse: Kraft in N
- Integration der Flaeche unter der Kurve.
- Anzeige der Flaeche und der mittleren Kraft sowie Plot mit schattierter Flaeche.

Start:   python app.py
Browser: http://127.0.0.1:8050
Einfuegen: Fenster anklicken und Strg+V druecken.

Benoetigte Pakete:
    pip install dash plotly pandas numpy scipy pillow
"""

import base64
import io
import os

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, ctx, dcc, html
from PIL import Image
from scipy import ndimage
from scipy.integrate import trapezoid

DEFAULT_INTEG_START_MM = 130.0
DEFAULT_INTEG_END_MM = 230.0
DEFAULT_X_AXIS_MIN_MM = 50.0
DEFAULT_X_AXIS_MAX_MM = 300.0
X_AXIS_SPAN_MM = DEFAULT_X_AXIS_MAX_MM - DEFAULT_X_AXIS_MIN_MM


def _env_flag(name: str, default: bool) -> bool:
    """Liest boolesche Umgebungsvariablen robust (1/true/on/yes)."""
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


DEBUG_MODE = _env_flag("APP_DEBUG", True)


# ===========================================================================
# KONFIGURATION DER ACHSEN-KALIBRIERUNG
# ---------------------------------------------------------------------------
# Diese Werte sind der FALLBACK fuer den Original-Screenshot. Im Normalbetrieb
# wird die Kalibrierung automatisch aus dem Bild bestimmt (auto_calibrate).
#   X0_COL      : Pixelspalte der Y-Achse bei Dehnung = DEFAULT_X_AXIS_MIN_MM
#   PX_PER_MM   : Pixel pro mm in X-Richtung
#   Y0_ROW      : Pixelzeile der Nulllinie (Kraft = 0 N)
#   PX_PER_N    : Pixel pro Newton in Y-Richtung
#   PLOT_TOP/BOTTOM/LEFT/RIGHT : Plotbereich (ohne Achsen/Beschriftung)
#
# Abstand der Achsen-Teilstriche in physikalischen Einheiten. Bei anderen
# Diagrammen ggf. anpassen (hier: kleine X-Striche = 10 mm, Y-Striche = 1.0 N).
# ===========================================================================
MM_PER_TICK = 10.0  # Dehnungs-Abstand zwischen zwei kleinen X-Teilstrichen
N_PER_TICK = 1.0  # Kraft-Abstand zwischen zwei kleinen Y-Teilstrichen

CALIB = dict(
    X0_COL=165.0,
    PX_PER_MM=(1683 - 545) / 150.0,  # 50er- und 150er-Gitterlinie
    Y0_ROW=723.0,
    PX_PER_N=71.0,  # 2-N-Abstand = 142 px
    PLOT_TOP=45,
    PLOT_BOTTOM=715,
    PLOT_LEFT=170,
    PLOT_RIGHT=1932,
)


# ===========================================================================
# AUTOMATISCHE ACHSENERKENNUNG
# ===========================================================================
def _peaks(arr, thr, min_dist=15):
    """Findet Indizes lokaler Maxima oberhalb thr mit Mindestabstand."""
    idx = np.where(arr > thr)[0]
    out = []
    for i in idx:
        if not out or i - out[-1] > min_dist:
            out.append(int(i))
        elif arr[i] > arr[out[-1]]:
            out[-1] = int(i)
    return out


def _axis_right_from_row(dark_row: np.ndarray, y_axis_col: int) -> int:
    """Schaetzt das rechte Ende der X-Achse aus der Achsenzeile.

    Nimmt den zusammenhaengenden dunklen Abschnitt, der die Y-Achse
    enthaelt, und liefert dessen rechtes Ende.
    """
    cols = np.where(dark_row)[0]
    if len(cols) == 0:
        return -1

    idx = int(np.argmin(np.abs(cols - y_axis_col)))
    right = int(cols[idx])
    for k in range(idx + 1, len(cols)):
        if cols[k] - cols[k - 1] > 8:
            break
        right = int(cols[k])
    return right


def _fallback_calib_from_image_shape(h: int, w: int) -> dict:
    """Skaliert den globalen Fallback grob auf die aktuelle Bildgroesse."""
    ref_h = CALIB["PLOT_BOTTOM"] + 2
    ref_w = CALIB["PLOT_RIGHT"] + 2
    sx = w / max(ref_w, 1)
    sy = h / max(ref_h, 1)
    return dict(
        X0_COL=float(CALIB["X0_COL"] * sx),
        Y0_ROW=float(CALIB["Y0_ROW"] * sy),
        PX_PER_MM=max(0.8, float(CALIB["PX_PER_MM"] * sx)),
        PX_PER_N=max(0.8, float(CALIB["PX_PER_N"] * sy)),
        PLOT_TOP=max(0, int(round(CALIB["PLOT_TOP"] * sy))),
        PLOT_BOTTOM=min(h - 1, int(round(CALIB["PLOT_BOTTOM"] * sy))),
        PLOT_LEFT=max(0, int(round(CALIB["PLOT_LEFT"] * sx))),
        PLOT_RIGHT=min(w - 1, int(round(CALIB["PLOT_RIGHT"] * sx))),
    )


def auto_calibrate(
    img: Image.Image, mm_per_tick: float = MM_PER_TICK, N_per_tick: float = N_PER_TICK
) -> dict:
    """Bestimmt die Achsen-Kalibrierung automatisch aus dem Diagramm.

    Vorgehen
    --------
    1. ACHSENLINIEN: Die durchgezogenen, dunklen (fast schwarzen) Achsen
       erzeugen je eine Spalte / Zeile mit sehr vielen dunklen Pixeln.
       - Y-Achse (Dehnung = X-Achsenstart) = Spalte mit den meisten dunklen Pixeln.
       - X-Achse (Kraft = 0) = Zeile mit den meisten dunklen Pixeln.
    2. TEILSTRICHE (Ticks):
       - Y-Ticks: kurze dunkle Marken direkt LINKS der Y-Achse.
       - X-Ticks: kurze dunkle Marken direkt UNTER der X-Achse.
       Der Median der Tick-Abstaende (in px) ergibt zusammen mit dem
       bekannten physikalischen Abstand pro Tick die Skalierung
       (px/mm bzw. px/N).
    3. PLOTBEREICH: oben am oberen Ende der Y-Achsenlinie, unten an der
       X-Achse, links neben der Y-Achse, rechts am Bildrand.

    Faellt etwas aus (z. B. keine Ticks gefunden), wird der globale
    Fallback CALIB verwendet."""
    a = np.array(img.convert("RGB")).astype(int)
    h, w, _ = a.shape
    gray = a.mean(2)
    dark = gray < 90  # fast schwarze Achsenpixel

    colcnt = dark.sum(0)
    rowcnt = dark.sum(1)

    # --- 1) Achsenlinien -------------------------------------------------
    y_axis_col = int(np.argmax(colcnt))  # vertikale Achse (Dehnung = X-Achsenstart)
    x_axis_row = int(np.argmax(rowcnt))  # horizontale Achse (Kraft = 0)

    # Plausibilitaet: Achsen muessen klar ausgepraegt sein
    if colcnt[y_axis_col] < 0.3 * h or rowcnt[x_axis_row] < 0.3 * w:
        return _fallback_calib_from_image_shape(h, w)

    # oberes Ende der Y-Achsenlinie (fuer PLOT_TOP)
    col_dark_rows = np.where(dark[:, y_axis_col])[0]
    axis_top = int(col_dark_rows.min()) if len(col_dark_rows) else 0

    # rechtes Ende der X-Achse (wichtig bei UIs mit rechter Seitenleiste)
    axis_right = _axis_right_from_row(dark[x_axis_row, :], y_axis_col)
    if axis_right <= y_axis_col + 40:
        axis_right = w - 2

    # --- 2) Teilstriche --------------------------------------------------
    lo = max(0, y_axis_col - 16)
    yt = _peaks(dark[:, lo : y_axis_col - 1].sum(1), 5)
    hi = min(h, x_axis_row + 17)
    xt = _peaks(dark[x_axis_row + 1 : hi, :].sum(0), 5)

    if len(yt) < 3 or len(xt) < 3:
        # Ticks nicht sicher erkennbar -> Achsenpositionen nutzen,
        # Skalierung aus Plot-Ausdehnung abschaetzen.
        px_per_mm_geom = (axis_right - (y_axis_col + 1)) / X_AXIS_SPAN_MM
        px_per_n_geom = (x_axis_row - axis_top) / 9.6
        return dict(
            X0_COL=float(y_axis_col),
            Y0_ROW=float(x_axis_row),
            PX_PER_MM=max(0.8, float(px_per_mm_geom)),
            PX_PER_N=max(0.8, float(px_per_n_geom)),
            PLOT_TOP=max(0, axis_top - 2),
            PLOT_BOTTOM=x_axis_row - 2,
            PLOT_LEFT=y_axis_col + 1,
            PLOT_RIGHT=min(w - 2, axis_right),
        )

    sy = float(np.median(np.diff(yt)))  # px je Y-Tick
    sx = float(np.median(np.diff(xt)))  # px je X-Tick

    px_per_mm = sx / mm_per_tick
    px_per_n = sy / N_per_tick

    # Tick-Erkennung auf Plausibilitaet pruefen. Bei extremen Werten
    # stattdessen Geometrie-basierte Skalierung verwenden.
    px_per_mm_geom = (axis_right - (y_axis_col + 1)) / X_AXIS_SPAN_MM
    px_per_n_geom = (x_axis_row - axis_top) / 9.6
    mm_ok = 0.4 * px_per_mm_geom <= px_per_mm <= 2.8 * px_per_mm_geom
    n_ok = 0.4 * px_per_n_geom <= px_per_n <= 2.8 * px_per_n_geom
    if not (mm_ok and n_ok):
        px_per_mm = max(0.8, float(px_per_mm_geom))
        px_per_n = max(0.8, float(px_per_n_geom))

    return dict(
        X0_COL=float(y_axis_col),
        Y0_ROW=float(x_axis_row),
        PX_PER_MM=float(px_per_mm),
        PX_PER_N=float(px_per_n),
        PLOT_TOP=max(0, axis_top - 2),
        PLOT_BOTTOM=x_axis_row - 2,
        PLOT_LEFT=y_axis_col + 1,
        PLOT_RIGHT=min(w - 2, axis_right),
    )


# ===========================================================================
# DIGITALISIERUNGSMETHODE
# ===========================================================================
def digitize_blue_curve(img: Image.Image, calib: dict = None):
    """Digitalisiert die Kurve aus einem Kraft-Weg-Diagramm.

    Wird keine Kalibrierung uebergeben, wird sie automatisch aus dem Bild
    bestimmt (auto_calibrate).

    Vorgehen
    --------
     1. Blau-Maske ueber den FARBTON: B > R+6 und B > G+6, plus
         Helligkeitsgrenzen, um Weiss/Gitter und zu helle Pixel auszuschliessen.
         Dadurch werden auch anti-aliaste (aufgehellte) Linienpixel erfasst.
     2. Kleine Luecken in der Maske schliessen und den zusammenhaengenden
         Kurvenzug ueber Connected Components bestimmen.
     3. Startpunkt robust waehlen und pro X-Spalte den oberen Punkt dieser
         Kurven-Komponente extrahieren.
    4. Pixel -> physikalische Einheiten umrechnen (mm bzw. N).

    Rueckgabe
    ---------
    mm : np.ndarray  Dehnung in mm
    N  : np.ndarray  Kraft in N
    """
    a = np.array(img.convert("RGB")).astype(int)
    R, G, B = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    mx = a.max(2)

    # Kalibrierung automatisch bestimmen, falls keine uebergeben wurde
    if calib is None:
        calib = auto_calibrate(img)

    # 1) Blau-Maske ueber dominanten Blauanteil und Saettigung,
    #    mit adaptiven Schwellwerten gegen unterschiedliche Monitorprofile.
    blue_dom = B - np.maximum(R, G)
    sat = mx - a.min(2)

    dom_pos = blue_dom[blue_dom > 0]
    dom_thr = 6.0
    if len(dom_pos) > 200:
        dom_thr = max(4.0, float(np.percentile(dom_pos, 35)))

    sat_floor = max(12.0, float(np.percentile(sat, 55)))
    navy = (blue_dom >= dom_thr) & (B >= 20) & (sat >= sat_floor)

    # auf Plotbereich begrenzen
    navy[: calib["PLOT_TOP"], :] = False
    navy[calib["PLOT_BOTTOM"] :, :] = False
    navy[:, : calib["PLOT_LEFT"]] = False
    navy[:, calib["PLOT_RIGHT"] :] = False

    _, w = navy.shape
    c_right = min(calib["PLOT_RIGHT"], w)
    c_left = calib["PLOT_LEFT"]

    # 2) Kleine Luecken in der Linie schliessen, damit die Kurve als
    #    zusammenhaengende Komponente erkannt werden kann.
    closed = ndimage.binary_closing(navy, structure=np.ones((3, 3)), iterations=1)

    # Kandidaten je Spalte (aus Originalmaske fuer praezise Y-Lage)
    cand = {c: np.where(navy[:, c])[0] for c in range(c_left, c_right)}

    # 3) Startpunkt robust waehlen:
    #    bevorzugt in der Diagrammmitte, sonst aus realen Kandidatenspalten.
    cols_with = np.array(
        [c for c in range(c_left, c_right) if len(cand.get(c, [])) > 0], dtype=int
    )
    if len(cols_with) == 0:
        raise ValueError(
            "Keine Kurve im Bild gefunden. Bitte Farbe/Kalibrierung pruefen."
        )

    seed_target_mm = DEFAULT_X_AXIS_MIN_MM + 0.5 * X_AXIS_SPAN_MM
    seed_target = int(
        (seed_target_mm - DEFAULT_X_AXIS_MIN_MM) * calib["PX_PER_MM"] + calib["X0_COL"]
    )
    if cols_with.min() <= seed_target <= cols_with.max():
        seed_c = int(cols_with[np.argmin(np.abs(cols_with - seed_target))])
    else:
        # eher rechte Diagrammhaelfte waehlen; dort sind obere/untere Kurve
        # meist besser getrennt.
        seed_c = int(cols_with[(2 * len(cols_with)) // 3])

    seed_r = int(cand[seed_c].min())  # oberster (= hoechste Kraft) Pixel

    # 3b) Verbundene Kurvenkomponente am Seed bestimmen.
    labels, n_labels = ndimage.label(closed, structure=np.ones((3, 3), dtype=bool))
    if n_labels <= 0:
        raise ValueError("Keine zusammenhaengende Kurve erkannt.")

    seed_label = int(labels[seed_r, seed_c])
    if seed_label == 0:
        raise ValueError("Seed liegt nicht auf einer erkannten Kurve.")

    component = labels == seed_label

    # Pro Spalte den obersten Punkt der Seed-Komponente nehmen.
    # Prioritaet hat die Originalmaske (praeziser), sonst Closed-Maske.
    track = {}
    for c in range(c_left, c_right):
        rs_precise = np.where(component[:, c] & navy[:, c])[0]
        if len(rs_precise) > 0:
            track[c] = int(rs_precise.min())
            continue

        rs_closed = np.where(component[:, c])[0]
        if len(rs_closed) > 0:
            track[c] = int(rs_closed.min())

    # Falls die Kurve deutlich vor dem erwarteten Ende aufhoert, am rechten
    # Rand mit etwas weicheren Farbgrenzen nachfassen (nur lokal-kontinuierlich).
    if len(track) > 0:
        cols_now = np.array(sorted(track), dtype=float)
        mm_now = DEFAULT_X_AXIS_MIN_MM + (
            (cols_now - calib["X0_COL"]) / calib["PX_PER_MM"]
        )
        if float(mm_now.max()) < (DEFAULT_X_AXIS_MAX_MM - 20.0):
            dom_thr_soft = max(2.0, 0.45 * dom_thr)
            sat_floor_soft = max(5.0, 0.30 * sat_floor)
            navy_soft = (blue_dom >= dom_thr_soft) & (B >= 12) & (sat >= sat_floor_soft)

            navy_soft[: calib["PLOT_TOP"], :] = False
            navy_soft[calib["PLOT_BOTTOM"] :, :] = False
            navy_soft[:, : calib["PLOT_LEFT"]] = False
            navy_soft[:, calib["PLOT_RIGHT"] :] = False

            last_c = int(max(track.keys()))
            p = int(track[last_c])
            misses = 0
            max_jump_soft = max(
                18, int(0.09 * max(1, calib["PLOT_BOTTOM"] - calib["PLOT_TOP"]))
            )
            for c in range(last_c + 1, c_right):
                rs = np.where(navy_soft[:, c])[0]
                if len(rs) == 0:
                    misses += 1
                    if misses > 10:
                        break
                    continue

                j = int(rs[np.argmin(np.abs(rs - p))])
                dj = abs(j - p)
                if dj <= max_jump_soft or ((j > p) and (dj <= 3 * max_jump_soft)):
                    track[c] = j
                    p = j
                    misses = 0
                    continue

                misses += 1
                if misses > 10:
                    break

    if len(track) < 3:
        raise ValueError("Zu wenige Kurvenpunkte erkannt.")

    cols = np.array(sorted(track), dtype=float)
    rows = np.array([track[int(c)] for c in cols], dtype=float)

    # 4) Umrechnung in physikalische Einheiten
    mm = DEFAULT_X_AXIS_MIN_MM + (cols - calib["X0_COL"]) / calib["PX_PER_MM"]
    N = (calib["Y0_ROW"] - rows) / calib["PX_PER_N"]
    N = np.clip(N, 0, None)

    return mm, N


def _slice_curve_interval(
    mm: np.ndarray, N: np.ndarray, start_mm: float, end_mm: float
):
    """Kurvenausschnitt definieren.

    Schneidet die Kurve auf ein gewünschtes Intervall zu und sorgt
    dafür, dass die Intervallgrenzen exakt enthalten sind.
    """
    if len(mm) == 0:  # keine Datenpunkte -> leeres Ergebnis
        return np.array([]), np.array([])

    order = np.argsort(mm)
    x = mm[order].astype(float)
    y = N[order].astype(float)

    lo = max(float(np.min(x)), float(min(start_mm, end_mm)))
    hi = min(float(np.max(x)), float(max(start_mm, end_mm)))
    if hi <= lo:  # ungültiger Bereich -> leeres Ergebnis
        return np.array([]), np.array([])

    inside = (x >= lo) & (x <= hi)
    xs = x[inside]
    ys = y[inside]

    y_lo = float(np.interp(lo, x, y))  # Interpolierter y-Wert an der unteren Grenze
    y_hi = float(np.interp(hi, x, y))  # Interpolierter y-Wert an der oberen Grenze

    if len(xs) == 0 or xs[0] > lo:  # untere Grenze nicht enthalten -> hinzufügen
        xs = np.insert(xs, 0, lo)
        ys = np.insert(ys, 0, y_lo)
    else:  # untere Grenze bereits enthalten -> sicherstellen, dass y-Wert korrekt ist
        xs[0] = lo
        ys[0] = y_lo

    if xs[-1] < hi:  # obere Grenze nicht enthalten -> hinzufügen
        xs = np.append(xs, hi)
        ys = np.append(ys, y_hi)
    else:  # obere Grenze bereits enthalten -> sicherstellen, dass y-Wert korrekt ist
        xs[-1] = hi
        ys[-1] = y_hi

    return xs, ys  # sortierte Arrays mit garantiert enthaltenen Intervallgrenzen


def compute_area_interval_Nm(
    mm: np.ndarray, N: np.ndarray, start_mm: float, end_mm: float
) -> float:
    """Flaeche nur im Bereich [start_mm, end_mm], mit Interpolation an den
    Grenzen."""
    xs, ys = _slice_curve_interval(mm, N, start_mm, end_mm)
    if len(xs) < 2:
        return 0.0
    return float(trapezoid(ys, xs / 1000.0))


def compute_mean_force_interval_N(
    mm: np.ndarray, N: np.ndarray, start_mm: float, end_mm: float
) -> float:
    """Mittlere Kraft im Bereich [start_mm, end_mm] in N."""
    xs, ys = _slice_curve_interval(mm, N, start_mm, end_mm)
    if len(xs) < 2:
        return 0.0

    path_length_m = float(xs[-1] - xs[0]) / 1000.0
    if path_length_m <= 0.0:
        return 0.0

    return float(trapezoid(ys, xs / 1000.0) / path_length_m)


def format_area_milli(area_Nm: float) -> str:
    """Formatiert Flaeche als ganzzahlige mN·m-Anzeige."""
    area_milli = int(round(float(area_Nm) * 1000.0))
    return f"{area_milli} mN\u00b7m"


def format_mean_force_N(force_N: float) -> str:
    """Formatiert die mittlere Kraft als Newton-Anzeige mit zwei
    Nachkommastellen."""
    return f"{round(float(force_N), 2)} N"


# ===========================================================================
# PLOT
# ===========================================================================
def make_figure(
    mm,
    N,
    integ_start_mm=DEFAULT_INTEG_START_MM,
    integ_end_mm=DEFAULT_INTEG_END_MM,
) -> go.Figure:
    fig = go.Figure()
    x_fill, y_fill = _slice_curve_interval(mm, N, integ_start_mm, integ_end_mm)
    # schattierte Flaeche im Integrationsbereich
    fig.add_trace(
        go.Scatter(
            x=x_fill,
            y=y_fill,
            mode="lines",
            line=dict(width=0),
            fill="tozeroy",
            fillcolor="#c9990d",
            name=(
                "Integrierte Fläche "
                f"({min(integ_start_mm, integ_end_mm):.1f} bis "
                f"{max(integ_start_mm, integ_end_mm):.1f} mm)"
            ),
            hoverinfo="skip",
        )
    )
    # Kurve
    fig.add_trace(
        go.Scatter(
            x=mm,
            y=N,
            mode="lines",
            line=dict(color="#004684", width=4),
            name="Gefundene Kurve",
            hovertemplate="Dehnung: %{x:.1f} mm<br>Kraft: %{y:.2f} N<extra></extra>",
        )
    )
    fig.update_layout(
        xaxis_title="Dehnung in mm",
        yaxis_title="Kraft in N",
        xaxis_title_font=dict(size=18),
        yaxis_title_font=dict(size=18),
        template="plotly_white",
        hoverlabel=dict(
            bgcolor="white",
            bordercolor="lightgray",
            font=dict(color="black", size=14),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=70, r=30, t=70, b=60),
        height=600,
    )
    fig.update_xaxes(
        range=[DEFAULT_X_AXIS_MIN_MM, DEFAULT_X_AXIS_MAX_MM],
        autorange=False,
        showline=True,
        linewidth=2,
        linecolor="#14141a",
        mirror=True,
        ticks="outside",
        tickfont=dict(size=14),
        tickwidth=2,
        ticklen=8,
        tickcolor="#14141a",
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="#14141a",
        showgrid=True,
        gridcolor="lightgray",
        griddash="dash",
    )
    fig.update_yaxes(
        range=[0, max(9.6, (float(np.nanmax(N)) * 1.05) if len(N) else 9.6)],
        showline=True,
        linewidth=2,
        linecolor="#14141a",
        mirror=True,
        ticks="outside",
        tickfont=dict(size=14),
        tickwidth=2,
        ticklen=8,
        tickcolor="#14141a",
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="#14141a",
        showgrid=True,
        gridcolor="lightgray",
        griddash="dash",
    )
    return fig


def empty_figure(
    msg="Bitte Screenshot per Strg+V einfügen oder Bild-Datei hochladen.",
):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        height=600,
        xaxis_title="Dehnung in mm",
        yaxis_title="Kraft in N",
        annotations=[
            dict(
                text=msg,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="red"),
            )
        ],
    )
    fig.update_xaxes(
        range=[DEFAULT_X_AXIS_MIN_MM, DEFAULT_X_AXIS_MAX_MM], autorange=False
    )
    return fig


def decode_image(contents: str) -> Image.Image:
    """Wandelt einen data-URL String (base64) in ein PIL-Image um."""
    header, _, b64 = contents.partition(",")
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw))


_init_fig = empty_figure()
_init_area = "\u2013"
_init_mean_force = "\u2013"
_init_curve_data = None


# ===========================================================================
# DASH-APP
# ===========================================================================
app = dash.Dash(__name__)
app.title = "Kraft-Weg-Diagramm"

# Clientseitiges JavaScript: faengt Strg+V / Cmd+V ab, liest das Bild aus der
# Zwischenablage und legt den data-URL in ein verstecktes dcc.Store.
app.clientside_callback(
    """
    function(_) {
        if (!window.__pasteListenerAdded) {
            window.__pasteListenerAdded = true;
            document.addEventListener('paste', function(e) {
                const items = (e.clipboardData || window.clipboardData).items;
                for (let i = 0; i < items.length; i++) {
                    if (items[i].type.indexOf('image') !== -1) {
                        const blob = items[i].getAsFile();
                        const reader = new FileReader();
                        reader.onload = function(ev) {
                            const store = window.dash_clientside.set_props;
                            store('pasted-image', {data: ev.target.result});
                        };
                        reader.readAsDataURL(blob);
                        e.preventDefault();
                    }
                }
            });
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("paste-init", "data"),
    Input("paste-init", "data"),
)

app.layout = html.Div(
    style={
        "maxWidth": "1100px",
        "margin": "0 auto",
        "fontFamily": "Inter, Arial, sans-serif",
        "padding": "10px 12px",
    },
    children=[
        html.H1(
            "Auswertung T-Peel Test mit Zwick TestXpert",
            style={
                "textAlign": "center",
                "color": "#c9990d",
                "margin": "4px 0 10px 0",
            },
        ),
        dcc.Upload(
            id="upload-image",
            children=html.Div(
                (
                    "Screenshot mit Strg+V einfügen, Datei hierher ziehen oder "
                    "klicken zum Auswählen einer Datei."
                ),
                style={
                    "fontSize": "18px",
                    "color": "#004684",
                    "fontWeight": "700",
                },
            ),
            style={
                "width": "100%",
                "height": "70px",
                "lineHeight": "70px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "borderColor": "#004684",
                "borderRadius": "10px",
                "textAlign": "center",
                "color": "#004684",
                "marginBottom": "40px",
            },
            multiple=False,
        ),
        html.Div(
            id="status-msg",
            style={
                "display": "none",
                "textAlign": "center",
                "color": "#c0392b",
                "minHeight": "14px",
                "marginBottom": "8px",
            },
        ),
        html.Div(
            style={"display": "flex", "justifyContent": "center", "margin": "10px 0"},
            children=[
                html.Div(
                    style={
                        "background": "#f4f6f8",
                        "border": "1px solid #004684",
                        "borderRadius": "12px",
                        "padding": "12px",
                        "display": "flex",
                        "gap": "10px",
                        "flexWrap": "wrap",
                        "justifyContent": "center",
                    },
                    children=[
                        html.Div(
                            style={
                                "background": "white",
                                "border": "1px solid #d3dbe3",
                                "borderRadius": "10px",
                                "padding": "10px 12px",
                                "minWidth": "300px",
                                "textAlign": "center",
                            },
                            children=[
                                html.Div(
                                    "Auswertungsintervall definieren",
                                    style={"fontSize": "18px", "color": "#004684"},
                                ),
                                html.Div(
                                    style={
                                        "display": "flex",
                                        "gap": "12px",
                                        "justifyContent": "center",
                                        "marginTop": "8px",
                                    },
                                    children=[
                                        html.Div(
                                            children=[
                                                html.Div(
                                                    "Start [mm]",
                                                    style={
                                                        "fontSize": "16px",
                                                        "color": "#004684",
                                                        "marginBottom": "4px",
                                                    },
                                                ),
                                                dcc.Input(
                                                    id="integ-start",
                                                    type="number",
                                                    value=DEFAULT_INTEG_START_MM,
                                                    step=1,
                                                    style={
                                                        "width": "150px",
                                                        "padding": "6px 8px",
                                                        "borderRadius": "8px",
                                                        "border": "1px solid #004684",
                                                    },
                                                ),
                                            ]
                                        ),
                                        html.Div(
                                            children=[
                                                html.Div(
                                                    "Ende [mm]",
                                                    style={
                                                        "fontSize": "16px",
                                                        "color": "#004684",
                                                        "marginBottom": "4px",
                                                    },
                                                ),
                                                dcc.Input(
                                                    id="integ-end",
                                                    type="number",
                                                    value=DEFAULT_INTEG_END_MM,
                                                    step=1,
                                                    style={
                                                        "width": "150px",
                                                        "padding": "6px 8px",
                                                        "borderRadius": "8px",
                                                        "border": "1px solid #004684",
                                                    },
                                                ),
                                            ]
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                "background": "white",
                                "border": "1px solid #d3dbe3",
                                "borderRadius": "10px",
                                "padding": "10px 12px",
                                "minWidth": "220px",
                                "textAlign": "center",
                            },
                            children=[
                                html.Div(
                                    "Integrierte Fläche im Intervall",
                                    style={"fontSize": "18px", "color": "#004684"},
                                ),
                                html.Div(
                                    id="area-value",
                                    children=_init_area,
                                    style={
                                        "fontSize": "24px",
                                        "fontWeight": "700",
                                        "color": "#004684",
                                        "marginTop": "8px",
                                    },
                                ),
                                html.Div(
                                    "(mN\u00b7m entspricht mJ)",
                                    style={"fontSize": "13px", "color": "grey"},
                                ),
                            ],
                        ),
                        html.Div(
                            style={
                                "background": "white",
                                "border": "1px solid #d3dbe3",
                                "borderRadius": "10px",
                                "padding": "10px 12px",
                                "minWidth": "220px",
                                "textAlign": "center",
                            },
                            children=[
                                html.Div(
                                    "Mittlere Kraft im Intervall",
                                    style={"fontSize": "18px", "color": "#004684"},
                                ),
                                html.Div(
                                    id="mean-force-value",
                                    children=_init_mean_force,
                                    style={
                                        "fontSize": "24px",
                                        "fontWeight": "700",
                                        "color": "#004684",
                                        "marginTop": "8px",
                                    },
                                ),
                            ],
                        ),
                    ],
                )
            ],
        ),
        dcc.Graph(
            id="kraft-weg-graph",
            figure=_init_fig,
            config={
                "displaylogo": False,
                "modeBarButtonsToRemove": [
                    "toImage",
                    "sendDataToCloud",
                    "editInChartStudio",
                    "zoom2d",
                    "zoomIn2d",
                    "zoomOut2d",
                    "select2d",
                    "pan2d",
                    "lasso2d",
                    "autoScale2d",
                    "resetScale2d",
                ],
            },
        ),
        # versteckte Stores
        dcc.Store(id="pasted-image"),
        dcc.Store(id="paste-init", data=0),
        dcc.Store(id="curve-data", data=_init_curve_data),
    ],
)


@app.callback(
    Output("kraft-weg-graph", "figure"),
    Output("area-value", "children"),
    Output("mean-force-value", "children"),
    Output("status-msg", "children"),
    Output("status-msg", "style"),
    Output("curve-data", "data"),
    Input("pasted-image", "data"),
    Input("upload-image", "contents"),
    Input("integ-start", "value"),
    Input("integ-end", "value"),
    State("curve-data", "data"),
    prevent_initial_call=True,
)
def update_from_image(pasted, uploaded, integ_start, integ_end, curve_data):
    # Quelle bestimmen (zuletzt ausgeloestes Ereignis)
    trigger = ctx.triggered_id

    status_style_hidden = {
        "display": "none",
        "textAlign": "center",
        "color": "#c0392b",
        "minHeight": "14px",
        "marginBottom": "8px",
    }
    status_style_visible = {
        "display": "block",
        "textAlign": "center",
        "color": "#c0392b",
        "minHeight": "14px",
        "marginBottom": "8px",
    }

    try:
        start_mm = float(DEFAULT_INTEG_START_MM if integ_start is None else integ_start)
        end_mm = float(DEFAULT_INTEG_END_MM if integ_end is None else integ_end)
    except Exception:
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            "Fehler: Start/Ende muessen Zahlen sein.",
            status_style_visible,
            dash.no_update,
        )

    if start_mm > end_mm:
        start_mm, end_mm = end_mm, start_mm

    if trigger == "pasted-image" and pasted:
        contents = pasted.get("data") if isinstance(pasted, dict) else pasted
    elif trigger == "upload-image" and uploaded:
        contents = uploaded
    else:
        if curve_data and isinstance(curve_data, dict):
            mm = np.asarray(curve_data.get("mm", []), dtype=float)
            N = np.asarray(curve_data.get("N", []), dtype=float)
            if len(mm) >= 2 and len(N) >= 2:
                area = compute_area_interval_Nm(mm, N, start_mm, end_mm)
                mean_force = compute_mean_force_interval_N(mm, N, start_mm, end_mm)
                fig = make_figure(mm, N, start_mm, end_mm)
                return (
                    fig,
                    format_area_milli(area),
                    format_mean_force_N(mean_force),
                    "",
                    status_style_hidden,
                    dash.no_update,
                )
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            "Bitte zuerst ein Bild einfügen oder hochladen.",
            status_style_visible,
            dash.no_update,
        )

    try:
        img = decode_image(contents)
        calib = auto_calibrate(img)
        mm, N = digitize_blue_curve(img, calib)
        area = compute_area_interval_Nm(mm, N, start_mm, end_mm)
        mean_force = compute_mean_force_interval_N(mm, N, start_mm, end_mm)
        fig = make_figure(mm, N, start_mm, end_mm)
        # Ergebnis als CSV mitschreiben (optional)
        try:
            pd.DataFrame({"dehnung_mm": mm.round(3), "kraft_N": N.round(4)}).to_csv(
                os.path.join(os.path.dirname(__file__), "xy-data.csv"), index=False
            )
        except Exception:
            pass
        return (
            fig,
            format_area_milli(area),
            format_mean_force_N(mean_force),
            "",
            status_style_hidden,
            {"mm": mm.tolist(), "N": N.tolist()},
        )
    except Exception as e:
        return (
            dash.no_update,
            dash.no_update,
            dash.no_update,
            f"Fehler: {e}",
            status_style_visible,
            dash.no_update,
        )


if __name__ == "__main__":
    app.run(
        debug=DEBUG_MODE,
        use_reloader=DEBUG_MODE,
        dev_tools_hot_reload=DEBUG_MODE,
        dev_tools_ui=DEBUG_MODE,
        dev_tools_hot_reload_watch_interval=0.5,
        dev_tools_hot_reload_interval=1.0,
    )
