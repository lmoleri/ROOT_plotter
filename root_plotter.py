#!/usr/bin/env python3
"""Interactive ROOT plotting GUI — histograms, graphs, and 2D histograms."""

from __future__ import annotations

import contextlib
import csv
import os
import re
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ROOT batch mode MUST be set before any PyQt5 import
import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.gROOT.ProcessLine("gErrorIgnoreLevel = kWarning;")

from PyQt5.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANVAS_WIDTH = 900
CANVAS_HEIGHT = 700
DEBOUNCE_MS = 400

PLOT_TYPES = [
    "TH1F (1D Histogram)",
    "TGraph (Scatter/Line)",
    "TH2F (2D Histogram)",
]

ROOT_COLORS: List[Tuple[str, int, str]] = [
    ("Black",   ROOT.kBlack,        "#000000"),
    ("Red",     ROOT.kRed,          "#ff0000"),
    ("Blue",    ROOT.kBlue,         "#0000ff"),
    ("Green",   ROOT.kGreen + 2,    "#00aa00"),
    ("Magenta", ROOT.kMagenta,      "#ff00ff"),
    ("Cyan",    ROOT.kCyan + 1,     "#00cccc"),
    ("Orange",  ROOT.kOrange,       "#ffaa00"),
    ("Violet",  ROOT.kViolet,       "#7700ff"),
]

ROOT_MARKERS: List[Tuple[str, int]] = [
    ("Full Circle",        ROOT.kFullCircle),
    ("Full Square",        ROOT.kFullSquare),
    ("Full Triangle Up",   ROOT.kFullTriangleUp),
    ("Full Triangle Down", ROOT.kFullTriangleDown),
    ("Open Circle",        ROOT.kOpenCircle),
    ("Open Square",        ROOT.kOpenSquare),
]

TH1F_DRAW_STYLES: List[Tuple[str, str]] = [
    ("Histogram (line)",  "HIST"),
    ("Error bars",        "E1"),
    ("Points",            "P"),
    ("Filled histogram",  "HIST F"),
    ("Bar chart",         "BAR"),
]

TGRAPH_DRAW_STYLES: List[Tuple[str, str]] = [
    ("Points only",     "P"),
    ("Line only",       "L"),
    ("Line + Points",   "LP"),
    ("Smooth curve",    "C"),
    ("Smooth + Points", "CP"),
]

TH2F_DRAW_OPTIONS: List[str] = ["COLZ", "LEGO", "SURF1", "SURF2", "CONT4Z"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SeriesData:
    name: str = "Series 1"
    raw_text: str = ""
    csv_path: str = ""
    use_csv: bool = False
    color_index: int = ROOT.kBlue
    color_hex: str = "#0000ff"
    marker_style: int = ROOT.kFullCircle
    draw_style: str = "HIST"
    line_width: int = 2
    fill_style: int = 0


@dataclass
class PlotConfig:
    plot_type: str = "TH1F (1D Histogram)"
    title: str = ""
    x_title: str = ""
    y_title: str = ""
    z_title: str = ""
    n_bins_x: int = 50
    n_bins_y: int = 50
    x_min: float = 0.0
    x_max: float = 0.0
    y_min: float = 0.0
    y_max: float = 0.0
    log_x: bool = False
    log_y: bool = False
    log_z: bool = False
    grid_x: bool = False
    grid_y: bool = False
    show_legend: bool = True
    latex_text: str = ""
    latex_x: float = 0.13
    latex_y: float = 0.88
    series: List[SeriesData] = field(default_factory=lambda: [SeriesData()])
    th2f_draw_option: str = "COLZ"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class PlotError(ValueError):
    """Raised when data cannot be parsed or ROOT rendering fails."""


# ---------------------------------------------------------------------------
# Data parsing (pure functions, no GUI)
# ---------------------------------------------------------------------------

def _split_tokens(text: str) -> List[str]:
    return [t for t in re.split(r"[,\s]+", text.strip()) if t]


def parse_1d_values(text: str) -> List[Tuple[float, float]]:
    """Parse 1D values. Returns (value, weight) pairs.

    Supports two formats:
    - 1-column: flat tokens → weight=1.0 for each
    - 2-column: 'value weight' per line (used when loading a macro)
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines and all(len(_split_tokens(ln)) >= 2 for ln in lines):
        result = []
        for i, line in enumerate(lines, 1):
            cols = _split_tokens(line)
            try:
                result.append((float(cols[0]), float(cols[1])))
            except ValueError as exc:
                raise PlotError(f"Non-numeric value on line {i}: {exc}") from exc
        return result
    tokens = _split_tokens(text)
    if not tokens:
        raise PlotError("No data found. Enter numbers separated by spaces, commas, or newlines.")
    try:
        return [(float(t), 1.0) for t in tokens]
    except ValueError as exc:
        raise PlotError(f"Non-numeric value: {exc}") from exc


def parse_2col_values(text: str) -> List[Tuple[float, float, float, float]]:
    """Parse x y [ex ey] rows. ex/ey default to 0 if omitted."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        raise PlotError("No data found. Enter x y pairs (one per line).")
    result = []
    for i, line in enumerate(lines, 1):
        cols = _split_tokens(line)
        if len(cols) < 2:
            continue  # single value treated as x-only, skip
        try:
            x, y = float(cols[0]), float(cols[1])
            ex = float(cols[2]) if len(cols) > 2 else 0.0
            ey = float(cols[3]) if len(cols) > 3 else 0.0
            result.append((x, y, ex, ey))
        except ValueError as exc:
            raise PlotError(f"Non-numeric value on line {i}: {exc}") from exc
    return result


def parse_3col_values(text: str) -> List[Tuple[float, float, float]]:
    """Parse x y [z] rows. z defaults to 1.0 (unweighted fill)."""
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        raise PlotError("No data found for 2D histogram.")
    result = []
    for i, line in enumerate(lines, 1):
        cols = _split_tokens(line)
        if len(cols) < 2:
            continue  # single value treated as x-only, skip
        try:
            x, y = float(cols[0]), float(cols[1])
            z = float(cols[2]) if len(cols) > 2 else 1.0
            result.append((x, y, z))
        except ValueError as exc:
            raise PlotError(f"Non-numeric value on line {i}: {exc}") from exc
    return result


def _open_csv(path: str):
    return open(path, newline="", encoding="utf-8-sig")


def load_csv_1d(path: str) -> List[Tuple[float, float]]:
    """Load 1D data from CSV. Returns (value, weight) pairs; weight from 2nd column or 1.0."""
    with _open_csv(path) as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise PlotError(f"CSV file is empty: {path}")
    start = 0
    try:
        float(rows[0][0])
    except (ValueError, IndexError):
        start = 1
    result = []
    for i, row in enumerate(rows[start:], start + 1):
        if not row or not row[0].strip():
            continue
        try:
            v = float(row[0])
            w = float(row[1]) if len(row) > 1 and row[1].strip() else 1.0
            result.append((v, w))
        except ValueError as exc:
            raise PlotError(f"Non-numeric value on CSV row {i}: {exc}") from exc
    if not result:
        raise PlotError("No numeric data found in CSV.")
    return result


def load_csv_2col(path: str) -> List[Tuple[float, float, float, float]]:
    with _open_csv(path) as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise PlotError(f"CSV file is empty: {path}")
    start = 0
    try:
        float(rows[0][0])
    except (ValueError, IndexError):
        start = 1
    result = []
    for i, row in enumerate(rows[start:], start + 1):
        row = [c for c in row if c.strip()]
        if len(row) < 2:
            continue
        try:
            x, y = float(row[0]), float(row[1])
            ex = float(row[2]) if len(row) > 2 else 0.0
            ey = float(row[3]) if len(row) > 3 else 0.0
            result.append((x, y, ex, ey))
        except ValueError as exc:
            raise PlotError(f"Non-numeric value on CSV row {i}: {exc}") from exc
    if not result:
        raise PlotError("No numeric data found in CSV.")
    return result


def load_csv_3col(path: str) -> List[Tuple[float, float, float]]:
    with _open_csv(path) as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        raise PlotError(f"CSV file is empty: {path}")
    start = 0
    try:
        float(rows[0][0])
    except (ValueError, IndexError):
        start = 1
    result = []
    for i, row in enumerate(rows[start:], start + 1):
        row = [c for c in row if c.strip()]
        if len(row) < 2:
            continue
        try:
            x, y = float(row[0]), float(row[1])
            z = float(row[2]) if len(row) > 2 else 1.0
            result.append((x, y, z))
        except ValueError as exc:
            raise PlotError(f"Non-numeric value on CSV row {i}: {exc}") from exc
    if not result:
        raise PlotError("No numeric data found in CSV.")
    return result


# ---------------------------------------------------------------------------
# ROOT rendering engine (pure functions, no GUI)
# ---------------------------------------------------------------------------

def _apply_style() -> None:
    ROOT.gStyle.SetOptStat(0)
    ROOT.gStyle.SetOptTitle(1)
    ROOT.gStyle.SetTitleFont(42, "")
    ROOT.gStyle.SetTitleSize(0.052, "")
    ROOT.gStyle.SetLabelFont(42, "xyz")
    ROOT.gStyle.SetTitleFont(42, "xyz")
    ROOT.gStyle.SetTitleSize(0.048, "xyz")
    ROOT.gStyle.SetLabelSize(0.042, "xyz")
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetFrameLineWidth(1)
    ROOT.gStyle.SetPadLeftMargin(0.13)
    ROOT.gStyle.SetPadBottomMargin(0.12)


@contextlib.contextmanager
def _suppress_root_output():
    sys.stdout.flush()
    sys.stderr.flush()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        os.close(devnull)


def _c_esc(s: str) -> str:
    """Escape a Python string for embedding in a C double-quoted literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _color_to_hex(color_index: int) -> str:
    """Convert a ROOT color index to '#rrggbb'."""
    tc = ROOT.gROOT.GetColor(int(color_index))
    if not tc:
        return "#000000"
    return "#{:02x}{:02x}{:02x}".format(
        int(tc.GetRed() * 255), int(tc.GetGreen() * 255), int(tc.GetBlue() * 255)
    )


def _get_1d_data(series: SeriesData) -> List[Tuple[float, float]]:
    if series.use_csv:
        if not series.csv_path:
            raise PlotError("No CSV file selected.")
        return load_csv_1d(series.csv_path)
    return parse_1d_values(series.raw_text)


def _get_2col_data(series: SeriesData) -> List[Tuple[float, float, float, float]]:
    if series.use_csv:
        if not series.csv_path:
            raise PlotError("No CSV file selected.")
        return load_csv_2col(series.csv_path)
    return parse_2col_values(series.raw_text)


def _get_3col_data(series: SeriesData) -> List[Tuple[float, float, float]]:
    if series.use_csv:
        if not series.csv_path:
            raise PlotError("No CSV file selected.")
        return load_csv_3col(series.csv_path)
    return parse_3col_values(series.raw_text)


def _build_legend(items: list, mode: str) -> ROOT.TLegend:
    n = len(items)
    x1, x2, y2 = 0.65, 0.92, 0.92
    y1 = max(0.60, y2 - n * 0.07)
    leg = ROOT.TLegend(x1, y1, x2, y2)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextFont(42)
    leg.SetTextSize(0.038)
    leg.SetMargin(0.2)          # fixed marker-column fraction — prevents width growth
    for obj, series in items:
        if mode == "h":
            entry_style = "f" if series.fill_style > 0 else "l"
        else:
            entry_style = "lp"
        leg.AddEntry(obj, series.name, entry_style)
    return leg


def _add_latex_overlay(drawn: list, config: PlotConfig) -> None:
    if not config.latex_text.strip():
        return
    pave = ROOT.TPaveText(
        config.latex_x, config.latex_y,
        config.latex_x + 0.50, config.latex_y + 0.065,
        "NDC",
    )
    pave.SetFillStyle(0)
    pave.SetBorderSize(0)
    pave.SetTextAlign(12)
    pave.SetTextFont(42)
    pave.SetTextSize(0.040)
    pave.AddText(config.latex_text)
    pave.Draw()
    drawn.append(pave)


def _draw_th1f(canvas: ROOT.TCanvas, config: PlotConfig, uid: str) -> Tuple[list, List[str]]:
    drawn: list = []
    hists: list = []
    warnings: List[str] = []

    for i, series in enumerate(config.series):
        try:
            values = _get_1d_data(series)
        except PlotError as exc:
            warnings.append(f"'{series.name}': {exc}")
            continue
        if not values:
            warnings.append(f"'{series.name}': no data")
            continue

        auto_range = (config.x_min == config.x_max == 0.0)
        if auto_range:
            vmin = min(v for v, _ in values)
            vmax = max(v for v, _ in values)
            margin = (vmax - vmin) * 0.05 if vmax > vmin else 1.0
            vmin -= margin
            vmax += margin
        else:
            vmin, vmax = config.x_min, config.x_max

        h = ROOT.TH1F(f"rph1f_{uid}_{i}", "", config.n_bins_x, vmin, vmax)
        h.SetDirectory(0)
        for v, w in values:
            h.Fill(v, w)

        h.SetLineColor(series.color_index)
        h.SetLineWidth(series.line_width)
        if series.fill_style > 0:
            h.SetFillColor(series.color_index)
            h.SetFillStyle(series.fill_style)

        hists.append((h, series))

    if not hists:
        raise PlotError("No valid data to plot.")

    global_max = max(h.GetMaximum() for h, _ in hists)
    auto_y = (config.y_min == config.y_max == 0.0)
    y_min = 0.0 if auto_y else config.y_min
    y_max = global_max * 1.20 if auto_y else config.y_max

    h0, s0 = hists[0]
    h0.SetTitle(f"{config.title};{config.x_title};{config.y_title}")
    h0.GetYaxis().SetRangeUser(y_min, y_max)
    h0.Draw(s0.draw_style)
    drawn.append(h0)

    for h, s in hists[1:]:
        h.Draw(f"{s.draw_style} SAME")
        drawn.append(h)

    if config.show_legend:
        leg = _build_legend(hists, mode="h")
        leg.Draw()
        drawn.append(leg)

    _add_latex_overlay(drawn, config)
    canvas.Modified()
    canvas.Update()
    return drawn, warnings


def _draw_tgraph(canvas: ROOT.TCanvas, config: PlotConfig, uid: str) -> Tuple[list, List[str]]:
    drawn: list = []
    mg = ROOT.TMultiGraph()
    mg.SetTitle(f"{config.title};{config.x_title};{config.y_title}")
    graphs: list = []
    warnings: List[str] = []

    for i, series in enumerate(config.series):
        try:
            rows = _get_2col_data(series)
        except PlotError as exc:
            warnings.append(f"'{series.name}': {exc}")
            continue
        if not rows:
            warnings.append(f"'{series.name}': no data")
            continue

        if config.log_y:
            skipped = sum(1 for _, y, _, _ in rows if y <= 0)
            rows = [(x, y, ex, ey) for x, y, ex, ey in rows if y > 0]
            if skipped:
                warnings.append(
                    f"'{series.name}': {skipped} point(s) with y ≤ 0 hidden (log Y)"
                )
        if config.log_x:
            skipped = sum(1 for x, _, _, _ in rows if x <= 0)
            rows = [(x, y, ex, ey) for x, y, ex, ey in rows if x > 0]
            if skipped:
                warnings.append(
                    f"'{series.name}': {skipped} point(s) with x ≤ 0 hidden (log X)"
                )
        if not rows:
            warnings.append(f"'{series.name}': no valid data after log-scale filtering")
            continue

        has_errors = any(ex != 0.0 or ey != 0.0 for _, _, ex, ey in rows)

        n = len(rows)
        if has_errors:
            g = ROOT.TGraphErrors(n)
        else:
            g = ROOT.TGraph(n)

        for j, (x, y, ex, ey) in enumerate(rows):
            g.SetPoint(j, x, y)
            if has_errors:
                g.SetPointError(j, ex, ey)

        g.SetTitle(series.name)
        g.SetMarkerStyle(series.marker_style)
        g.SetMarkerColor(series.color_index)
        g.SetMarkerSize(1.2)
        g.SetLineColor(series.color_index)
        g.SetLineWidth(series.line_width)

        ROOT.SetOwnership(g, False)
        mg.Add(g, series.draw_style)
        graphs.append((g, series))

    if not graphs:
        raise PlotError("No valid data to plot.")

    if config.log_y:
        canvas.SetLogy(1)
    if config.log_x:
        canvas.SetLogx(1)
    mg.Draw("A")

    auto_x = (config.x_min == config.x_max == 0.0)
    auto_y = (config.y_min == config.y_max == 0.0)
    if not auto_x:
        mg.GetXaxis().SetLimits(config.x_min, config.x_max)
    if not auto_y:
        mg.GetYaxis().SetRangeUser(config.y_min, config.y_max)

    drawn.append(mg)

    if config.show_legend:
        leg = _build_legend(graphs, mode="g")
        leg.Draw()
        drawn.append(leg)

    _add_latex_overlay(drawn, config)
    canvas.Modified()
    canvas.Update()
    return drawn, warnings


def _draw_th2f(canvas: ROOT.TCanvas, config: PlotConfig, uid: str) -> Tuple[list, List[str]]:
    drawn: list = []
    series = config.series[0]

    rows = _get_3col_data(series)
    if not rows:
        raise PlotError("No valid data to plot.")

    xs = [r[0] for r in rows]
    ys = [r[1] for r in rows]

    auto_x = (config.x_min == config.x_max == 0.0)
    auto_y = (config.y_min == config.y_max == 0.0)

    if auto_x:
        span_x = max(xs) - min(xs) if max(xs) > min(xs) else 1.0
        xmin = min(xs) - span_x * 0.05
        xmax = max(xs) + span_x * 0.05
    else:
        xmin, xmax = config.x_min, config.x_max

    if auto_y:
        span_y = max(ys) - min(ys) if max(ys) > min(ys) else 1.0
        ymin = min(ys) - span_y * 0.05
        ymax = max(ys) + span_y * 0.05
    else:
        ymin, ymax = config.y_min, config.y_max

    if "COLZ" in config.th2f_draw_option or "CONT" in config.th2f_draw_option:
        canvas.SetRightMargin(0.15)

    h2 = ROOT.TH2F(
        f"rph2f_{uid}",
        f"{config.title};{config.x_title};{config.y_title};{config.z_title}",
        config.n_bins_x, xmin, xmax,
        config.n_bins_y, ymin, ymax,
    )
    h2.SetDirectory(0)
    for x, y, z in rows:
        h2.Fill(x, y, z)

    h2.Draw(config.th2f_draw_option)
    drawn.append(h2)

    _add_latex_overlay(drawn, config)
    canvas.Modified()
    canvas.Update()
    return drawn, []


def _render_to_canvas(config: PlotConfig, uid: str) -> Tuple[ROOT.TCanvas, list, List[str]]:
    canvas = ROOT.TCanvas(f"rp_canvas_{uid}", "", CANVAS_WIDTH, CANVAS_HEIGHT)
    if config.plot_type.startswith("TH1F"):
        drawn, warnings = _draw_th1f(canvas, config, uid)
    elif config.plot_type.startswith("TGraph"):
        drawn, warnings = _draw_tgraph(canvas, config, uid)
    elif config.plot_type.startswith("TH2F"):
        drawn, warnings = _draw_th2f(canvas, config, uid)
    else:
        raise PlotError(f"Unknown plot type: {config.plot_type}")

    if config.log_x:
        canvas.SetLogx(1)
    if config.log_y:
        canvas.SetLogy(1)
    if config.log_z and config.plot_type.startswith("TH2F"):
        canvas.SetLogz(1)
    if config.grid_x:
        canvas.SetGridx(1)
    if config.grid_y:
        canvas.SetGridy(1)

    canvas.Modified()
    canvas.Update()
    return canvas, drawn, warnings


def render_plot(config: PlotConfig) -> Tuple[str, List[str]]:
    """Render config to a temp PNG. Returns (path, warnings). Caller must delete path."""
    uid = uuid.uuid4().hex[:8]
    canvas, drawn, warnings = _render_to_canvas(config, uid)
    tmpf = tempfile.NamedTemporaryFile(suffix=".png", prefix="rootplot_", delete=False)
    tmpf.close()
    with _suppress_root_output():
        canvas.SaveAs(tmpf.name)
    return tmpf.name, warnings


def export_pdf(config: PlotConfig, output_path: str) -> None:
    uid = uuid.uuid4().hex[:8]
    canvas, drawn, _warnings = _render_to_canvas(config, uid)
    with _suppress_root_output():
        canvas.SaveAs(output_path)
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise PlotError(f"ROOT failed to write PDF: {output_path}")


# ---------------------------------------------------------------------------
# C macro generation helpers
# ---------------------------------------------------------------------------

def _macro_gstyle(L: list) -> None:
    L += [
        "  gStyle->SetOptStat(0);",
        "  gStyle->SetOptTitle(1);",
        '  gStyle->SetTitleFont(42, "");',
        '  gStyle->SetTitleSize(0.052, "");',
        '  gStyle->SetLabelFont(42, "xyz");',
        '  gStyle->SetTitleFont(42, "xyz");',
        '  gStyle->SetTitleSize(0.048, "xyz");',
        '  gStyle->SetLabelSize(0.042, "xyz");',
        "  gStyle->SetPadTickX(1);",
        "  gStyle->SetPadTickY(1);",
        "  gStyle->SetPadLeftMargin(0.13);",
        "  gStyle->SetPadBottomMargin(0.12);",
        "",
    ]


def _macro_legend(L: list, var_names: list, mode: str) -> None:
    n = len(var_names)
    y1 = max(0.60, 0.92 - n * 0.07)
    L.append(f"  TLegend *leg = new TLegend(0.65, {y1:.4f}, 0.92, 0.92);")
    L.append("  leg->SetBorderSize(0); leg->SetFillStyle(0);")
    L.append("  leg->SetTextFont(42); leg->SetTextSize(0.038);")
    for vname, series in var_names:
        entry = "f" if (mode == "h" and series.fill_style > 0) else ("l" if mode == "h" else "lp")
        L.append(f'  leg->AddEntry({vname}, "{_c_esc(series.name)}", "{entry}");')
    L.append("  leg->Draw();")


def _macro_th1f(L: list, config: PlotConfig) -> None:
    valid: list = []
    for i, series in enumerate(config.series):
        try:
            values = _get_1d_data(series)
        except PlotError:
            continue
        if values:
            valid.append((i, series, values))

    if not valid:
        L.append("  // No valid series data")
        return

    auto_x = (config.x_min == config.x_max == 0.0)
    if auto_x:
        all_vs = [v for _, _, vals in valid for v, _ in vals]
        vmin = min(all_vs); vmax = max(all_vs)
        margin = (vmax - vmin) * 0.05 if vmax > vmin else 1.0
        xmin, xmax = vmin - margin, vmax + margin
    else:
        xmin, xmax = config.x_min, config.x_max

    nbins = config.n_bins_x
    title_str = f"{_c_esc(config.title)};{_c_esc(config.x_title)};{_c_esc(config.y_title)}"

    var_names = []
    for i, series, values in valid:
        n = len(values)
        vals_s = ", ".join(f"{v:.6g}" for v, _ in values)
        wgts_s = ", ".join(f"{w:.6g}" for _, w in values)
        vn = f"h_{i}"
        L += [
            f"  // Series: {series.name}",
            f"  const int N_{i} = {n};",
            f"  Double_t vals_{i}[] = {{{vals_s}}};",
            f"  Double_t wgts_{i}[] = {{{wgts_s}}};",
            f'  TH1F *{vn} = new TH1F("{_c_esc(series.name)}", "{title_str}",'
            f" {nbins}, {xmin:.6g}, {xmax:.6g});",
            f"  {vn}->SetDirectory(0);",
            f"  for (int j = 0; j < N_{i}; j++) {vn}->Fill(vals_{i}[j], wgts_{i}[j]);",
            f'  {vn}->SetLineColor(TColor::GetColor("{series.color_hex}"));',
            f"  {vn}->SetLineWidth({series.line_width});",
        ]
        if series.fill_style > 0:
            L.append(f'  {vn}->SetFillColor(TColor::GetColor("{series.color_hex}"));')
            L.append(f"  {vn}->SetFillStyle({series.fill_style});")
        draw_opt = series.draw_style if not var_names else f"{series.draw_style} SAME"
        L.append(f'  {vn}->Draw("{draw_opt}");')
        L.append("")
        var_names.append((vn, series))

    if not (config.y_min == config.y_max == 0.0):
        L.append(f"  {var_names[0][0]}->GetYaxis()->SetRangeUser({config.y_min:.6g}, {config.y_max:.6g});")

    if config.show_legend:
        _macro_legend(L, var_names, "h")


def _macro_tgraph(L: list, config: PlotConfig) -> None:
    L.append("  TMultiGraph *mg = new TMultiGraph();")
    L.append(f'  mg->SetTitle("{_c_esc(config.title)};{_c_esc(config.x_title)};{_c_esc(config.y_title)}");')
    L.append("")

    var_names = []
    for i, series in enumerate(config.series):
        try:
            rows = _get_2col_data(series)
        except PlotError:
            continue
        if not rows:
            continue
        n = len(rows)
        xs_s = ", ".join(f"{x:.6g}" for x, _, _, _ in rows)
        ys_s = ", ".join(f"{y:.6g}" for _, y, _, _ in rows)
        exs_s = ", ".join(f"{ex:.6g}" for _, _, ex, _ in rows)
        eys_s = ", ".join(f"{ey:.6g}" for _, _, _, ey in rows)
        vn = f"g_{i}"
        L += [
            f"  // Series: {series.name}",
            f"  const int N_{i} = {n};",
            f"  Double_t x_{i}[] = {{{xs_s}}};",
            f"  Double_t y_{i}[] = {{{ys_s}}};",
            f"  Double_t ex_{i}[] = {{{exs_s}}};",
            f"  Double_t ey_{i}[] = {{{eys_s}}};",
            f"  TGraphErrors *{vn} = new TGraphErrors(N_{i}, x_{i}, y_{i}, ex_{i}, ey_{i});",
            f'  {vn}->SetTitle("{_c_esc(series.name)}");',
            f"  {vn}->SetMarkerStyle({series.marker_style});",
            f'  {vn}->SetMarkerColor(TColor::GetColor("{series.color_hex}"));',
            f"  {vn}->SetMarkerSize(1.2);",
            f'  {vn}->SetLineColor(TColor::GetColor("{series.color_hex}"));',
            f"  {vn}->SetLineWidth({series.line_width});",
            f'  mg->Add({vn}, "{series.draw_style}");',
            "",
        ]
        var_names.append((vn, series))

    if not var_names:
        L.append("  // No valid series data")
        return

    L.append('  mg->Draw("A");')
    if not (config.x_min == config.x_max == 0.0):
        L.append(f"  mg->GetXaxis()->SetLimits({config.x_min:.6g}, {config.x_max:.6g});")
    if not (config.y_min == config.y_max == 0.0):
        L.append(f"  mg->GetYaxis()->SetRangeUser({config.y_min:.6g}, {config.y_max:.6g});")

    if config.show_legend:
        _macro_legend(L, var_names, "g")


def _macro_th2f(L: list, config: PlotConfig) -> None:
    if not config.series:
        return
    series = config.series[0]
    try:
        rows = _get_3col_data(series)
    except PlotError:
        L.append("  // No valid data for TH2F")
        return
    if not rows:
        return

    xs = [r[0] for r in rows]; ys = [r[1] for r in rows]
    auto_x = (config.x_min == config.x_max == 0.0)
    auto_y = (config.y_min == config.y_max == 0.0)
    if auto_x:
        sx = max(xs) - min(xs) if max(xs) > min(xs) else 1.0
        xmin, xmax = min(xs) - sx * 0.05, max(xs) + sx * 0.05
    else:
        xmin, xmax = config.x_min, config.x_max
    if auto_y:
        sy = max(ys) - min(ys) if max(ys) > min(ys) else 1.0
        ymin, ymax = min(ys) - sy * 0.05, max(ys) + sy * 0.05
    else:
        ymin, ymax = config.y_min, config.y_max

    n = len(rows)
    xs_s = ", ".join(f"{r[0]:.6g}" for r in rows)
    ys_s = ", ".join(f"{r[1]:.6g}" for r in rows)
    ws_s = ", ".join(f"{r[2]:.6g}" for r in rows)
    title_str = (f"{_c_esc(config.title)};{_c_esc(config.x_title)};"
                 f"{_c_esc(config.y_title)};{_c_esc(config.z_title)}")

    if "COLZ" in config.th2f_draw_option or "CONT" in config.th2f_draw_option:
        L.append("  c->SetRightMargin(0.15);")
    L += [
        f"  const int N = {n};",
        f"  Double_t x_[] = {{{xs_s}}};",
        f"  Double_t y_[] = {{{ys_s}}};",
        f"  Double_t w_[] = {{{ws_s}}};",
        f'  TH2F *h2 = new TH2F("{_c_esc(series.name)}", "{title_str}",'
        f" {config.n_bins_x}, {xmin:.6g}, {xmax:.6g},"
        f" {config.n_bins_y}, {ymin:.6g}, {ymax:.6g});",
        "  h2->SetDirectory(0);",
        "  for (int j = 0; j < N; j++) h2->Fill(x_[j], y_[j], w_[j]);",
        f'  h2->Draw("{config.th2f_draw_option}");',
        "",
    ]


def export_macro(config: PlotConfig, output_path: str) -> None:
    """Generate a self-contained ROOT C macro that reproduces the current plot."""
    base = os.path.splitext(os.path.basename(output_path))[0]
    fname = re.sub(r"\W+", "_", base) or "plot"
    if fname and fname[0].isdigit():
        fname = "_" + fname

    L: List[str] = [
        f"void {fname}() {{",
        "  // Generated by ROOT Plotter",
    ]
    _macro_gstyle(L)
    L.append(f'  TCanvas *c = new TCanvas("c", "", {CANVAS_WIDTH}, {CANVAS_HEIGHT});')
    L.append("")

    if config.plot_type.startswith("TH1F"):
        _macro_th1f(L, config)
    elif config.plot_type.startswith("TGraph"):
        _macro_tgraph(L, config)
    elif config.plot_type.startswith("TH2F"):
        _macro_th2f(L, config)

    if config.log_x:
        L.append("  c->SetLogx();")
    if config.log_y:
        L.append("  c->SetLogy();")
    if config.log_z and config.plot_type.startswith("TH2F"):
        L.append("  c->SetLogz();")
    if config.grid_x:
        L.append("  c->SetGridx();")
    if config.grid_y:
        L.append("  c->SetGridy();")

    if config.latex_text.strip():
        lx, ly = config.latex_x, config.latex_y
        L += [
            "",
            f'  TPaveText *pave = new TPaveText({lx:.4f}, {ly:.4f},'
            f" {lx + 0.50:.4f}, {ly + 0.065:.4f}, \"NDC\");",
            "  pave->SetFillStyle(0); pave->SetBorderSize(0);",
            "  pave->SetTextAlign(12); pave->SetTextFont(42); pave->SetTextSize(0.040);",
            f'  pave->AddText("{_c_esc(config.latex_text)}");',
            "  pave->Draw();",
        ]

    L += ["", "  c->Modified(); c->Update();", "}"]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# ROOT file export
# ---------------------------------------------------------------------------

def export_root(config: PlotConfig, output_path: str) -> None:
    """Save ROOT objects and the styled canvas to a .root file."""
    uid = uuid.uuid4().hex[:8]
    canvas, drawn, _warnings = _render_to_canvas(config, uid)

    with _suppress_root_output():
        tfile = ROOT.TFile.Open(output_path, "RECREATE")
    if not tfile or tfile.IsZombie():
        raise PlotError(f"Cannot create ROOT file: {output_path}")

    canvas.Write("canvas")
    _data_classes = ("TH1F", "TH2F", "TMultiGraph", "TGraph", "TGraphErrors")
    for obj in drawn:
        if obj.ClassName() in _data_classes:
            obj.Write()

    tfile.Close()
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise PlotError(f"ROOT failed to write file: {output_path}")


# ---------------------------------------------------------------------------
# C macro loading — run macro, introspect canvas, reconstruct PlotConfig
# ---------------------------------------------------------------------------

def _introspect_th1f(prims: list, config: PlotConfig, warnings: list) -> None:
    hists = [p for p in prims if p and p.InheritsFrom("TH1") and not p.InheritsFrom("TH2")]
    if not hists:
        return
    config.plot_type = "TH1F (1D Histogram)"
    h0 = hists[0]
    parts = h0.GetTitle().split(";")
    config.title = parts[0] if parts else ""
    config.x_title = parts[1] if len(parts) > 1 else ""
    config.y_title = parts[2] if len(parts) > 2 else ""
    config.n_bins_x = h0.GetNbinsX()
    config.x_min = h0.GetXaxis().GetXmin()
    config.x_max = h0.GetXaxis().GetXmax()

    for i, h in enumerate(hists):
        lines_data = []
        for b in range(1, h.GetNbinsX() + 1):
            content = h.GetBinContent(b)
            if content != 0.0:
                center = h.GetXaxis().GetBinCenter(b)
                lines_data.append(f"{center:.6g}  {content:.6g}")
        color_idx = int(h.GetLineColor())
        config.series.append(SeriesData(
            name=h.GetName() or f"Series {i + 1}",
            raw_text="\n".join(lines_data),
            color_index=color_idx,
            color_hex=_color_to_hex(color_idx),
            line_width=int(h.GetLineWidth()),
            fill_style=int(h.GetFillStyle()),
            draw_style="HIST",
        ))


def _introspect_tgraph(prims: list, config: PlotConfig, warnings: list) -> None:
    config.plot_type = "TGraph (Scatter/Line)"
    mgs = [p for p in prims if p and p.ClassName() == "TMultiGraph"]
    bare = [p for p in prims
            if p and p.InheritsFrom("TGraph") and p.ClassName() != "TMultiGraph"]

    if mgs:
        mg = mgs[0]
        parts = mg.GetTitle().split(";")
        config.title = parts[0] if parts else ""
        config.x_title = parts[1] if len(parts) > 1 else ""
        config.y_title = parts[2] if len(parts) > 2 else ""
        gl = mg.GetListOfGraphs()
        all_graphs = [gl.At(j) for j in range(gl.GetSize())]
    else:
        all_graphs = bare
        if bare:
            parts = bare[0].GetTitle().split(";")
            config.title = parts[0] if parts else ""
            config.x_title = parts[1] if len(parts) > 1 else ""
            config.y_title = parts[2] if len(parts) > 2 else ""

    for g in all_graphs:
        if not g:
            continue
        n = g.GetN()
        has_err = "Errors" in g.ClassName() or "Asymm" in g.ClassName()
        rows = []
        for j in range(n):
            x, y = g.GetX()[j], g.GetY()[j]
            ex = g.GetEX()[j] if has_err else 0.0
            ey = g.GetEY()[j] if has_err else 0.0
            rows.append(f"{x:.6g}  {y:.6g}  {ex:.6g}  {ey:.6g}")
        color_idx = int(g.GetMarkerColor())
        config.series.append(SeriesData(
            name=g.GetTitle() or f"Series {len(config.series) + 1}",
            raw_text="\n".join(rows),
            color_index=color_idx,
            color_hex=_color_to_hex(color_idx),
            marker_style=int(g.GetMarkerStyle()),
            line_width=int(g.GetLineWidth()),
            draw_style="LP",
        ))


def _introspect_th2f(prims: list, config: PlotConfig, warnings: list) -> None:
    h2s = [p for p in prims if p and p.InheritsFrom("TH2")]
    if not h2s:
        return
    config.plot_type = "TH2F (2D Histogram)"
    h2 = h2s[0]
    parts = h2.GetTitle().split(";")
    config.title = parts[0] if parts else ""
    config.x_title = parts[1] if len(parts) > 1 else ""
    config.y_title = parts[2] if len(parts) > 2 else ""
    config.z_title = parts[3] if len(parts) > 3 else ""
    config.n_bins_x = h2.GetNbinsX()
    config.n_bins_y = h2.GetNbinsY()
    config.x_min = h2.GetXaxis().GetXmin()
    config.x_max = h2.GetXaxis().GetXmax()
    config.y_min = h2.GetYaxis().GetXmin()
    config.y_max = h2.GetYaxis().GetXmax()

    lines_data = []
    for ix in range(1, h2.GetNbinsX() + 1):
        for iy in range(1, h2.GetNbinsY() + 1):
            content = h2.GetBinContent(ix, iy)
            if content != 0.0:
                xc = h2.GetXaxis().GetBinCenter(ix)
                yc = h2.GetYaxis().GetBinCenter(iy)
                lines_data.append(f"{xc:.6g}  {yc:.6g}  {content:.6g}")

    config.series.append(SeriesData(
        name=h2.GetName() or "Data",
        raw_text="\n".join(lines_data),
    ))


def load_macro(path: str) -> Tuple[PlotConfig, List[str]]:
    """Execute a ROOT C macro, inspect its canvas, and return a reconstructed PlotConfig."""
    warnings_list: List[str] = []

    try:
        abs_path = os.path.abspath(path)
        filename = os.path.basename(abs_path)
        funcname = re.sub(r"\W+", "_", os.path.splitext(filename)[0])

        # Read with Python to avoid ROOT's path resolver (which chokes on spaces)
        with open(abs_path, "r") as fh:
            code = fh.read()

        # Rename the entry function with a unique suffix before declaring so that
        # loading the same macro more than once doesn't cause a Cling redefinition
        # error (gInterpreter.Declare adds symbols permanently to the interpreter).
        unique_funcname = f"{funcname}_{uuid.uuid4().hex[:8]}"
        code_patched = re.sub(
            r"\bvoid\s+" + re.escape(funcname) + r"\b",
            f"void {unique_funcname}",
            code,
            count=1,
        )

        n_before = ROOT.gROOT.GetListOfCanvases().GetSize()

        ok = ROOT.gInterpreter.Declare(code_patched)
        if not ok:
            raise PlotError("Macro compilation failed — check the terminal for errors.")

        func = getattr(ROOT, unique_funcname, None)
        if func is None:
            raise PlotError(
                f"Macro loaded but entry function '{funcname}' not found. "
                "Make sure the function name matches the file name."
            )
        func()
    except PlotError:
        raise
    except Exception as exc:
        raise PlotError(f"Failed to execute macro: {exc}") from exc

    clist = ROOT.gROOT.GetListOfCanvases()
    n_after = clist.GetSize()

    canvas = None
    if n_after > n_before:
        # At least one new canvas was created — take the last one
        canvas = clist.At(n_after - 1)
    else:
        # Fallback: look for a canvas named "c" (used by all macros we generate)
        obj = ROOT.gROOT.FindObject("c")
        if obj and obj.InheritsFrom("TCanvas"):
            canvas = obj

    if canvas is None:
        raise PlotError(
            "Macro did not create a TCanvas. "
            "Check the terminal for any compilation errors."
        )

    config = PlotConfig()
    config.log_x = bool(canvas.GetLogx())
    config.log_y = bool(canvas.GetLogy())
    config.log_z = bool(canvas.GetLogz())
    config.series = []

    pl = canvas.GetListOfPrimitives()
    prims = [pl.At(i) for i in range(pl.GetSize())]

    # Legend and LaTeX overlay
    for p in prims:
        if not p:
            continue
        cn = p.ClassName()
        if cn == "TLegend":
            config.show_legend = True
        elif cn == "TPaveText":
            ll = p.GetListOfLines()
            if ll and ll.GetSize() > 0:
                t = ll.At(0)
                if t:
                    config.latex_text = t.GetTitle()
            config.latex_x = float(p.GetX1NDC())
            config.latex_y = float(p.GetY1NDC())

    # Detect plot type and extract data
    for p in prims:
        if not p:
            continue
        try:
            if p.InheritsFrom("TH2"):
                _introspect_th2f(prims, config, warnings_list)
                break
            elif p.InheritsFrom("TH1"):
                _introspect_th1f(prims, config, warnings_list)
                break
            elif p.ClassName() == "TMultiGraph":
                _introspect_tgraph(prims, config, warnings_list)
                break
            elif p.InheritsFrom("TGraph"):
                _introspect_tgraph(prims, config, warnings_list)
                break
        except Exception as exc:
            warnings_list.append(f"Introspection warning: {exc}")

    if not config.series:
        raise PlotError("No recognizable plot objects (TH1, TGraph, TH2) found in macro canvas.")

    return config, warnings_list


# ---------------------------------------------------------------------------
# GUI: ScalableImageLabel
# ---------------------------------------------------------------------------

class ScalableImageLabel(QLabel):
    """QLabel that scales its pixmap to fill available space."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._source_pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(300, 200)
        self._show_placeholder()
        self.setStyleSheet("background: #1e1e1e;")

    def _show_placeholder(self) -> None:
        self.setText("No plot yet.\nConfigure settings and click Update Plot.")
        self.setStyleSheet(
            "background: #1e1e1e; color: #888; font-size: 14px;"
        )

    def set_plot_pixmap(self, pm: QPixmap) -> None:
        self._source_pixmap = pm
        self.setStyleSheet("background: #1e1e1e;")
        self._rescale()

    def clear_plot(self) -> None:
        self._source_pixmap = None
        self.setPixmap(QPixmap())
        self._show_placeholder()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source_pixmap and not self._source_pixmap.isNull():
            self.setText("")
            scaled = self._source_pixmap.scaled(
                self.width() - 4,
                self.height() - 4,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            self.setPixmap(scaled)


# ---------------------------------------------------------------------------
# GUI: SeriesWidget
# ---------------------------------------------------------------------------

class SeriesWidget(QWidget):
    """UI for one data series."""

    changed = pyqtSignal()

    def __init__(self, index: int, plot_type: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._index = index
        self._plot_type = plot_type
        self._color_hex = ROOT_COLORS[index % len(ROOT_COLORS)][2]
        self._color_root = ROOT_COLORS[index % len(ROOT_COLORS)][1]
        self._csv_full_path = ""
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Header: name + remove
        header = QHBoxLayout()
        self.name_edit = QLineEdit(f"Series {self._index + 1}")
        self.name_edit.setPlaceholderText("Series name (for legend)")
        self.name_edit.textChanged.connect(self.changed)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.setFixedWidth(65)
        header.addWidget(QLabel("Name:"))
        header.addWidget(self.name_edit, 1)
        header.addWidget(self.remove_btn)
        layout.addLayout(header)

        # Appearance: color + marker + draw style
        appearance = QHBoxLayout()

        self.color_btn = QPushButton()
        self.color_btn.setFixedSize(60, 24)
        self._update_color_button()
        self.color_btn.clicked.connect(self._pick_color)
        appearance.addWidget(QLabel("Color:"))
        appearance.addWidget(self.color_btn)

        self.marker_combo = QComboBox()
        for label, _ in ROOT_MARKERS:
            self.marker_combo.addItem(label)
        self.marker_combo.currentIndexChanged.connect(self.changed)
        self._marker_label = QLabel("Marker:")
        appearance.addWidget(self._marker_label)
        appearance.addWidget(self.marker_combo)

        self.style_combo = QComboBox()
        self._populate_style_combo()
        self.style_combo.currentIndexChanged.connect(self.changed)
        appearance.addWidget(QLabel("Style:"))
        appearance.addWidget(self.style_combo, 1)
        layout.addLayout(appearance)

        # Line width
        lw_row = QHBoxLayout()
        lw_row.addWidget(QLabel("Line width:"))
        self.lw_spin = QSpinBox()
        self.lw_spin.setRange(1, 8)
        self.lw_spin.setValue(2)
        self.lw_spin.valueChanged.connect(self.changed)
        lw_row.addWidget(self.lw_spin)
        lw_row.addStretch()
        layout.addLayout(lw_row)

        # Data input toggle
        input_toggle = QHBoxLayout()
        self.paste_radio = QRadioButton("Paste data")
        self.csv_radio = QRadioButton("Load CSV")
        self.paste_radio.setChecked(True)
        self.paste_radio.toggled.connect(self._toggle_input_mode)
        input_toggle.addWidget(self.paste_radio)
        input_toggle.addWidget(self.csv_radio)
        input_toggle.addStretch()
        layout.addLayout(input_toggle)

        # Stacked widget: paste | CSV
        self.input_stack = QStackedWidget()

        paste_page = QWidget()
        paste_layout = QVBoxLayout(paste_page)
        paste_layout.setContentsMargins(0, 0, 0, 0)
        self.text_edit = QPlainTextEdit()
        self.text_edit.setPlaceholderText(self._placeholder_text())
        self.text_edit.setFixedHeight(80)
        self.text_edit.textChanged.connect(self.changed)
        paste_layout.addWidget(self.text_edit)
        self.input_stack.addWidget(paste_page)

        csv_page = QWidget()
        csv_layout = QVBoxLayout(csv_page)
        csv_layout.setContentsMargins(0, 0, 0, 0)
        csv_row = QHBoxLayout()
        self.csv_path_label = QLabel("No file selected")
        self.csv_path_label.setWordWrap(True)
        self.csv_browse_btn = QPushButton("Browse...")
        self.csv_browse_btn.clicked.connect(self._browse_csv)
        csv_row.addWidget(self.csv_path_label, 1)
        csv_row.addWidget(self.csv_browse_btn)
        csv_layout.addLayout(csv_row)
        self.input_stack.addWidget(csv_page)

        layout.addWidget(self.input_stack)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        self._refresh_visibility()

    def _placeholder_text(self) -> str:
        if self._plot_type.startswith("TH1F"):
            return "Values (space/comma/newline separated):\n1.5 2.3 4.1 0.8 3.2"
        if self._plot_type.startswith("TGraph"):
            return "x y [±x ±y] — one row per point:\n1.0 2.0\n3.0 4.5 0.1 0.2"
        return "x y [weight] — one row per bin:\n1.0 2.0 3.5\n4.0 5.0"

    def _populate_style_combo(self) -> None:
        self.style_combo.blockSignals(True)
        self.style_combo.clear()
        styles = TH1F_DRAW_STYLES if self._plot_type.startswith("TH1F") else TGRAPH_DRAW_STYLES
        for label, _ in styles:
            self.style_combo.addItem(label)
        self.style_combo.blockSignals(False)

    def _refresh_visibility(self) -> None:
        is_graph = self._plot_type.startswith("TGraph")
        self._marker_label.setVisible(is_graph)
        self.marker_combo.setVisible(is_graph)

    def _toggle_input_mode(self) -> None:
        self.input_stack.setCurrentIndex(0 if self.paste_radio.isChecked() else 1)
        self.changed.emit()

    def _browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load CSV File", "", "CSV Files (*.csv *.tsv *.txt);;All Files (*)"
        )
        if path:
            self._csv_full_path = path
            self.csv_path_label.setText(os.path.basename(path))
            self.csv_path_label.setToolTip(path)
            self.changed.emit()

    def _pick_color(self) -> None:
        initial = QColor(self._color_hex)
        color = QColorDialog.getColor(initial, self, "Pick Series Color")
        if color.isValid():
            self._color_hex = color.name()
            r, g, b = color.redF(), color.greenF(), color.blueF()
            self._color_root = ROOT.TColor.GetColor(r, g, b)
            self._update_color_button()
            self.changed.emit()

    def _update_color_button(self) -> None:
        self.color_btn.setStyleSheet(
            f"background-color: {self._color_hex}; border: 1px solid #555;"
        )

    def refresh_for_plot_type(self, plot_type: str) -> None:
        self._plot_type = plot_type
        self._populate_style_combo()
        self._refresh_visibility()
        self.text_edit.setPlaceholderText(self._placeholder_text())

    def get_series_data(self) -> SeriesData:
        if self._plot_type.startswith("TH1F"):
            style_list = TH1F_DRAW_STYLES
        else:
            style_list = TGRAPH_DRAW_STYLES

        idx = self.style_combo.currentIndex()
        draw_style = style_list[idx][1] if 0 <= idx < len(style_list) else style_list[0][1]
        fill_style = 3004 if "F" in draw_style else 0

        marker_idx = self.marker_combo.currentIndex()
        marker_style = ROOT_MARKERS[marker_idx][1] if 0 <= marker_idx < len(ROOT_MARKERS) else ROOT.kFullCircle

        return SeriesData(
            name=self.name_edit.text() or f"Series {self._index + 1}",
            raw_text=self.text_edit.toPlainText(),
            csv_path=self._csv_full_path,
            use_csv=self.csv_radio.isChecked(),
            color_index=self._color_root,
            color_hex=self._color_hex,
            marker_style=marker_style,
            draw_style=draw_style,
            line_width=self.lw_spin.value(),
            fill_style=fill_style,
        )


# ---------------------------------------------------------------------------
# GUI: PlotSettingsPanel
# ---------------------------------------------------------------------------

class PlotSettingsPanel(QScrollArea):
    """Scrollable left panel with all plot configuration widgets."""

    config_changed = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumWidth(250)

        self._series_widgets: List[SeriesWidget] = []
        self._plot_type = PLOT_TYPES[0]

        inner = QWidget()
        self._main_layout = QVBoxLayout(inner)
        self._main_layout.setSpacing(6)
        self._main_layout.setContentsMargins(6, 6, 6, 6)
        self.setWidget(inner)

        self._build_plot_type_section()
        self._build_titles_section()
        self._build_binning_section()
        self._build_axis_range_section()
        self._build_options_section()
        self._build_latex_section()
        self._build_series_section()
        self._main_layout.addStretch()

        self._add_series()

    # --- Section builders ---

    def _build_plot_type_section(self) -> None:
        grp = QGroupBox("Plot Type")
        lay = QVBoxLayout(grp)
        self.plot_type_combo = QComboBox()
        for pt in PLOT_TYPES:
            self.plot_type_combo.addItem(pt)
        self.plot_type_combo.currentIndexChanged.connect(self._on_plot_type_changed)
        lay.addWidget(self.plot_type_combo)
        self._main_layout.addWidget(grp)

    def _build_titles_section(self) -> None:
        grp = QGroupBox("Titles")
        form_layout = QVBoxLayout(grp)

        def row(label: str, placeholder: str) -> QLineEdit:
            h = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(55)
            edit = QLineEdit()
            edit.setPlaceholderText(placeholder)
            edit.textChanged.connect(self.config_changed)
            h.addWidget(lbl)
            h.addWidget(edit)
            form_layout.addLayout(h)
            return edit

        self.title_edit = row("Title:", "Plot title (supports ROOT LaTeX)")
        self.x_title_edit = row("X axis:", "X axis label")
        self.y_title_edit = row("Y axis:", "Y axis label")
        self.z_title_edit = row("Z axis:", "Z axis label (TH2F only)")
        self._main_layout.addWidget(grp)

    def _build_binning_section(self) -> None:
        self._binning_grp = QGroupBox("Binning")
        lay = QHBoxLayout(self._binning_grp)

        lay.addWidget(QLabel("Bins X:"))
        self.nbins_x_spin = QSpinBox()
        self.nbins_x_spin.setRange(1, 2000)
        self.nbins_x_spin.setValue(50)
        self.nbins_x_spin.valueChanged.connect(self.config_changed)
        lay.addWidget(self.nbins_x_spin)

        self._binsy_label = QLabel("Bins Y:")
        self._binsy_label.setVisible(False)
        lay.addWidget(self._binsy_label)
        self.nbins_y_spin = QSpinBox()
        self.nbins_y_spin.setRange(1, 2000)
        self.nbins_y_spin.setValue(50)
        self.nbins_y_spin.valueChanged.connect(self.config_changed)
        self.nbins_y_spin.setVisible(False)
        lay.addWidget(self.nbins_y_spin)

        self._main_layout.addWidget(self._binning_grp)

    def _build_axis_range_section(self) -> None:
        grp = QGroupBox("Axis Range  (leave 0 / 0 for auto)")
        lay = QVBoxLayout(grp)

        def range_row(label: str):
            h = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(55)
            lo = QDoubleSpinBox()
            lo.setRange(-1e9, 1e9)
            lo.setDecimals(4)
            lo.setValue(0.0)
            hi = QDoubleSpinBox()
            hi.setRange(-1e9, 1e9)
            hi.setDecimals(4)
            hi.setValue(0.0)
            lo.valueChanged.connect(self.config_changed)
            hi.valueChanged.connect(self.config_changed)
            h.addWidget(lbl)
            h.addWidget(lo)
            h.addWidget(QLabel("to"))
            h.addWidget(hi)
            lay.addLayout(h)
            return lo, hi

        self.xmin_spin, self.xmax_spin = range_row("X:")
        self.ymin_spin, self.ymax_spin = range_row("Y:")
        self._main_layout.addWidget(grp)

    def _build_options_section(self) -> None:
        grp = QGroupBox("Options")
        lay = QVBoxLayout(grp)

        row1 = QHBoxLayout()
        self.logx_cb = QCheckBox("Log X")
        self.logy_cb = QCheckBox("Log Y")
        self.logz_cb = QCheckBox("Log Z")
        for cb in (self.logx_cb, self.logy_cb, self.logz_cb):
            cb.toggled.connect(self.config_changed)
            row1.addWidget(cb)
        row1.addStretch()
        lay.addLayout(row1)

        row_grid = QHBoxLayout()
        self.gridx_cb = QCheckBox("Grid X")
        self.gridy_cb = QCheckBox("Grid Y")
        for cb in (self.gridx_cb, self.gridy_cb):
            cb.toggled.connect(self.config_changed)
            row_grid.addWidget(cb)
        row_grid.addStretch()
        lay.addLayout(row_grid)

        row2 = QHBoxLayout()
        self.legend_cb = QCheckBox("Show Legend")
        self.legend_cb.setChecked(True)
        self.legend_cb.toggled.connect(self.config_changed)
        row2.addWidget(self.legend_cb)

        self._th2f_label = QLabel("2D Style:")
        self.th2f_option_combo = QComboBox()
        for opt in TH2F_DRAW_OPTIONS:
            self.th2f_option_combo.addItem(opt)
        self.th2f_option_combo.currentIndexChanged.connect(self.config_changed)
        self._th2f_label.setVisible(False)
        self.th2f_option_combo.setVisible(False)
        row2.addWidget(self._th2f_label)
        row2.addWidget(self.th2f_option_combo)
        row2.addStretch()
        lay.addLayout(row2)

        self._main_layout.addWidget(grp)

    def _build_latex_section(self) -> None:
        grp = QGroupBox("LaTeX Text Overlay")
        lay = QVBoxLayout(grp)

        self.latex_edit = QLineEdit()
        self.latex_edit.setPlaceholderText("#sqrt{s} = 13 TeV,  #it{L} = 139 fb^{-1}")
        self.latex_edit.textChanged.connect(self.config_changed)
        lay.addWidget(self.latex_edit)

        pos_row = QHBoxLayout()
        pos_row.addWidget(QLabel("NDC x:"))
        self.latex_x_spin = QDoubleSpinBox()
        self.latex_x_spin.setRange(0.0, 0.95)
        self.latex_x_spin.setSingleStep(0.01)
        self.latex_x_spin.setValue(0.13)
        self.latex_x_spin.valueChanged.connect(self.config_changed)
        pos_row.addWidget(self.latex_x_spin)
        pos_row.addWidget(QLabel("y:"))
        self.latex_y_spin = QDoubleSpinBox()
        self.latex_y_spin.setRange(0.0, 0.95)
        self.latex_y_spin.setSingleStep(0.01)
        self.latex_y_spin.setValue(0.88)
        self.latex_y_spin.valueChanged.connect(self.config_changed)
        pos_row.addWidget(self.latex_y_spin)
        pos_row.addStretch()
        lay.addLayout(pos_row)

        self._main_layout.addWidget(grp)

    def _build_series_section(self) -> None:
        self._series_grp = QGroupBox("Data Series")
        self._series_layout = QVBoxLayout(self._series_grp)
        self._series_layout.setSpacing(2)

        self._add_series_btn = QPushButton("+ Add Series")
        self._add_series_btn.clicked.connect(self._add_series)

        self._main_layout.addWidget(self._series_grp)
        self._main_layout.addWidget(self._add_series_btn)

    # --- Logic ---

    def _on_plot_type_changed(self) -> None:
        self._plot_type = self.plot_type_combo.currentText()
        is_th2f = self._plot_type.startswith("TH2F")
        is_graph = self._plot_type.startswith("TGraph")

        self.z_title_edit.setVisible(is_th2f)
        self.logz_cb.setVisible(is_th2f)
        self._th2f_label.setVisible(is_th2f)
        self.th2f_option_combo.setVisible(is_th2f)
        self._binsy_label.setVisible(is_th2f)
        self.nbins_y_spin.setVisible(is_th2f)
        # TH2F only makes sense with a single series
        self._add_series_btn.setEnabled(not is_th2f)
        self.legend_cb.setVisible(not is_th2f)
        # Binning group only meaningful for histograms
        self._binning_grp.setVisible(not is_graph)

        for sw in self._series_widgets:
            sw.refresh_for_plot_type(self._plot_type)

        self.config_changed.emit()

    def _add_series(self) -> None:
        idx = len(self._series_widgets)
        sw = SeriesWidget(idx, self._plot_type, self)
        sw.changed.connect(self.config_changed)
        sw.remove_btn.clicked.connect(lambda checked=False, w=sw: self._remove_series(w))
        self._series_widgets.append(sw)
        self._series_layout.addWidget(sw)
        self.config_changed.emit()

    def _remove_series(self, sw: SeriesWidget) -> None:
        if len(self._series_widgets) <= 1:
            return
        self._series_widgets.remove(sw)
        sw.setParent(None)  # type: ignore[arg-type]
        sw.deleteLater()
        self.config_changed.emit()

    def apply_config(self, config: PlotConfig) -> None:
        """Populate all widgets from a PlotConfig (used when loading a macro)."""
        self.blockSignals(True)
        try:
            # Plot type — triggers _on_plot_type_changed (visibility only; config_changed blocked)
            idx = PLOT_TYPES.index(config.plot_type) if config.plot_type in PLOT_TYPES else 0
            self.plot_type_combo.setCurrentIndex(idx)

            self.title_edit.setText(config.title)
            self.x_title_edit.setText(config.x_title)
            self.y_title_edit.setText(config.y_title)
            self.z_title_edit.setText(config.z_title)
            self.nbins_x_spin.setValue(config.n_bins_x)
            self.nbins_y_spin.setValue(config.n_bins_y)
            self.xmin_spin.setValue(config.x_min)
            self.xmax_spin.setValue(config.x_max)
            self.ymin_spin.setValue(config.y_min)
            self.ymax_spin.setValue(config.y_max)
            self.logx_cb.setChecked(config.log_x)
            self.logy_cb.setChecked(config.log_y)
            self.logz_cb.setChecked(config.log_z)
            self.gridx_cb.setChecked(config.grid_x)
            self.gridy_cb.setChecked(config.grid_y)
            self.legend_cb.setChecked(config.show_legend)
            self.latex_edit.setText(config.latex_text)
            self.latex_x_spin.setValue(config.latex_x)
            self.latex_y_spin.setValue(config.latex_y)
            if config.th2f_draw_option in TH2F_DRAW_OPTIONS:
                self.th2f_option_combo.setCurrentIndex(
                    TH2F_DRAW_OPTIONS.index(config.th2f_draw_option)
                )

            # Rebuild series widgets
            for sw in list(self._series_widgets):
                self._series_layout.removeWidget(sw)
                sw.setParent(None)  # type: ignore[arg-type]
                sw.deleteLater()
            self._series_widgets.clear()

            series_list = config.series if config.series else [SeriesData()]
            style_list = (
                TH1F_DRAW_STYLES if config.plot_type.startswith("TH1F") else TGRAPH_DRAW_STYLES
            )
            for series in series_list:
                sw = SeriesWidget(len(self._series_widgets), config.plot_type, self)
                sw.changed.connect(self.config_changed)
                sw.remove_btn.clicked.connect(
                    lambda checked=False, w=sw: self._remove_series(w)
                )
                sw.name_edit.setText(series.name)
                sw.text_edit.setPlainText(series.raw_text)
                sw._color_hex = series.color_hex
                sw._color_root = series.color_index
                sw._update_color_button()
                for mi, (_, mval) in enumerate(ROOT_MARKERS):
                    if mval == series.marker_style:
                        sw.marker_combo.setCurrentIndex(mi)
                        break
                for si, (_, sval) in enumerate(style_list):
                    if sval == series.draw_style:
                        sw.style_combo.setCurrentIndex(si)
                        break
                sw.lw_spin.setValue(series.line_width)
                self._series_widgets.append(sw)
                self._series_layout.addWidget(sw)
        finally:
            self.blockSignals(False)
        self.config_changed.emit()

    def get_config(self) -> PlotConfig:
        series = [sw.get_series_data() for sw in self._series_widgets]
        return PlotConfig(
            plot_type=self.plot_type_combo.currentText(),
            title=self.title_edit.text(),
            x_title=self.x_title_edit.text(),
            y_title=self.y_title_edit.text(),
            z_title=self.z_title_edit.text(),
            n_bins_x=self.nbins_x_spin.value(),
            n_bins_y=self.nbins_y_spin.value(),
            x_min=self.xmin_spin.value(),
            x_max=self.xmax_spin.value(),
            y_min=self.ymin_spin.value(),
            y_max=self.ymax_spin.value(),
            log_x=self.logx_cb.isChecked(),
            log_y=self.logy_cb.isChecked(),
            log_z=self.logz_cb.isChecked(),
            grid_x=self.gridx_cb.isChecked(),
            grid_y=self.gridy_cb.isChecked(),
            show_legend=self.legend_cb.isChecked(),
            latex_text=self.latex_edit.text(),
            latex_x=self.latex_x_spin.value(),
            latex_y=self.latex_y_spin.value(),
            series=series,
            th2f_draw_option=self.th2f_option_combo.currentText(),
        )

    # Override size hints so the QScrollArea never forces the window to widen
    # when series widgets with large minimum-width hints are added.
    def sizeHint(self) -> QSize:
        return QSize(370, super().sizeHint().height())

    def minimumSizeHint(self) -> QSize:
        return QSize(250, super().minimumSizeHint().height())


# ---------------------------------------------------------------------------
# GUI: PlotTab
# ---------------------------------------------------------------------------

class PlotTab(QWidget):
    """One plot tab: settings sidebar + preview image."""

    def __init__(self, tab_index: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tab_index = tab_index
        self._temp_png: Optional[str] = None

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(DEBOUNCE_MS)
        self._debounce.timeout.connect(self._do_render)

        self._build_ui()

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # Toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(6, 4, 6, 4)

        self.update_btn = QPushButton("Update Plot")
        self.update_btn.setFixedHeight(28)
        self.update_btn.clicked.connect(self._do_render)

        self.export_btn = QPushButton("Export PDF...")
        self.export_btn.setFixedHeight(28)
        self.export_btn.clicked.connect(self._export_pdf)
        self.export_btn.setEnabled(False)

        self.export_c_btn = QPushButton("Export .C...")
        self.export_c_btn.setFixedHeight(28)
        self.export_c_btn.clicked.connect(self._export_macro)
        self.export_c_btn.setEnabled(False)

        self.export_root_btn = QPushButton("Export .root...")
        self.export_root_btn.setFixedHeight(28)
        self.export_root_btn.clicked.connect(self._export_root_file)
        self.export_root_btn.setEnabled(False)

        self.load_c_btn = QPushButton("Load .C...")
        self.load_c_btn.setFixedHeight(28)
        self.load_c_btn.clicked.connect(self._load_macro_action)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888; font-style: italic; font-size: 12px;")
        self.status_label.setMinimumWidth(0)   # prevent long warning text from growing the window
        self.status_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

        toolbar.addWidget(self.update_btn)
        toolbar.addWidget(self.export_btn)
        toolbar.addWidget(self.export_c_btn)
        toolbar.addWidget(self.export_root_btn)
        toolbar.addWidget(self.load_c_btn)
        toolbar.addStretch()
        toolbar.addWidget(self.status_label)
        outer.addLayout(toolbar)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        outer.addWidget(sep)

        # Splitter: settings | preview
        splitter = QSplitter(Qt.Horizontal)

        self.settings = PlotSettingsPanel()
        self.settings.config_changed.connect(self._on_config_changed)
        splitter.addWidget(self.settings)

        self.preview = ScalableImageLabel()
        splitter.addWidget(self.preview)

        splitter.setSizes([370, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        outer.addWidget(splitter, 1)

    def _on_config_changed(self) -> None:
        self._debounce.start()

    def _do_render(self) -> None:
        self._debounce.stop()
        config = self.settings.get_config()

        self._delete_temp_png()
        try:
            png_path, warnings = render_plot(config)
            self._temp_png = png_path
            pm = QPixmap(png_path)
            if pm.isNull():
                raise PlotError("ROOT generated an unreadable image.")
            self.preview.set_plot_pixmap(pm)
            self.export_btn.setEnabled(True)
            self.export_c_btn.setEnabled(True)
            self.export_root_btn.setEnabled(True)
            if warnings:
                msg = ";  ".join(warnings)
                self.status_label.setStyleSheet(
                    "color: #cc8800; font-style: italic; font-size: 12px;"
                )
                self.status_label.setText(f"Warning: {msg}")
            else:
                self.status_label.setStyleSheet(
                    "color: #888; font-style: italic; font-size: 12px;"
                )
                self.status_label.setText("")
        except PlotError as exc:
            self.preview.clear_plot()
            self.export_btn.setEnabled(False)
            self.export_c_btn.setEnabled(False)
            self.export_root_btn.setEnabled(False)
            self.status_label.setStyleSheet(
                "color: #cc3333; font-style: italic; font-size: 12px;"
            )
            self.status_label.setText(f"Error: {exc}")
        except Exception as exc:
            self.preview.clear_plot()
            self.export_btn.setEnabled(False)
            self.export_c_btn.setEnabled(False)
            self.export_root_btn.setEnabled(False)
            self.status_label.setStyleSheet(
                "color: #cc3333; font-style: italic; font-size: 12px;"
            )
            self.status_label.setText(f"Unexpected error: {exc}")

    def _export_pdf(self) -> None:
        config = self.settings.get_config()
        default_name = f"plot_{self._tab_index}.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PDF", default_name, "PDF Files (*.pdf);;All Files (*)"
        )
        if not path:
            return
        if not path.lower().endswith(".pdf"):
            path += ".pdf"
        try:
            export_pdf(config, path)
            self.status_label.setText(f"Saved: {os.path.basename(path)}")
        except PlotError as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", f"Unexpected error:\n{exc}")

    def _export_macro(self) -> None:
        config = self.settings.get_config()
        default_name = f"plot_{self._tab_index}.C"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export C Macro", default_name, "C Macros (*.C);;All Files (*)"
        )
        if not path:
            return
        if not path.endswith(".C"):
            path += ".C"
        try:
            export_macro(config, path)
            self.status_label.setStyleSheet("color: #888; font-style: italic; font-size: 12px;")
            self.status_label.setText(f"Saved: {os.path.basename(path)}")
        except PlotError as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", f"Unexpected error:\n{exc}")

    def _export_root_file(self) -> None:
        config = self.settings.get_config()
        default_name = f"plot_{self._tab_index}.root"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export ROOT File", default_name, "ROOT Files (*.root);;All Files (*)"
        )
        if not path:
            return
        if not path.lower().endswith(".root"):
            path += ".root"
        try:
            export_root(config, path)
            self.status_label.setStyleSheet("color: #888; font-style: italic; font-size: 12px;")
            self.status_label.setText(f"Saved: {os.path.basename(path)}")
        except PlotError as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Export Error", f"Unexpected error:\n{exc}")

    def _load_macro_action(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load C Macro", "", "C Macros (*.C *.c *.cxx);;All Files (*)"
        )
        if not path:
            return
        try:
            config, warnings = load_macro(path)
            self.settings.apply_config(config)
            if warnings:
                self.status_label.setStyleSheet(
                    "color: #cc8800; font-style: italic; font-size: 12px;"
                )
                self.status_label.setText("Loaded with warnings: " + ";  ".join(warnings))
            else:
                self.status_label.setStyleSheet(
                    "color: #888; font-style: italic; font-size: 12px;"
                )
                self.status_label.setText(f"Loaded: {os.path.basename(path)}")
            self._do_render()
        except PlotError as exc:
            QMessageBox.warning(self, "Load Failed", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "Load Error", f"Unexpected error:\n{exc}")

    def _delete_temp_png(self) -> None:
        if self._temp_png and os.path.exists(self._temp_png):
            try:
                os.unlink(self._temp_png)
            except OSError:
                pass
        self._temp_png = None

    def cleanup(self) -> None:
        self._delete_temp_png()


# ---------------------------------------------------------------------------
# GUI: RootPlotterApp
# ---------------------------------------------------------------------------

class RootPlotterApp(QMainWindow):
    """Main window with closeable tabs and a '+' button."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ROOT Plotter")
        self.resize(1350, 820)
        self._tab_counter = 0
        self._tabs: List[PlotTab] = []
        self._build_ui()
        self._add_tab()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(0)

        self.tab_widget = QTabWidget()
        self.tab_widget.setTabsClosable(True)
        self.tab_widget.tabCloseRequested.connect(self._close_tab)

        add_btn = QPushButton("+")
        add_btn.setFixedSize(28, 24)
        add_btn.setToolTip("Add new plot tab")
        add_btn.setStyleSheet(
            "QPushButton { font-weight: bold; font-size: 14px; border: none; } "
            "QPushButton:hover { background: #555; }"
        )
        add_btn.clicked.connect(self._add_tab)
        self.tab_widget.setCornerWidget(add_btn, Qt.TopRightCorner)

        layout.addWidget(self.tab_widget)

    def _add_tab(self) -> None:
        self._tab_counter += 1
        tab = PlotTab(self._tab_counter, self)
        idx = self.tab_widget.addTab(tab, f"Plot {self._tab_counter}")
        self.tab_widget.setCurrentIndex(idx)
        self._tabs.append(tab)
        self._update_close_visibility()

    def _close_tab(self, index: int) -> None:
        if self.tab_widget.count() <= 1:
            return
        tab = self.tab_widget.widget(index)
        self.tab_widget.removeTab(index)
        if tab in self._tabs:
            self._tabs.remove(tab)
        if hasattr(tab, "cleanup"):
            tab.cleanup()
        tab.deleteLater()
        self._update_close_visibility()

    def _update_close_visibility(self) -> None:
        only_one = self.tab_widget.count() == 1
        bar = self.tab_widget.tabBar()
        for i in range(self.tab_widget.count()):
            for side in (QTabBar.LeftSide, QTabBar.RightSide):
                btn = bar.tabButton(i, side)
                if btn:
                    btn.setVisible(not only_one)

    def closeEvent(self, event) -> None:
        for tab in self._tabs:
            tab.cleanup()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    _apply_style()

    app = QApplication(sys.argv)
    app.setApplicationName("ROOT Plotter")
    app.setStyle("Fusion")

    window = RootPlotterApp()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
