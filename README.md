# ROOT Plotter

An interactive desktop GUI for producing publication-quality plots with [CERN ROOT](https://root.cern), built with PyQt5.

## Features

- **Plot types**: 1D histogram (TH1F), scatter/line graph (TGraph / TGraphErrors), 2D histogram (TH2F)
- **Multiple series** per plot, each with independent color, marker style, and draw style
- **Data input**: paste numbers directly or load CSV files
- **Axis options**: custom titles, log scale (X / Y / Z), manual range or auto
- **Styling**: legend toggle, LaTeX text overlay with adjustable NDC position
- **PDF export** via ROOT's built-in PDF backend (publication-ready fonts and vector graphics)
- **Multiple tabs**, each with an independent plot
- **Live preview**: plot updates automatically 400 ms after any setting change

## Requirements

- **ROOT** ≥ 6 with PyROOT enabled
- **Python** ≥ 3.9
- **PyQt5** ≥ 5.15

### Installing ROOT

The recommended way is via conda:

```bash
conda install -c conda-forge root
```

Or follow the [official installation guide](https://root.cern/install/).

Verify the installation:

```bash
python -c "import ROOT; print(ROOT.__version__)"
```

### Installing Python dependencies

```bash
pip install -r requirements.txt
```

## Usage

```bash
python root_plotter.py
```

### Data input formats

| Plot type | Paste format |
|-----------|-------------|
| **TH1F** | Space / comma / newline-separated values: `1.5 2.3 4.1 0.8` |
| **TGraph** | One point per line — `x y` or `x y ±x ±y` (errors optional): `1.0 2.0 0.1 0.2` |
| **TH2F** | One point per line — `x y` or `x y weight`: `1.0 2.0 3.5` |

Single-value lines in TGraph and TH2F input are silently skipped.

For CSV files, the first row is treated as a header if non-numeric, and the first 1–2 numeric columns are used automatically.

### LaTeX in titles and overlays

ROOT's LaTeX subset is supported everywhere a text field accepts it:

```
#sqrt{s} = 13 TeV,  #it{L} = 139 fb^{-1}
E_{T}^{miss} [GeV]
#alpha_{s}(M_{Z})
```

## Screenshot

![ROOT Plotter GUI](https://github.com/lmoleri/ROOT_plotter/raw/main/screenshot.png)

*(add a screenshot by saving one as `screenshot.png` in the repo root)*
