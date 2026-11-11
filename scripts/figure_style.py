"""Shared figure conventions for the manuscript figures.

* Typeface: Latin Modern Roman, the manuscript body font, so that matplotlib figures match the TikZ/pgfplots
  figures. The OpenType files ship with every TeX distribution; set LM_FONT_DIR to the folder holding
  lmroman10-regular.otf if it is not found automatically (DejaVu Serif is used otherwise, with a warning).
* Colour encodes the adaptation strategy only (Okabe-Ito); backbone scale is shown by marker shape
  (circle: ViT-Large, square: ViT-Giant) and seeds by grey level and line style.
* Panel labels are bold "(a)", "(b)", ... at the top left, as referenced in the captions.
"""
import glob
import os
import warnings
from pathlib import Path

STRATEGY_RGB = {
    "Frozen": (86, 180, 233),      # sky blue
    "Partial FT": (0, 158, 115),   # bluish green
    "LoRA": (230, 159, 0),         # orange
    "Full FT": (0, 114, 178),      # blue
    "Random init.": (120, 120, 120),  # grey, dashed in line plots
    "nnU-Net": (0, 0, 0),             # black, dashed in line plots
}
BACKBONE_MARKER = {"Large": "o", "Giant": "s"}
# seed: (matplotlib grey, pgfplots colour, line style, pgfplots mark)
SEED_GREY = {42: ("#000000", "black", "solid", "*"),
             2026: ("#6e6e6e", "black!57", "dashed", "square*"),
             3407: ("#a5a5a5", "black!35", "dashdotted", "triangle*")}


def strategy_hex(name):
    return "#%02x%02x%02x" % STRATEGY_RGB[name]


def strategy_pgf(name):
    r, g, b = STRATEGY_RGB[name]
    return f"{{rgb,255:red,{r};green,{g};blue,{b}}}"


def _latin_modern_dir():
    env = os.environ.get("LM_FONT_DIR")
    candidates = [env] if env else []
    candidates += glob.glob(str(Path.home() / "AppData/Local/TectonicProject/Tectonic/cache/bundles/data/*"))
    candidates += glob.glob(str(Path.home() / "Library/Caches/Tectonic/bundles/data/*"))
    candidates += glob.glob(str(Path.home() / ".cache/Tectonic/bundles/data/*"))
    for root in ("/usr/share/texmf", "/usr/share/texlive/texmf-dist", "/usr/local/texlive/*/texmf-dist",
                 "C:/texlive/*/texmf-dist", "/Library/TeX/Root/texmf-dist"):
        candidates += glob.glob(root + "/fonts/opentype/public/lm")
    for folder in candidates:
        if folder and (Path(folder) / "lmroman10-regular.otf").is_file():
            return Path(folder)
    return None


def use_matplotlib_style(size=7):
    """Latin Modern text (DejaVu Serif fallback), embedded TrueType-compatible fonts, no top/right spines."""
    import matplotlib as mpl
    from matplotlib import font_manager
    serif = ["DejaVu Serif"]
    folder = _latin_modern_dir()
    if folder is None:
        warnings.warn("Latin Modern not found (set LM_FONT_DIR); figures fall back to DejaVu Serif.")
    else:
        for name in ("lmroman10-regular.otf", "lmroman10-bold.otf", "lmroman10-italic.otf"):
            if (folder / name).is_file():
                font_manager.fontManager.addfont(str(folder / name))
        serif = ["Latin Modern Roman"] + serif
    mpl.rcParams.update({"font.family": "serif", "font.serif": serif, "mathtext.fontset": "cm",
                         "font.size": size, "pdf.fonttype": 42, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})


def panel_label(ax, index, x=-0.02, y=1.03, size=8, **kw):
    """Bold '(a)', '(b)', ... at the top left of an axes."""
    ax.text(x, y, f"({'abcdefgh'[index]})", transform=ax.transAxes, fontsize=size, fontweight="bold",
            ha=kw.pop("ha", "right"), va=kw.pop("va", "bottom"), **kw)
