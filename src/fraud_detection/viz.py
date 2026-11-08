"""Plot defaults shared by the notebook, reports and dashboard.

Colours follow a colour-blind-validated palette: legit = blue, fraud = orange,
and the reserved status colours for approve / review / block decisions.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

LEGIT = "#2a78d6"
FRAUD = "#eb6834"
THIRD = "#1baf7a"
NEUTRAL = "#8a8985"
CLASS_PALETTE = {0: LEGIT, 1: FRAUD, "Legit": LEGIT, "Fraud": FRAUD}
SERIES = [LEGIT, FRAUD, THIRD, "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
STATUS = {"approve": "#0ca30c", "review": "#fab219", "block": "#d03b3b"}
SEQUENTIAL = "Blues"


def setup_style():
    sns.set_theme(style="whitegrid", context="notebook", palette=SERIES)
    mpl.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 150, "figure.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e8e7e3", "grid.linewidth": 0.8,
        "axes.edgecolor": "#c3c2b7", "axes.titleweight": "bold", "axes.titlesize": 12,
        "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
        "legend.frameon": False, "lines.linewidth": 2,
    })


def finish(ax, title=None, xlabel=None, ylabel=None):
    if title:
        ax.set_title(title, loc="left")
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    return ax
