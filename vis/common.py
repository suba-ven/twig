from pathlib import Path
import matplotlib.pyplot as plt


def publication_style():
    plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight", "font.size": 10})


def save_figure(figure, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); figure.savefig(path)
