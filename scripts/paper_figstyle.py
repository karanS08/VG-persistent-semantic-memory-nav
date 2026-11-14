"""Print-size normalisation for the ACRA paper figures (2026-09-29).

Figures are drawn at their exact printed width (ACRA: column 243.9 pt =
3.375 in, text width 505.9 pt = 7.0 in) and included at 100% in main.tex,
so text prints at the same size in every figure: 8 pt axis labels and panel
titles, 7 pt tick labels, legends and annotations. finalize() overrides any
per-call fontsize, removes the figure-level title (captions carry it) and
saves without cropping so the PDF page is exactly width x height."""
import matplotlib.pyplot as plt
from matplotlib.text import Text

COL, FULL = 3.375, 7.0
LABEL_PT, SMALL_PT = 8, 7


def finalize(fig, path, width, height, rect=(0, 0, 1, 1), tight=True):
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    for ax in fig.axes:
        ax.title.set_fontsize(LABEL_PT)
        ax.xaxis.label.set_fontsize(LABEL_PT)
        ax.yaxis.label.set_fontsize(LABEL_PT)
        for t in ax.get_xticklabels() + ax.get_yticklabels():
            t.set_fontsize(SMALL_PT)
        for t in ax.texts:
            t.set_fontsize(SMALL_PT)
        if ax.get_legend():
            for t in ax.get_legend().get_texts():
                t.set_fontsize(SMALL_PT)
    for leg in fig.legends:
        for t in leg.get_texts():
            t.set_fontsize(SMALL_PT)
    for t in fig.texts:
        if isinstance(t, Text):
            t.set_fontsize(SMALL_PT)
    fig.set_size_inches(width, height)
    if tight:  # tight=False keeps manually placed axes (image panels)
        fig.tight_layout(rect=rect)
    with plt.rc_context({"savefig.bbox": "standard"}):  # never crop: page = width x height
        fig.savefig(path)
    plt.close(fig)
    print("wrote", path, f"{width}x{height} in")
