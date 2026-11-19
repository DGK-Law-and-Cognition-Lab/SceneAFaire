"""One figure style for every figure in the scènes à faire manuscript.

Import it at the top of any figure notebook (in Colab, add Analysis/notebooks to sys.path first):

    import saf_style as S
    S.use()                       # matplotlib defaults, sized for the printed page
    fig, ax = plt.subplots(figsize=S.size(1.0, 3.2))
    ...
    S.save(fig, 'fig_name')       # PDF (vector) and 300-dpi PNG in the manuscript's figures/ folder

The same palette is copied into the R figure file (Analysis/study1_analysis/study1_figures.Rmd),
so the Study 1 figures match. Change a color here and there, nowhere else.

Rules the style encodes (David, Nov 2026):
  - matplotlib's default look: white background, boxed axes, DejaVu Sans, no gridlines;
  - prevalence is always blue and distinctiveness always red; everything else is gray;
  - a light gray marks values that are not reliable, filtered out, or below a threshold;
  - intervals and reference lines are black (reference lines dashed);
  - one sequential colormap (Greys) for heatmaps, with the value written in each cell;
  - plots are upright (categories along the bottom, values going up); no horizontal bars;
  - no figure-level title: the caption in the manuscript carries it (panel titles are fine).
"""
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

PREVALENCE = '#1f77b4'        # tab:blue
DISTINCTIVENESS = '#d62728'   # tab:red
NEUTRAL = '#7f7f7f'           # tab:gray: composite, other scores, single-series bars
LIGHT = '#c7c7c7'             # not reliable, filtered, below threshold
INK = 'black'                 # intervals, reference lines, text
CMAP = 'Greys'                # the only sequential colormap
ERR = dict(ecolor=INK, capsize=3, lw=1.0)   # error-bar keywords for ax.bar / ax.errorbar
TEXT_WIDTH_IN = 6.5           # printed text width of the manuscript (1-inch margins, letter paper)
FONT_SIZE = 9


def use():
    """matplotlib's own defaults, with the font size and output settings for the printed page."""
    plt.rcdefaults()
    mpl.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'legend.fontsize': FONT_SIZE - 1,
        'xtick.labelsize': FONT_SIZE - 1,
        'ytick.labelsize': FONT_SIZE - 1,
        'pdf.fonttype': 42,       # embed TrueType fonts so the PDF text stays editable and searchable
        'ps.fonttype': 42,
        'savefig.facecolor': 'white',
    })


def size(width_frac=1.0, height_in=3.0):
    """(width, height) in inches; width is a fraction of the text width."""
    return (TEXT_WIDTH_IN * width_frac, height_in)


def manuscript_figures_dir(project_root):
    d = Path(project_root) / 'Docs' / 'Writing' / 'manuscript' / 'Scene A Faire Computational' / 'figures'
    d.mkdir(parents=True, exist_ok=True)
    return d


def save(fig, name, out_dir, show=True):
    """Save as PDF (for the manuscript) and as a 300-dpi PNG (for viewing), then show it inline."""
    out_dir = Path(out_dir)
    fig.tight_layout()
    for ext, kw in (('pdf', {}), ('png', {'dpi': 300})):
        fig.savefig(out_dir / f'{name}.{ext}', bbox_inches='tight', facecolor='white', **kw)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_dir / f'{name}.pdf'
