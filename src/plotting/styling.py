"""TODO: Add docstring."""

# ------------------------------------------------------------------------------------ #
#                                        IMPORTS                                       #
# ------------------------------------------------------------------------------------ #
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from cycler import cycler

# ------------------------------------------------------------------------------------ #
#                                       CONSTANTS                                      #
# ------------------------------------------------------------------------------------ #
FONT_PATH = Path(__file__).parent / "fonts" / "cmuserif.ttf"
FONT_SCALE = 1.5


# ------------------------------------------------------------------------------------ #
#                                       FUNCTIONS                                      #
# ------------------------------------------------------------------------------------ #
def set_plot_font_and_scale(
    font_path: str = FONT_PATH,
    font_scale: float = FONT_SCALE,
) -> None:
    """TODO: Add docstring."""
    font_prop = fm.FontProperties(fname=font_path)
    fm.fontManager.addfont(font_path)
    font_family = font_prop.get_name()
    plt.rcParams["font.family"] = font_family

    plt.rcParams["axes.titlesize"] = font_scale * 10
    plt.rcParams["axes.labelsize"] = font_scale * 10
    plt.rcParams["axes.linewidth"] = font_scale

    plt.rcParams["xtick.labelsize"] = font_scale * 8
    plt.rcParams["xtick.major.size"] = font_scale * 4
    plt.rcParams["ytick.labelsize"] = font_scale * 8
    plt.rcParams["ytick.major.size"] = font_scale * 4

    plt.rcParams["legend.fontsize"] = font_scale * 6

    plt.rcParams["lines.markersize"] = font_scale * 3
    plt.rcParams["lines.linewidth"] = font_scale * 2


def set_plot_style(
    fig_size: tuple = (6, 4),
    font_scale: float = FONT_SCALE,
    color_map: str = "Dark2",
) -> None:
    """Set global matplotlib plot style with clean defaults."""
    plt.rcParams["figure.figsize"] = fig_size
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams["savefig.dpi"] = 600

    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.minor.top"] = True
    plt.rcParams["ytick.minor.right"] = True
    plt.rcParams["axes.linewidth"] = font_scale

    plt.rcParams["legend.frameon"] = True

    plt.rcParams["axes.unicode_minus"] = (
        False  # in case font does not support minus sign
    )

    cmap = plt.get_cmap(color_map)
    colors = [cmap(i) for i in range(cmap.N)]
    plt.rcParams["axes.prop_cycle"] = cycler(color=colors)
