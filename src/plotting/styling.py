"""TODO: Add docstring."""

# ------------------------------------------------------------------------------------ #
#                                        IMPORTS                                       #
# ------------------------------------------------------------------------------------ #
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------ #
#                                      FONT SETUP                                      #
# ------------------------------------------------------------------------------------ #
FONT_PATH = "fonts/cmuserif.ttf"
FONT_PROP = fm.FontProperties(fname=FONT_PATH)
fm.fontManager.addfont(FONT_PATH)
FONT_FAMILY = FONT_PROP.get_name()
plt.rcParams["font.family"] = FONT_FAMILY
FONT_SCALE = 1.5


# ------------------------------------------------------------------------------------ #
#                                       FUNCTIONS                                      #
# ------------------------------------------------------------------------------------ #
def style_plot(xlabel: str, ylabel: str, font_scale: float = FONT_SCALE) -> None:
    """Apply consistent font styling and layout to plots."""
    plt.xlabel(xlabel, fontsize=font_scale * 10)
    plt.ylabel(ylabel, fontsize=font_scale * 10)
    plt.xticks(fontsize=font_scale * 8)
    plt.yticks(fontsize=font_scale * 8)
    plt.gca().tick_params(
        axis="both",
        direction="in",
        length=font_scale * 4,
        width=font_scale,
    )
    plt.minorticks_on()
    plt.legend(fontsize=font_scale * 6, frameon=False)
    plt.tight_layout()
