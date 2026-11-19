# -*- coding: utf-8 -*-

"""

Helper/extra functions used throughout the plotting code.

"""


#### PLOTTING HELPER/EXTRA FUNCTIONS ####


def get_latex_rate_symbol(p, q):
    """ Returns the correct symbol for the corresponding (p, q).
        Returns symbol or expression as a LaTeX math expression.
        Also returns the name of the rate, L_p_q.
    """
    return fr"$\lambda^{{{p} \to {q}}}$", f"L_{p}_{q}"


def map_num_to_letter(num: int) -> str:
    """ Maps 0 to 'A', 1 to 'B', 2 to 'C', up to 25 to 'Z'. """
    if 0 <= num <= 25:
        return chr(65 + num)
    else:
        raise ValueError("Input must be between 0 and 25 inclusive.")


def apply_font_sizes(fig, font_size, tick_font_size):
    """ Given fig, change all its font sizes according to
        font_size and tick_font_size.
    """
    for ax in fig.axes:
        ax.title.set_fontsize(font_size)
        ax.xaxis.label.set_fontsize(font_size)
        ax.yaxis.label.set_fontsize(font_size)
        ax.tick_params(axis="both", labelsize=tick_font_size)

        ax.xaxis.get_offset_text().set_fontsize(tick_font_size)
        ax.yaxis.get_offset_text().set_fontsize(tick_font_size)

        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                text.set_fontsize(font_size)
            legend.get_title().set_fontsize(font_size)


