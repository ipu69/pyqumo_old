from pyqumo.stats import rel_err


class TextColor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def fmt_err(expected, actual, max_err=.1, min_abs_val=.001):
    """Get error formatted with color.
    """
    err = rel_err(expected=expected, actual=actual)
    if abs(expected) > min_abs_val and err > max_err:
        color = TextColor.FAIL
    else:
        color = TextColor.OKGREEN
    return highlight(f'{err:.4f}', color)


def highlight(s, *colors):
    """Return a string with highlighted value.
    """
    if not isinstance(colors, str):
        colors_str = "".join(colors)
    else:
        colors_str = colors
    return f'{colors_str}{s}{TextColor.ENDC}'


def pluralize(n):
    return '' if n == 1 else 's'
