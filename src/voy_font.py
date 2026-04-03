"""
Voynich Unicode font rendering utilities for Jupyter notebooks.

Usage:
    from voy_font import load_voynich_font, voynich_style, voynich_df, voynich_text, voyn_print

    # Load the font (call once per notebook)
    load_voynich_font()

    # Display a Unicode string in Voynich font
    voynich_text("𰑀𰒁𰑂")

    # Display a list of strings (e.g. from vms_uprep.prepare())
    voynich_text(lines)

    # Print any type directly
    voyn_print("𰑀𰒁𰑂")
    voyn_print(df)

    # Style a dataframe inline
    voynich_style(df.head())

    # Or use the decorator to create a styled dataframe class
    @voynich_df
    class StarsDF:
        data = stars_unicode_df

    StarsDF.head()
"""

import base64
import html as _html
from pathlib import Path
from IPython.display import HTML, display
import pandas as pd

_FONT_DIR = Path(__file__).resolve().parent.parent / "voynich_fonts/Voynich"
_FONT_PATH = _FONT_DIR / "VoynichUnicode.ttf"
_CUSTOM_FONT_PATH = _FONT_DIR / "CustomVoynichUnicode.ttf"
_DEFAULT_FONT_SIZE = "18px"
_FONTS_LOADED: set[str] = set()


def load_voynich_font(font_path: Path | str | None = None, *, custom: bool = True):
    """Load the Voynich Unicode font into the notebook via embedded base64 CSS.

    Call once per notebook session. Subsequent calls for the same font are no-ops.

    Args:
        font_path: Explicit path to a .ttf file. Overrides the custom flag.
        custom: If True (default), load CustomVoynichUnicode.ttf (BMP PUA).
                If False, load the original VoynichUnicode.ttf.
    """
    if font_path:
        path = Path(font_path)
    elif custom:
        path = _CUSTOM_FONT_PATH
    else:
        path = _FONT_PATH

    key = str(path)
    if key in _FONTS_LOADED:
        return

    font_b64 = base64.b64encode(path.read_bytes()).decode()
    display(HTML(
        "<style>"
        "@font-face {"
        "  font-family: 'VoynichUnicode';"
        f"  src: url(data:font/ttf;base64,{font_b64}) format('truetype');"
        "}"
        "</style>"
    ))
    _FONTS_LOADED.add(key)


def voynich_text(
    text: str | list[str],
    font_size: str = _DEFAULT_FONT_SIZE,
    line_break: bool = True,
):
    """Display a Unicode string (or list of strings) rendered in Voynich Unicode font.

    Args:
        text: A single string or list of strings (e.g. output of prepare()).
        font_size: CSS font size.
        line_break: If True, each list item is rendered on its own line.

    Returns:
        An IPython HTML object for notebook display.
    """
    load_voynich_font()
    if isinstance(text, list):
        sep = "<br>" if line_break else " "
        body = sep.join(_html.escape(s) for s in text)
    else:
        body = _html.escape(text)
    return HTML(
        f'<div style="font-family: VoynichUnicode; font-size: {font_size}; '
        f'line-height: 1.6;">{body}</div>'
    )


def voynich_style(df: pd.DataFrame, font_size: str = _DEFAULT_FONT_SIZE):
    """Return a Styler that renders the dataframe in Voynich Unicode font."""
    load_voynich_font()
    return df.style.set_table_attributes(
        f'style="font-family: VoynichUnicode; font-size: {font_size};"'
    )


def voyn_print(
    data: str | list[str] | pd.DataFrame,
    font_size: str = _DEFAULT_FONT_SIZE,
    **kwargs,
):
    """Display a string, list of strings, or DataFrame rendered in Voynich Unicode font.

    Args:
        data: A string, list of strings, or pandas DataFrame.
        font_size: CSS font size.
        **kwargs: Passed to voynich_text (e.g. line_break) when data is a string.
    """
    load_voynich_font()
    if isinstance(data, pd.DataFrame):
        display(voynich_style(data, font_size))
    else:
        display(voynich_text(data, font_size, **kwargs))


def voynich_df(cls):
    """Class decorator that wraps a DataFrame so it renders in Voynich Unicode font.

    Usage:
        @voynich_df
        class Stars:
            data = stars_unicode_df

        Stars            # displays styled head()
        Stars.head(10)   # displays styled head(10)
        Stars.df          # access the raw DataFrame
    """
    raw_df = cls.data

    class VoynichDataFrame:
        df = raw_df

        def __init__(self):
            raise TypeError(f"Use {cls.__name__} directly, do not instantiate")

        @classmethod
        def head(cls, n: int = 5, font_size: str = _DEFAULT_FONT_SIZE):
            return voynich_style(cls.df.head(n), font_size)

        @classmethod
        def tail(cls, n: int = 5, font_size: str = _DEFAULT_FONT_SIZE):
            return voynich_style(cls.df.tail(n), font_size)

        @classmethod
        def sample(cls, n: int = 5, font_size: str = _DEFAULT_FONT_SIZE, **kwargs):
            return voynich_style(cls.df.sample(n, **kwargs), font_size)

        @classmethod
        def style(cls, df: pd.DataFrame | None = None, font_size: str = _DEFAULT_FONT_SIZE):
            """Style the full dataframe, or an arbitrary sub-selection."""
            return voynich_style(df if df is not None else cls.df, font_size)

        @classmethod
        def _repr_html_(cls):
            return cls.head()._repr_html_()

    VoynichDataFrame.__name__ = cls.__name__
    VoynichDataFrame.__qualname__ = cls.__qualname__
    return VoynichDataFrame
