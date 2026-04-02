"""
Voynich Unicode font rendering utilities for Jupyter notebooks.

Usage:
    from voy_font import load_voynich_font, voynich_style, voynich_df

    # Load the font (call once per notebook)
    load_voynich_font()

    # Style a dataframe inline
    voynich_style(df.head())

    # Or use the decorator to create a styled dataframe class
    @voynich_df
    class StarsDF:
        data = stars_unicode_df

    StarsDF.head()
"""

import base64
from pathlib import Path
from IPython.display import HTML, display
import pandas as pd

_FONT_DIR = Path(__file__).resolve().parent.parent / "voynich_fonts/Voynich"
_FONT_PATH = _FONT_DIR / "VoynichUnicode.ttf"
_DEFAULT_FONT_SIZE = "18px"
_FONT_LOADED = False


def load_voynich_font(font_path: Path | str | None = None):
    """Load the Voynich Unicode font into the notebook via embedded base64 CSS.

    Call once per notebook session. Subsequent calls are no-ops.
    """
    global _FONT_LOADED
    if _FONT_LOADED:
        return

    path = Path(font_path) if font_path else _FONT_PATH
    font_b64 = base64.b64encode(path.read_bytes()).decode()
    display(HTML(
        "<style>"
        "@font-face {"
        "  font-family: 'VoynichUnicode';"
        f"  src: url(data:font/ttf;base64,{font_b64}) format('truetype');"
        "}"
        "</style>"
    ))
    _FONT_LOADED = True


def voynich_style(df: pd.DataFrame, font_size: str = _DEFAULT_FONT_SIZE):
    """Return a Styler that renders the dataframe in Voynich Unicode font."""
    load_voynich_font()
    return df.style.set_table_attributes(
        f'style="font-family: VoynichUnicode; font-size: {font_size};"'
    )


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
