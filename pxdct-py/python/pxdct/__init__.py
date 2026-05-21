from .pxdct import *
from .pxdct import __doc__

import numpy as np


def dct(x, type=2, *, kind=None, dtype="f64"):
    """Compute a Discrete Cosine Transform.

    Parameters
    ----------
    x : array-like
        Input signal (1-D).
    type : int, optional
        DCT type 1–8. Default 2.
    kind : str, optional
        Explicit specifier e.g. ``"dct4"``. Overrides *type*.
    dtype : ``"f32"`` or ``"f64"``
        Floating-point precision. Default ``"f64"``.

    Returns
    -------
    numpy.ndarray
        Transformed copy.

    Examples
    --------
    >>> import numpy as np, pxdct
    >>> pxdct.dct(np.ones(4), type=2)
    array([4., 0., 0., 0.])
    """
    from .pxdct import dct as _dct
    k = kind if kind is not None else f"dct{type}"
    return _dct(np.asarray(x), kind=k, dtype=dtype)


def dst(x, type=2, *, kind=None, dtype="f64"):
    """Compute a Discrete Sine Transform.

    Parameters
    ----------
    x : array-like
        Input signal (1-D).
    type : int, optional
        DST type 1–8. Default 2.
    kind : str, optional
        Explicit specifier e.g. ``"dst7"``. Overrides *type*.
    dtype : ``"f32"`` or ``"f64"``
        Floating-point precision. Default ``"f64"``.

    Returns
    -------
    numpy.ndarray
        Transformed copy.
    """
    from .pxdct import dct as _dct
    k = kind if kind is not None else f"dst{type}"
    return _dct(np.asarray(x), kind=k, dtype=dtype)


def plan(kind, length, dtype="f64"):
    """Create a reusable :class:`DctPlan`.

    Prefer this over the one-shot :func:`dct` / :func:`dst` when calling
    the same transform size repeatedly (e.g. inside a loop).

    Parameters
    ----------
    kind : str
        Transform specifier e.g. ``"dct2"``, ``"dst7"``.
    length : int
        Number of points.
    dtype : ``"f32"`` or ``"f64"``
        Floating-point precision. Default ``"f64"``.

    Returns
    -------
    DctPlan

    Examples
    --------
    >>> p = pxdct.plan('dct2', 1024)
    >>> p(signal)                     # allocates output
    >>> p(signal, out)                # fills pre-allocated buffer
    """
    from .pxdct import DctPlan
    return DctPlan(kind, length, dtype)


def plan2d(kind_width, width, kind_height=None, height=None, dtype="f64"):
    """Create a reusable :class:`DctPlan2D`.

    Parameters
    ----------
    kind_width : str
        Transform kind for the column (horizontal) dimension.
    width : int
        Number of columns.
    kind_height : str, optional
        Transform kind for the row (vertical) dimension.
        Defaults to *kind_width*.
    height : int, optional
        Number of rows. Defaults to *width* (square).
    dtype : ``"f32"`` or ``"f64"``
        Floating-point precision. Default ``"f64"``.

    Returns
    -------
    DctPlan2D

    Examples
    --------
    >>> p = pxdct.plan2d('dct2', 512)            # 512×512
    >>> p = pxdct.plan2d('dct2', 640, height=480) # rectangular
    """
    from .pxdct import DctPlan, DctPlan2D
    kh = kind_height if kind_height is not None else kind_width
    h  = height      if height      is not None else width
    wp = DctPlan(kind_width, width, dtype)
    hp = DctPlan(kh, h, dtype)
    return DctPlan2D(wp, hp)