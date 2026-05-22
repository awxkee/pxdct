from .pxdct import *
from .pxdct import __doc__

import numpy as np


def dct(x, type=2, *, kind=None, dtype="f64", scaling="none"):
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
    scaling : ``"none"``, ``"scale"``, or ``"ortho"``
        Normalization applied after the raw transform:

        * ``"none"``  – un-normalized textbook output (default).
        * ``"scale"`` – multiply every element by ``sqrt(2 / length)``.
        * ``"ortho"`` – per-type orthonormal scaling; a forward/inverse pair
          at the same length round-trips to the identity.

    Returns
    -------
    numpy.ndarray
        Transformed copy.

    Examples
    --------
    >>> import numpy as np, pxdct
    >>> pxdct.dct(np.ones(4), type=2)
    array([4., 0., 0., 0.])
    >>> pxdct.dct(np.ones(4), type=2, scaling='ortho')
    array([2., 0., 0., 0.])
    """
    from .pxdct import dct as _dct
    k = kind if kind is not None else f"dct{type}"
    return _dct(np.asarray(x), kind=k, dtype=dtype, scaling=scaling)


def dst(x, type=2, *, kind=None, dtype="f64", scaling="none"):
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
    scaling : ``"none"``, ``"scale"``, or ``"ortho"``
        Normalization applied after the raw transform. See :func:`dct`
        for details. Default ``"none"``.

    Returns
    -------
    numpy.ndarray
        Transformed copy.
    """
    from .pxdct import dct as _dct
    k = kind if kind is not None else f"dst{type}"
    return _dct(np.asarray(x), kind=k, dtype=dtype, scaling=scaling)


def plan(kind, length, dtype="f64", scaling="none"):
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
    scaling : ``"none"``, ``"scale"``, or ``"ortho"``
        Normalization applied after the raw transform. See :func:`dct`
        for details. Default ``"none"``.

    Returns
    -------
    DctPlan

    Examples
    --------
    >>> p = pxdct.plan('dct2', 1024)
    >>> p(signal)                          # allocates output
    >>> p(signal, out)                     # fills pre-allocated buffer
    >>> # ortho-normalised round-trip
    >>> fwd = pxdct.plan('dct2', 1024, scaling='ortho')
    >>> inv = pxdct.plan('dct3', 1024, scaling='ortho')
    >>> np.allclose(inv(fwd(signal)), signal)
    True
    """
    from .pxdct import DctPlan
    return DctPlan(kind, length, dtype, scaling)


def plan2d(kind_width, width, kind_height=None, height=None, dtype="f64", scaling="none"):
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
    scaling : ``"none"``, ``"scale"``, or ``"ortho"``
        Normalization applied to both dimensions. See :func:`dct`
        for details. Default ``"none"``.

    Returns
    -------
    DctPlan2D

    Notes
    -----
    **Output layout:** for performance the final transpose is omitted.
    Input is W×H row-major; output is H×W row-major (transposed).
    To restore W×H order in NumPy: ``arr.reshape(width, height).T.copy()``.

    For a lossless round-trip, construct the inverse plan with axes swapped::

        fwd = pxdct.plan2d('dct2', width, height=height)
        inv = pxdct.plan2d('dct3', height, height=width)  # note: axes swapped
        inv.execute(fwd_output)  # recovers original W×H layout

    Examples
    --------
    >>> p = pxdct.plan2d('dct2', 512)              # 512×512
    >>> p = pxdct.plan2d('dct2', 640, height=480)  # rectangular
    >>> p = pxdct.plan2d('dct2', 8, scaling='ortho')  # ortho
    >>>
    >>> # output is H×W — reshape to recover 2-D array in original orientation
    >>> img_flat = img.ravel().astype('float64')   # W×H row-major
    >>> p.execute(img_flat)                        # now H×W row-major
    >>> coeffs_2d = img_flat.reshape(height, width)  # correct view of output
    """
    from .pxdct import DctPlan, DctPlan2D
    kh = kind_height if kind_height is not None else kind_width
    h  = height      if height      is not None else width
    wp = DctPlan(kind_width, width,  dtype, scaling)
    hp = DctPlan(kh,         h,      dtype, scaling)
    return DctPlan2D(wp, hp)