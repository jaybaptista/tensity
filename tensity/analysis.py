import numpy as np
from .legacy import (
    sub_polygon_tmf,
    get_contour_locs,
    get_struct_loc,
    oldPatternRepresentation,
)
from .load_data import load_density, getHeader
from scipy.ndimage.measurements import label
from tqdm import tqdm
import h5py as h5
import jax.numpy as jnp
import pyregion
import matplotlib.pyplot as plt
import pandas as pd

def patternRepresentation(data, threshold):
    """
    A function that returns the Marching Squares Algorithm representation of a 2D image.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array containing the data.
    threshold : float
        The threshold to be used in the Marching Squares Algorithm. This threshold (nu) binarizes the data,
        only drawing contours on pixels that exceed the threshold.
    """
    binary_map = data > threshold
    patterns = (
        1 * binary_map[:-1, :-1]
        + 2 * binary_map[1:, :-1]
        + 4 * binary_map[1:, 1:]
        + 8 * binary_map[:-1, 1:]
    )
    return patterns


def getPixelContributions(data, threshold, returnSideLengths=False, patterns=None):
    """
    A function that returns the pixel contributions to the functional and tensorial contributions.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array containing the data.
    threshold : float
        The threshold to be used in the Marching Squares Algorithm. This threshold (nu) binarizes the data,
        only calculating contributions from pixels exceed the threshold.

    Returns
    -------
    dfs : numpy.ndarray
        A 2D numpy array containing the pixel contributions to the F functional.
    dus : numpy.ndarray
        A 2D numpy array containing the pixel contributions to the U functional.
    dchis : numpy.ndarray
        A 2D numpy array containing the pixel contributions to the Chi functional.
    W_00 : numpy.ndarray
        A 2D numpy array containing the pixel contributions to the 00th component of the W_2_11 tensor.
    W_01 : numpy.ndarray
        A 2D numpy array containing the pixel contributions to the 01th component of the W_2_11 tensor.
    W_10 : numpy.ndarray
        A 2D numpy array containing the pixel contributions to the 10th component of the W_2_11 tensor.
    W_11 : numpy.ndarray
        A 2D numpy array containing the pixel contributions to the 11th component of the W_2_11 tensor.

    """
    data = np.array(data)

    if patterns is None:
        patterns = patternRepresentation(data, threshold)

    # functional contributions
    dfs = np.zeros((data.shape[0] - 1, data.shape[1] - 1))
    dus = np.zeros((data.shape[0] - 1, data.shape[1] - 1))
    dchis = np.zeros((data.shape[0] - 1, data.shape[1] - 1))

    # Tensorial Contributions
    W_00 = np.zeros((data.shape[0] - 1, data.shape[1] - 1))
    W_01 = np.zeros((data.shape[0] - 1, data.shape[1] - 1))
    W_10 = np.zeros((data.shape[0] - 1, data.shape[1] - 1))
    W_11 = np.zeros((data.shape[0] - 1, data.shape[1] - 1))

    # Side identification

    a1 = (data[:-1, :-1] - threshold) / (data[:-1, :-1] - data[1:, :-1])
    a2 = (data[1:, :-1] - threshold) / (data[1:, :-1] - data[1:, 1:])
    a3 = (data[:-1, 1:] - threshold) / (data[:-1, 1:] - data[1:, 1:])
    a4 = (data[:-1, :-1] - threshold) / (data[:-1, :-1] - data[:-1, 1:])

    # Pattern 1

    if 1 in patterns:
        m = (patterns == 1) & (a1 != 0) & (a4 != 0)
        dfs[m] = (0.5 * a1 * a4)[m]
        dus[m] = np.sqrt(a1**2 + a4**2)[m]
        dchis[m] = 0.25

        e0 = a1[m]
        e1 = a4[m]

        W_00[m] = (e0 * e0) / np.sqrt(a1**2 + a4**2)[m]
        W_01[m] = (e0 * e1) / np.sqrt(a1**2 + a4**2)[m]
        W_10[m] = (e0 * e1) / np.sqrt(a1**2 + a4**2)[m]
        W_11[m] = (e1 * e1) / np.sqrt(a1**2 + a4**2)[m]

    if 2 in patterns:
        m = (patterns == 2) & (a1 != 1) & (a2 != 0)
        dfs[m] = (0.5 * (1 - a1) * a2)[m]
        dus[m] = (np.sqrt((1 - a1) ** 2 + a2**2))[m]
        dchis[m] = 0.25

        e0 = (1 - a1)[m]
        e1 = (-a2)[m]

        W_00[m] = (e0 * e0) / np.sqrt((1 - a1) ** 2 + a2**2)[m]
        W_01[m] = (e0 * e1) / np.sqrt((1 - a1) ** 2 + a2**2)[m]
        W_10[m] = (e0 * e1) / np.sqrt((1 - a1) ** 2 + a2**2)[m]
        W_11[m] = (e1 * e1) / np.sqrt((1 - a1) ** 2 + a2**2)[m]

    if 3 in patterns:
        m = patterns == 3
        dfs[m] = (a2 + 0.5 * (a4 - a2))[m]
        dus[m] = np.sqrt(1 + (a4 - a2) ** 2)[m]

        e0 = 1
        e1 = (a4 - a2)[m]

        W_00[m] = (e0 * e0) / np.sqrt(1 + (a4 - a2) ** 2)[m]
        W_01[m] = (e0 * e1) / np.sqrt(1 + (a4 - a2) ** 2)[m]
        W_10[m] = (e0 * e1) / np.sqrt(1 + (a4 - a2) ** 2)[m]
        W_11[m] = (e1 * e1) / np.sqrt(1 + (a4 - a2) ** 2)[m]

    if 4 in patterns:
        m = (patterns == 4) & (a2 != 1) & (a3 != 1)
        dfs[m] = (0.5 * (1 - a2) * (1 - a3))[m]
        dus[m] = np.sqrt((1 - a2) ** 2 + (1 - a3) ** 2)[m]
        dchis[m] = 0.25

        e0 = (-1 * (1 - a3))[m]
        e1 = (-1 * (1 - a2))[m]

        W_00[m] = (e0 * e0) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]
        W_01[m] = (e0 * e1) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]
        W_10[m] = (e0 * e1) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]
        W_11[m] = (e1 * e1) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]

    if 5 in patterns:
        m = (patterns == 5)
        dfs[m] = (1 - 0.5 * (1 - a1) * a2 - 0.5 * a3 * (1 - a4))[m]
        dus[m] = (
            np.sqrt((1 - a1) ** 2 + a2**2)[m] + np.sqrt(a3**2 + (1 - a4) ** 2)[m]
        )
        dchis[m] = 0.5


        m0_1 = (patterns == 5) & (a1 != 1)
        m1_1 = (patterns == 5) & (a2 != 0)

        m0_2 = (patterns == 5) & (a3 != 0)
        m1_2 = (patterns == 5) & (a4 != 1)

        m_mixed_1 = (patterns == 5) & (a1 != 1) & (a2 != 0)
        m_mixed_2 = (patterns == 5) & (a4 != 1) & (a3 != 0)

        e0_1 = -1 * (1 - a1)[m0_1]
        e1_1 = a2[m1_1]

        e0_2 = a3[m0_2]
        e1_2 = -1 * (1 - a4)[m1_2]

        e_mixed_1 = (-1 * (1 - a1) * a2)[m_mixed_1]
        e_mixed_2 = (a3 * -1 * (1 - a4))[m_mixed_2]

        # W_00[m] = (e0_1**2) / np.sqrt((1 - a1) ** 2 + a2**2)[m] + (e0_2**2) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]
        # W_01[m] = (e0_1 * e1_1) / np.sqrt((1 - a1) ** 2 + a2**2)[m] + (e0_2 * e1_2) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]
        # W_10[m] = (e0_1 * e1_1) / np.sqrt((1 - a1) ** 2 + a2**2)[m] + (e0_2 * e1_2) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]
        # W_11[m] = (e1_1**2) / np.sqrt((1 - a1) ** 2 + a2**2)[m] + (e1_2**2) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]

        W_00[m0_1] += (e0_1**2) / np.sqrt((1 - a1) ** 2 + a2**2)[m0_1]
        W_00[m0_2] += (e0_2**2) / np.sqrt(a3**2 + (1 - a4) ** 2)[m0_2]

        W_01[m_mixed_1] += (e_mixed_1) / np.sqrt((1 - a1) ** 2 + a2**2)[m_mixed_1]
        W_01[m_mixed_2] += (e_mixed_2) / np.sqrt(a3**2 + (1 - a4) ** 2)[m_mixed_2]

        W_10[m_mixed_1] += (e_mixed_1) / np.sqrt((1 - a1) ** 2 + a2**2)[m_mixed_1]
        W_10[m_mixed_2] += (e_mixed_2) / np.sqrt(a3**2 + (1 - a4) ** 2)[m_mixed_2]

        W_11[m1_1] += (e1_1**2) / np.sqrt((1 - a1) ** 2 + a2**2)[m1_1]
        W_11[m1_2] += (e1_2**2) / np.sqrt(a3**2 + (1 - a4) ** 2)[m1_2]


    if 6 in patterns:
        m = patterns == 6
        dfs[m] = ((1 - a3) + 0.5 * (a3 - a1))[m]
        dus[m] = np.sqrt(1 + (a3 - a1) ** 2)[m]
        dchis[m] = 0

        e0 = (a3 - a1)[m]
        e1 = -1

        W_00[m] = (e0 * e0) / np.sqrt(1 + (a3 - a1) ** 2)[m]
        W_01[m] = (e0 * e1) / np.sqrt(1 + (a3 - a1) ** 2)[m]
        W_10[m] = (e0 * e1) / np.sqrt(1 + (a3 - a1) ** 2)[m]
        W_11[m] = (e1 * e1) / np.sqrt(1 + (a3 - a1) ** 2)[m]

    if 7 in patterns:
        m = (patterns == 7) & (a3 != 0) & (a4 != 1)
        dfs[m] = (1 - 0.5 * a3 * (1 - a4))[m]
        dus[m] = np.sqrt(a3**2 + (1 - a4) ** 2)[m]
        dchis[m] = -0.25

        e0 = a3[m]
        e1 = -1 * (1 - a4)[m]

        W_00[m] = (e0 * e0) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]
        W_01[m] = (e0 * e1) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]
        W_10[m] = (e0 * e1) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]
        W_11[m] = (e1 * e1) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]

    if 8 in patterns:
        m = (patterns == 8) & (a3 != 0) & (a4 != 1)
        dfs[m] = (0.5 * a3 * (1 - a4))[m]
        dus[m] = np.sqrt(a3**2 + (1 - a4) ** 2)[m]
        dchis[m] = 0.25

        e0 = -a3[m]
        e1 = (1 - a4)[m]

        W_00[m] = (e0 * e0) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]
        W_01[m] = (e0 * e1) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]
        W_10[m] = (e0 * e1) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]
        W_11[m] = (e1 * e1) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]

    if 9 in patterns:
        m = patterns == 9
        dfs[m] = (a1 + 0.5 * (a3 - a1))[m]
        dus[m] = np.sqrt(1 + (a3 - a1) ** 2)[m]

        e0 = (a1 - a3)[m]
        e1 = 1

        W_00[m] = (e0 * e0) / np.sqrt(1 + (a3 - a1) ** 2)[m]
        W_01[m] = (e0 * e1) / np.sqrt(1 + (a3 - a1) ** 2)[m]
        W_10[m] = (e0 * e1) / np.sqrt(1 + (a3 - a1) ** 2)[m]
        W_11[m] = (e1 * e1) / np.sqrt(1 + (a3 - a1) ** 2)[m]

    if 10 in patterns:
        m = patterns == 10
        dfs[m] = (1 - 0.5 * a1 * a4 + 0.5 * (1 - a2) * (1 - a3))[m]
        dus[m] = (np.sqrt(a1**2 + a4**2) + np.sqrt((1 - a2) ** 2 + (1 - a3) ** 2))[
            m
        ]
        dchis[m] = 0.5

        m0_1 = (patterns == 10) & (a1 != 0)
        m1_1 = (patterns == 10) & (a4 != 0)

        m0_2 = (patterns == 10) & (a3 != 1)
        m1_2 = (patterns == 10) & (a2 != 1)

        m_mixed_1 = (patterns == 10) & (a1 != 0) & (a4 != 0)
        m_mixed_2 = (patterns == 10) & (a3 != 1) & (a2 != 1)

        e0_1 = -a1[m0_1]
        e1_1 = -a4[m1_1]

        e0_2 = (1 - a3)[m0_2]
        e1_2 = (1 - a2)[m1_2]

        e_mixed_1 = (a1 * a4)[m_mixed_1]
        e_mixed_2 = ((1 - a3) * (1 - a2))[m_mixed_2]

        # W_00[m] = (e0_1**2) / np.sqrt(a1**2 + a4**2)[m] + (e0_2**2) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]
        # W_01[m] = (e0_1 * e1_1) / np.sqrt(a1**2 + a4**2)[m] + (e0_2 * e1_2) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]
        # W_10[m] = (e0_1 * e1_1) / np.sqrt(a1**2 + a4**2)[m] + (e0_2 * e1_2) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]
        # W_11[m] = (e1_1**2) / np.sqrt(a1**2 + a4**2)[m] + (e1_2**2) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]

        W_00[m0_1] += (e0_1**2) / np.sqrt(a1**2 + a4**2)[m0_1]
        W_00[m0_2] += (e0_2**2) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m0_2]

        W_01[m_mixed_1] += e_mixed_1 / np.sqrt(a1**2 + a4**2)[m_mixed_1]
        W_01[m_mixed_2] += e_mixed_2 / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m_mixed_2]

        W_10[m_mixed_1] += e_mixed_1 / np.sqrt(a1**2 + a4**2)[m_mixed_1]
        W_10[m_mixed_2] += e_mixed_2 / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m_mixed_2]

        W_11[m1_1] += (e1_1**2) / np.sqrt(a1**2 + a4**2)[m1_1]
        W_11[m1_2] += (e1_2**2) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m1_2]

    if 11 in patterns:
        m = (patterns == 11) & (a3 != 1) & (a2 != 1)
        dfs[m] = (1 - 0.5 * (1 - a2) * (1 - a3))[m]
        dus[m] = np.sqrt((1 - a2) ** 2 + (1 - a3) ** 2)[m]
        dchis[m] = -0.25

        e0 = (1 - a3)[m]
        e1 = (1 - a2)[m]

        W_00[m] = (e0 * e0) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]
        W_01[m] = (e0 * e1) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]
        W_10[m] = (e0 * e1) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]
        W_11[m] = (e1 * e1) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]

    if 12 in patterns:
        m = patterns == 12
        dfs[m] = ((1 - a2) + 0.5 * (a2 - a4))[m]
        dus[m] = np.sqrt(1 + (a2 - a4) ** 2)[m]

        e0 = -1
        e1 = (a2 - a4)[m]

        W_00[m] = (e0 * e0) / np.sqrt(1 + (a2 - a4) ** 2)[m]
        W_01[m] = (e0 * e1) / np.sqrt(1 + (a2 - a4) ** 2)[m]
        W_10[m] = (e0 * e1) / np.sqrt(1 + (a2 - a4) ** 2)[m]
        W_11[m] = (e1 * e1) / np.sqrt(1 + (a2 - a4) ** 2)[m]

    if 13 in patterns:
        m = (patterns == 13) & (a1 != 1) & (a2 != 0)
        dfs[m] = (1 - 0.5 * (1 - a1) * a2)[m]
        dus[m] = np.sqrt((1 - a1) ** 2 + a2**2)[m]
        dchis[m] = -0.25

        e0 = -(1 - a1)[m]
        e1 = a2[m]

        W_00[m] = (e0 * e0) / np.sqrt((1 - a1) ** 2 + a2**2)[m]
        W_01[m] = (e0 * e1) / np.sqrt((1 - a1) ** 2 + a2**2)[m]
        W_10[m] = (e0 * e1) / np.sqrt((1 - a1) ** 2 + a2**2)[m]
        W_11[m] = (e1 * e1) / np.sqrt((1 - a1) ** 2 + a2**2)[m]

    if 14 in patterns:
        m = (patterns == 14) & (a1 != 0) & (a4 != 0)
        dfs[m] = (1 - 0.5 * a1 * a4)[m]
        dus[m] = np.sqrt(a1**2 + a4**2)[m]
        dchis[m] = -0.25

        e0 = -a1[m]
        e1 = -a4[m]

        W_00[m] = (e0 * e0) / np.sqrt(a1**2 + a4**2)[m]
        W_01[m] = (e0 * e1) / np.sqrt(a1**2 + a4**2)[m]
        W_10[m] = (e0 * e1) / np.sqrt(a1**2 + a4**2)[m]
        W_11[m] = (e1 * e1) / np.sqrt(a1**2 + a4**2)[m]

    if 15 in patterns:
        m = patterns == 15
        dfs[m] = 1

    if returnSideLengths:
        return dfs, dus, dchis, W_00, W_01, W_10, W_11, a1, a2, a3, a4
    else:
        return dfs, dus, dchis, W_00, W_01, W_10, W_11


def getScalarFunctionals(dxs, jax=False):
    """
    A function that returns the scalar functionals of the data.

    Parameters
    ----------
    dxs : numpy.ndarray
        A 3D numpy array containing the pixel contributions to the scalar functionals [df, du, dchi].
    jax : bool (optional) (default=False)
        A boolean that determines whether to use jax.numpy or numpy. If True, jax.numpy is used.

    Returns
    -------
    scalar_functionals : numpy.ndarray
        A 1D numpy array containing the scalar functionals [f, u, chi].
    """

    # dxs := [df, du, dchi]
    if jax:
        return jnp.nansum(dxs, axis=(1, 2))
    else:
        return np.nansum(dxs, axis=(1, 2))


def getTensorialFunctionals(data, threshold, W_00, W_01, W_10, W_11, jax=False):
    """
    A function that returns the tensorial functionals of the data.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array containing the data.
    threshold : float
        The threshold to be used in the Marching Squares Algorithm. This threshold (nu) binarizes the data,
        only calculating contributions from pixels exceed the threshold.
    W_00 : numpy.ndarray
        A 2D numpy array containing the pixel contributions to the 00th component of the W_2_11 tensor.
    W_01 : numpy.ndarray
        A 2D numpy array containing the pixel contributions to the 01th component of the W_2_11 tensor.
    W_10 : numpy.ndarray
        A 2D numpy array containing the pixel contributions to the 10th component of the W_2_11 tensor.
    W_11 : numpy.ndarray
        A 2D numpy array containing the pixel contributions to the 11th component of the W_2_11 tensor.`
    jax : bool (optional) (default=False)
        A boolean that determines whether to use jax.numpy or numpy. If True, jax.numpy is used.

    Returns
    -------
    alpha : float
        The alpha value (this is the bulk alignment of structures).
    beta : float
        The beta value (this is the anisotropy/shape/ellipticity of individual substructures).
    A : float
        The alignment value (1 - alpha)/(1 - beta). If alpha~beta~0.5, the system is highly aligned.
    """
    W_00_sum = np.nansum(W_00) if not jax else jnp.nansum(W_00)
    W_01_sum = np.nansum(W_01) if not jax else jnp.nansum(W_01)
    W_10_sum = np.nansum(W_10) if not jax else jnp.nansum(W_10)
    W_11_sum = np.nansum(W_11) if not jax else jnp.nansum(W_11)

    evals, vecs = np.linalg.eig(np.array([[W_00_sum, W_01_sum], [W_10_sum, W_11_sum]]))

    if not jax:
        W_00[np.isnan(W_00)] = 0
        W_01[np.isnan(W_01)] = 0
        W_10[np.isnan(W_10)] = 0
        W_11[np.isnan(W_11)] = 0
    else:
        W_00[jnp.isnan(W_00)] = 0
        W_01[jnp.isnan(W_01)] = 0
        W_10[jnp.isnan(W_10)] = 0
        W_11[jnp.isnan(W_11)] = 0

    e_subpoly = sub_polygon_tmf(data, threshold, W_00, W_11, W_01, W_10)

    alpha = (
        np.min(evals) / np.max(evals) if not jax else jnp.min(evals) / jnp.max(evals)
    )
    beta = np.mean(e_subpoly) if not jax else jnp.mean(e_subpoly)
    A = (1 - alpha) / (1 - beta)

    return alpha, beta, A


def getImageEigen(data, threshold=None, W_contributions=None):
    """
    A function that returns the eigenvalues and eigenvectors of the W_2_11 tensor.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D numpy array containing the data.
    threshold : float (optional) (default=None)
        The threshold to be used in the Marching Squares Algorithm. This threshold (nu) binarizes the data,
        only calculating contributions from pixels exceed the threshold.

    Returns
    -------
    evals : numpy.ndarray
        A 1D numpy array containing the eigenvalues of the W_2_11 tensor.
    evecs : numpy.ndarray
        A 2D numpy array containing the eigenvectors of the W_2_11 tensor.
    """

    if W_contributions is None:
        if threshold is None:
            raise ValueError("Threshold must be specified.")

        _, _, _, W_00, W_01, W_10, W_11 = getPixelContributions(data, threshold)
        W_contributions = [W_00, W_01, W_10, W_11]

    W_00 = W_contributions[0]
    W_01 = W_contributions[1]
    W_10 = W_contributions[2]
    W_11 = W_contributions[3]

    W_tensor = np.array(
        [[np.nansum(W_00), np.nansum(W_01)], [np.nansum(W_10), np.nansum(W_11)]]
    )

    T = np.array([[0, 1], [-1, 0]])

    W_tensor = T @ W_tensor @ np.linalg.inv(T)

    evals, evecs = np.linalg.eig(W_tensor)

    idx = evals.argsort()[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    return evals, evecs, W_tensor


def getBinaryMap(data, threshold, sentinel=None):
    bmap = (data > threshold).astype(int)
    if sentinel is not None:
        bmap[bmap == 0] = sentinel
    return bmap


def getAngleMap(
    data, threshold, W_00, W_01, W_10, W_11, complement=False, eigenratio=False
):
    """ """

    angle_map = np.zeros_like(W_00)

    if eigenratio:
        ratio_map = np.zeros_like(W_00)

    # we first convert the data into a binary image depending on the threshold
    binary_img = getBinaryMap(data, threshold)

    # then we identify the sub-polygons or isolated polygons in the image
    # This defines the type of connection between nodes. Below mean all kinds of connections (also diagonals) are a connnection
    structure = np.ones((3, 3)).astype(int)
    labeled, ncomponents = label(binary_img, structure)

    # fix NaNs in the W matrix

    W_00[np.isnan(W_00)] = 0
    W_01[np.isnan(W_01)] = 0
    W_10[np.isnan(W_10)] = 0
    W_11[np.isnan(W_11)] = 0

    print(f"Components identified: {ncomponents}")

    for i in range(1, ncomponents + 1):
        # Get the points associated with the island
        subpoly_pts = get_contour_locs(labeled, i)
        row_inds_f = subpoly_pts[:, 0]
        col_inds_f = subpoly_pts[:, 1]

        # Get the W matrix indices for those items
        W_00_i = W_00[row_inds_f, col_inds_f]
        W_01_i = W_01[row_inds_f, col_inds_f]
        W_10_i = W_10[row_inds_f, col_inds_f]
        W_11_i = W_11[row_inds_f, col_inds_f]

        # and then we do the computation!
        W_00_i = np.sum(W_00_i)
        W_01_i = np.sum(W_01_i)
        W_10_i = np.sum(W_10_i)
        W_11_i = np.sum(W_11_i)

        # then we compute the matrix out of this
        W_i = np.array([[W_00_i, W_01_i], [W_10_i, W_11_i]])

        # then compute eigenvalues ratio for this matrix
        evals, evecs = np.linalg.eig(W_i)

        # sort the eigenvalues, and sort eigenvectors by the eigenvalues
        idx = evals.argsort()[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        v1 = evecs[:, 0]  # longest axis eigenvector
        v2 = evecs[:, 1]

        # check if eigenvectors are positive with the right hand rule,
        # if not, flip the sign

        dotprod_1 = np.dot(v1, [1, 0])

        angle = np.arccos(dotprod_1 / np.linalg.norm(v1))

        if complement:
            dotprod_2 = np.dot(-1 * v1, [1, 0])
            angle = np.min(
                [
                    np.arccos(dotprod_1 / np.linalg.norm(v1)),
                    np.arccos(dotprod_2 / np.linalg.norm(v1)),
                ]
            )

        if eigenratio:
            ratio_map[row_inds_f, col_inds_f] = evals[0] / evals[1]

        angle_map[row_inds_f, col_inds_f] = angle

    if eigenratio:
        return angle_map, ratio_map
    else:
        return angle_map

def getFilamentationMap(
    data, threshold, dfs, dus
):
    """ """

    filament_map = np.zeros_like(dfs)
    binary_img = getBinaryMap(data, threshold)

    structure = np.ones((3, 3)).astype(int)
    labeled, ncomponents = label(binary_img, structure)

    print(f"Components identified: {ncomponents}")

    for i in range(1, ncomponents + 1):
        subpoly_pts = get_contour_locs(labeled, i)
        row_inds_f = subpoly_pts[:, 0]
        col_inds_f = subpoly_pts[:, 1]

        dfs_i = dfs[row_inds_f, col_inds_f]
        dus_i = dus[row_inds_f, col_inds_f]

        f_i = np.sum(dfs_i)
        u_i = np.sum(dus_i)

        filamentarity = (u_i**2 - (2*np.pi*f_i)) / (u_i**2 + (2*np.pi*f_i))
        filament_map[row_inds_f, col_inds_f] = filamentarity
    
    return filament_map

def getComponentFeatures(
    data, threshold, dfs, dus
):
    """ """

    f = []
    u = []
    fil = []

    binary_img = getBinaryMap(data, threshold)

    structure = np.ones((3, 3)).astype(int)
    labeled, ncomponents = label(binary_img, structure)

    print(f"Components identified: {ncomponents}")

    for i in range(1, ncomponents + 1):
        subpoly_pts = get_contour_locs(labeled, i)
        row_inds_f = subpoly_pts[:, 0]
        col_inds_f = subpoly_pts[:, 1]

        dfs_i = dfs[row_inds_f, col_inds_f]
        dus_i = dus[row_inds_f, col_inds_f]

        f_i = np.sum(dfs_i)
        u_i = np.sum(dus_i)

        filamentarity = (u_i**2 - (2*np.pi*f_i)) / (u_i**2 + (2*np.pi*f_i))
        f.append(f_i)
        u.append(u_i)
        fil.append(filamentarity)

    
    df = pd.DataFrame({"f": f, "u": u, "filamentarity": fil})

    return df

def getWTensorPerIsland(data, threshold, tmask, W_00, W_01, W_10, W_11):
    Ws = []

    binary_img = getBinaryMap(data, tmask)
    structure = np.ones((3, 3)).astype(int)
    labeled, ncomponents = label(binary_img, structure)

    for i in range(1, ncomponents + 1):
        subpoly_pts = get_contour_locs(labeled, i)

        row_inds_f = subpoly_pts[:, 0]
        col_inds_f = subpoly_pts[:, 1]

        # all W_ij components associated with that island
        W_00_i = W_00[row_inds_f, col_inds_f]
        W_01_i = W_01[row_inds_f, col_inds_f]
        W_10_i = W_10[row_inds_f, col_inds_f]
        W_11_i = W_11[row_inds_f, col_inds_f]

        # calculate alpha

        W_00_sum = np.nansum(W_00_i)
        W_01_sum = np.nansum(W_01_i)
        W_10_sum = np.nansum(W_10_i)
        W_11_sum = np.nansum(W_11_i)

        W_i = np.array([[W_00_sum, W_01_sum], [W_10_sum, W_11_sum]])

        Ws.append(W_i)

    return Ws


def alignment(data, threshold, tmask, W_00, W_01, W_10, W_11):
    A_j = []
    avg_alignment = 0

    binary_img = getBinaryMap(data, tmask)
    structure = np.ones((3, 3)).astype(int)
    labeled, ncomponents = label(binary_img, structure)

    for i in range(1, ncomponents + 1):
        img_coords = get_struct_loc(labeled, i)
        subpoly_pts = get_contour_locs(labeled, i)

        img_inds_row = img_coords[0, :]
        img_inds_col = img_coords[1, :]

        row_inds_f = subpoly_pts[:, 0]
        col_inds_f = subpoly_pts[:, 1]

        # all W_ij components associated with that island
        W_00_i = W_00[row_inds_f, col_inds_f]
        W_01_i = W_01[row_inds_f, col_inds_f]
        W_10_i = W_10[row_inds_f, col_inds_f]
        W_11_i = W_11[row_inds_f, col_inds_f]

        # calculate alpha

        W_00_sum = np.nansum(W_00_i)
        W_01_sum = np.nansum(W_01_i)
        W_10_sum = np.nansum(W_10_i)
        W_11_sum = np.nansum(W_11_i)

        # this is the alpha parameter for the island
        evals, vecs = np.linalg.eig(
            np.array([[W_00_sum, W_01_sum], [W_10_sum, W_11_sum]])
        )
        alpha_j = np.min(evals) / np.max(evals)
        print("alpha_j: ", alpha_j)

        # calculate the beta parameter mask

        mask = np.zeros_like(W_00)
        mask[row_inds_f, col_inds_f] = 1

        masked_W_00 = W_00 * mask
        masked_W_01 = W_01 * mask
        masked_W_10 = W_10 * mask
        masked_W_11 = W_11 * mask

        mask = np.zeros_like(data)
        mask[img_inds_row, img_inds_col] = 1
        masked_data = data * mask

        e_subpoly = sub_polygon_tmf(
            masked_data, threshold, masked_W_00, masked_W_11, masked_W_01, masked_W_10
        )
        plt.figure()
        plt.imshow(masked_data)
        plt.show()
        beta_j = np.mean(e_subpoly)
        print("beta_j: ", beta_j)

        A = (1 - alpha_j) / (1 - beta_j)
        print("A_j: ", A)

        A_j.append(A)

    avg_alignment = np.mean(A_j)

    return avg_alignment

def getContours(data, threshold):
    pass

#### stuff I throw into Sherlock


def loop(data, thresholds, path="file.hdf5", compress=True):
    f = h5.File(path, "w")

    for t in tqdm(thresholds):
        # dfs[t], dus[t], dchis[t], W_00[t], W_01[t], W_10[t], W_11[t] = getPixelContributions(data, t)
        dfs, dus, dchis, W_00, W_01, W_10, W_11 = getPixelContributions(data, t)
        result = [dfs, dus, dchis, W_00, W_01, W_10, W_11]
        with h5.File(path, "r+") as f:
            if compress:
                f.create_dataset(f"t.{t}", data=result, compression="gzip")
            else:
                f.create_dataset(f"t.{t}", data=result)


######


def processScalarsSherlock(path, write_path, region_path=None, data_path=None):
    """
    A function that processes the scalar functionals of the data given a pixel
    contribution file.

    Parameters
    ----------
    path : str
        The path to the pixel contribution file.
    write_path : str
        The path to the file where the scalar functionals be written.
    region_path : str (optional) (default=None)
        The path to the region file. If None, no region file is used. The region
        file is used to mask the data and only
    """
    f = h5.File(write_path, "w")

    mask = None

    fs = None
    us = None
    chis = None

    with h5.File(path, "r") as f:
        keys = list(f.keys())

        thresholds = np.zeros(len(keys))
        scalars = np.zeros((len(keys), 3))

        for i, key in enumerate(tqdm(keys)):
            thresholds[i] = float(key[2:])

            dxs = None

            dus = f[key][0]
            dfs = f[key][1]
            dchis = f[key][2]

            if region_path is not None:
                reg = pyregion.open(region_path)
                header = getHeader(data_path)
                mask = reg.get_mask(
                    header=getHeader(data_path),
                    shape=(header["NAXIS2"], header["NAXIS1"]),
                )
                dus[~mask] = 0
                dfs[~mask] = 0
                dchis[~mask] = 0

            dxs = np.array([dus, dfs, dchis]).reshape(3, -1)

            scalars[i] = getScalarFunctionals(dxs, jax=True)

        # shape of scalars: (thresholds, 3)

        fs = scalars[:, 0]
        us = scalars[:, 1]
        chis = scalars[:, 2]

    # write everything to disk
    with h5.File(write_path, "r+") as f:
        f.create_dataset("thresholds", data=thresholds)
        f.create_dataset("fs", data=fs)
        f.create_dataset("us", data=us)
        f.create_dataset("chis", data=chis)


def processAngleMap(
    path, write_path, data_path, dtype="herschel"
):
    """
    A function that processes the scalar functionals of the data given a pixel
    contribution file.

    Parameters
    ----------
    path : str
        The path to the pixel contribution file.
    write_path : str
        The path to the file where the scalar functionals be written.
    region_path : str (optional) (default=None)
        The path to the region file. If None, no region file is used. The region
        file is used to mask the data and only
    """

    f = h5.File(write_path, "w")

    data = None

    if type == "planck_herschel":
        data = load_density(data_path, type="planck_herschel")[0]
    else:
        data = load_density(data_path, type="herschel")

    maps = []

    

    with h5.File(path, "r") as f:
        keys = list(f.keys())
        thresholds = np.zeros(len(keys))

        for i, key in enumerate(tqdm(keys)):
            thresholds[i] = float(key[2:])
            map_i = getAngleMap(data, thresholds[i], f[key][3], f[key][4], f[key][5], f[key][6])
            maps.append(map_i)

    with h5.File(write_path, "r+") as f:
        f.create_dataset("thresholds", data=thresholds)
        f.create_dataset("maps", data=maps)
        


def processTensorsSherlock(
    path, write_path, data_path, type="herschel", region_path=None
):
    f = h5.File(write_path, "w")

    data = None

    if type == "planck_herschel":
        data = load_density(data_path, type="planck_herschel")[0]
    else:
        data = load_density(data_path, type="herschel")

    alphas = None
    betas = None
    As = None

    with h5.File(path, "r") as f:
        keys = list(f.keys())
        print("these are the keys: ", keys)

        thresholds = np.zeros(len(keys))
        tensors = np.zeros((len(keys), 3))

        for i, key in enumerate(tqdm(keys)):
            print(key)
            thresholds[i] = float(key[2:])

            dW00 = f[key][3]
            dW01 = f[key][4]
            dW10 = f[key][5]
            dW11 = f[key][6]

            if region_path is not None:
                reg = pyregion.open(region_path)
                header = getHeader(data_path)
                mask = reg.get_mask(
                    header=getHeader(data_path),
                    shape=(header["NAXIS2"], header["NAXIS1"]),
                )
                dW00[~mask] = 0
                dW01[~mask] = 0
                dW10[~mask] = 0
                dW11[~mask] = 0
                data[~mask] = 0

            tensors[i] = getTensorialFunctionals(
                data, thresholds[i], dW00, dW01, dW10, dW11
            )

        # shape of tensors is (thresholds, 3)

        alphas = tensors[:, 0]
        betas = tensors[:, 1]
        As = tensors[:, 2]

    with h5.File(write_path, "r+") as f:
        f.create_dataset("thresholds", data=thresholds)
        f.create_dataset("alphas", data=alphas)
        f.create_dataset("betas", data=betas)
        f.create_dataset("As", data=As)


#########
def getMinkowskiMap(
    path,
    data_path,
    kernel,
    write_path="file.hdf5",
    dtype="herschel",
    stride=10,
    t_stride=5,
):
    wf = h5.File(write_path, "w")

    data = load_density(data_path, type=dtype)

    header = getHeader(data_path)
    shape = np.array([header["NAXIS1"], header["NAXIS2"]])
    buffer = int(kernel / 2)

    xs = np.arange(int(buffer), int(shape[0] - buffer))
    ys = np.arange(int(buffer), int(shape[1] - buffer))

    with h5.File(path, "r") as file:
        keys = list(file.keys())

        all_pixel_counts = np.zeros(shape=(shape[1], shape[0]))
        all_alphas = np.zeros(shape=(len(keys), shape[1], shape[0]))
        all_betas = np.zeros(shape=(len(keys), shape[1], shape[0]))

        for x_i in tqdm(xs[::stride]):
            for y_i in ys[::stride]:
                pix_count = np.zeros_like(all_pixel_counts[0])
                pix_count[y_i - buffer : y_i + buffer, x_i - buffer : x_i + buffer] = 1
                all_pixel_counts += pix_count

                for i, key in enumerate(keys[::t_stride]):
                    thresh = float(key[2:])
                    W_00 = file[key][3]
                    W_01 = file[key][4]
                    W_10 = file[key][5]
                    W_11 = file[key][6]

                    W_00 = W_00[
                        y_i - buffer : y_i + buffer, x_i - buffer : x_i + buffer
                    ]
                    W_01 = W_01[
                        y_i - buffer : y_i + buffer, x_i - buffer : x_i + buffer
                    ]
                    W_10 = W_10[
                        y_i - buffer : y_i + buffer, x_i - buffer : x_i + buffer
                    ]
                    W_11 = W_11[
                        y_i - buffer : y_i + buffer, x_i - buffer : x_i + buffer
                    ]
                    data_i = data[
                        y_i - buffer : y_i + buffer, x_i - buffer : x_i + buffer
                    ]

                    alpha_i, beta_i, _ = getTensorialFunctionals(
                        data_i, thresh, W_00, W_01, W_10, W_11
                    )
                    all_alphas[i, y_i, x_i] = alpha_i
                    all_betas[i, y_i, x_i] = beta_i

        # average over first axis
        all_alphas = np.nanmean(all_alphas, axis=0) / all_pixel_counts
        all_betas = np.nanmean(all_betas, axis=0) / all_pixel_counts

    with h5.File(write_path, "r+") as write_file:
        write_file.create_dataset("alphas", data=all_alphas)
        write_file.create_dataset("betas", data=all_betas)
        write_file.create_dataset("pixel_counts", data=all_pixel_counts)
        write_file.create_dataset(
            "thresholds", data=np.array([float(key[2:]) for key in keys[::t_stride]])
        )
