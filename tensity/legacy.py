import numpy as np
import numpy.linalg as LA
from scipy.ndimage.measurements import label
from scipy.spatial import ConvexHull


def oldPatternRepresentation(data, threshold):
    patterns = np.zeros((data.shape[0] - 1, data.shape[1] - 1))

    for i in range(data.shape[0] - 1):
        for j in range(data.shape[1] - 1):
            pattern = 0
            if data[i, j] > threshold:
                pattern += 1
            if data[i + 1, j] > threshold:
                pattern += 2
            if data[i + 1, j + 1] > threshold:
                pattern += 4
            if data[i, j + 1] > threshold:
                pattern += 8

            patterns[i, j] = pattern

    return patterns


### Viraj's Code ###


def estimate_marchingsquare(data, threshold, returnContributions=False):
    """
    Function that computes the SMFs and TMFs for a data 2D array

    Parameters:
    ------------
    data = 2D array, the image in a 2d array format
    threshold = float, the threshold used to compute the MFs. Image above this threshold is considered in computing MFs

    """
    # getting the dimensions of the data to look at so we can loop over each cell in the image
    width = data.shape[0]
    height = data.shape[1]

    # the initial starting values for the 3 Minkowski functionals
    f, u, chi = 0, 0, 0
    # looping over each square/cell in the image

    # I also wish to save in an array the pattern assigned to each cell for later use!
    # generate an empty array with one less length along each axis
    pattern_storage = np.ones((width - 1, height - 1)) * -99

    # we should store the W matrix components of all cells in an array so we can use them in a post process step later
    w_2_11_00_ALL = np.zeros((width - 1, height - 1))
    w_2_11_11_ALL = np.zeros((width - 1, height - 1))
    w_2_11_01_ALL = np.zeros((width - 1, height - 1))
    w_2_11_10_ALL = np.zeros((width - 1, height - 1))

    df_ALL = np.zeros((width - 1, height - 1))
    du_ALL = np.zeros((width - 1, height - 1))
    dchi_ALL = np.zeros((width - 1, height - 1))

    # these are the different matrix components
    # the below will be used to compute the total W matrix components of image
    w_2_11_00_sum = 0
    w_2_11_11_sum = 0
    w_2_11_01_sum = 0
    w_2_11_10_sum = 0

    # TEST

    w_2_11_00_matrix = []
    w_2_11_11_matrix = []
    w_2_11_01_matrix = []
    w_2_11_10_matrix = []

    #

    # looping over each pixel in rows and columns
    for i in range(width - 1):
        for j in range(height - 1):
            # for i in range(width-1):
            #     for j in range(height-1):

            # we evaluate the pattern of the cell currently under consideration
            pattern = 0

            # NaN parsing

            if (
                (np.isnan(data[i, j]))
                or (np.isnan(data[i + 1, j]))
                or (np.isnan(data[i + 1, j + 1]))
                or (np.isnan(data[i, j + 1]))
                or (np.isinf(data[i, j]))
                or (np.isinf(data[i + 1, j]))
                or (np.isinf(data[i + 1, j + 1]))
                or (np.isinf(data[i, j + 1]))
            ):
                # This assumes that we're at the edge of the image, so we don't need to do anything
                # so we skip the cell (the alignment and anisotropy calculations break down severely too :/ )
                pass

            # doing the conditionals to evaluate what marching square pattern to choose for current cell
            if data[i, j] > threshold:
                pattern += 1
            if data[i + 1, j] > threshold:
                pattern += 2
            if data[i + 1, j + 1] > threshold:
                pattern += 4
            if data[i, j + 1] > threshold:
                pattern += 8
            # we save the pattern info to the pattern array
            pattern_storage[i, j] = pattern

            # now going to each kind of pattern
            # for each pattern we identify the edge length that contributes to the overall perimeter of the image
            # and other things that contribute to the other MFs and W matrix for TMFs
            # we will stick with the norm that e0 = {a1, a3} and e1 = {a2, a4} where {e0,e1} is an orthogonal basis.

            # Note that each pattern has a specific direction attached to it as well
            # We account for this direction in the sign of the edge components
            # Refer to Figure 7 of Schroeder Turk et al. 2010 paper on Minkowski Functionals for more info on this
            if pattern == 0:
                # no contribution to boundary of figure from this pattern
                pass
            elif pattern == 1:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, (j + 1)])

                df = 0.5 * a1 * a4
                f = f + df

                du = np.sqrt(a1 * a1 + a4 * a4)
                u = u + du

                dchi = 0.25
                chi = chi + dchi

                # identifying the sides!
                e0 = +a1
                e1 = +a4
                # updating the matrix elements
                # need to multiply by the inverse edge length as well!

                w_2_11_00_i = e0 * e0 / np.sqrt(a1 * a1 + a4 * a4)
                w_2_11_11_i = e1 * e1 / np.sqrt(a1 * a1 + a4 * a4)
                w_2_11_01_i = e0 * e1 / np.sqrt(a1 * a1 + a4 * a4)
                w_2_11_10_i = e0 * e1 / np.sqrt(a1 * a1 + a4 * a4)

                if np.isclose([e0, e1], 0, atol=1e-3).all():
                    w_2_11_00_i = 0
                    w_2_11_11_i = 0
                    w_2_11_01_i = 0
                    w_2_11_10_i = 0

                w_2_11_00_sum += w_2_11_00_i
                w_2_11_11_sum += w_2_11_11_i
                w_2_11_01_sum += w_2_11_01_i
                w_2_11_10_sum += w_2_11_10_i

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 2:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a2 = (data[i + 1, j] - threshold) / (
                    data[i + 1, j] - data[i + 1, (j + 1)]
                )

                df = 0.5 * (1 - a1) * a2
                f = f + df

                du = np.sqrt((1 - a1) * (1 - a1) + a2 * a2)
                u = u + du

                dchi = 0.25
                chi = chi + dchi

                # identifying the sides!
                e0 = 1 - a1
                e1 = -a2
                # updating the matrix elements
                w_2_11_00_i = e0 * e0 / np.sqrt((1 - a1) * (1 - a1) + a2 * a2)
                w_2_11_11_i = e1 * e1 / np.sqrt((1 - a1) * (1 - a1) + a2 * a2)
                w_2_11_01_i = e0 * e1 / np.sqrt((1 - a1) * (1 - a1) + a2 * a2)
                w_2_11_10_i = e0 * e1 / np.sqrt((1 - a1) * (1 - a1) + a2 * a2)

                if np.isclose([e0, e1], 0, atol=1e-3).all():
                    w_2_11_00_i = 0
                    w_2_11_11_i = 0
                    w_2_11_01_i = 0
                    w_2_11_10_i = 0

                w_2_11_00_sum += w_2_11_00_i
                w_2_11_11_sum += w_2_11_11_i
                w_2_11_01_sum += w_2_11_01_i
                w_2_11_10_sum += w_2_11_10_i

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 3:
                a2 = (data[i + 1, j] - threshold) / (
                    data[i + 1, j] - data[i + 1, (j + 1)]
                )
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, (j + 1)])

                df = +a2 + 0.5 * (a4 - a2)
                f = f + df

                du = np.sqrt(1 + (a4 - a2) * (a4 - a2))
                u = u + du

                dchi = 0

                # identifying the sides!
                e0 = 1
                e1 = a4 - a2
                # updating the matrix elements
                w_2_11_00_i = e0 * e0 / np.sqrt(1 + (a4 - a2) * (a4 - a2))
                w_2_11_11_i = e1 * e1 / np.sqrt(1 + (a4 - a2) * (a4 - a2))
                w_2_11_01_i = e0 * e1 / np.sqrt(1 + (a4 - a2) * (a4 - a2))
                w_2_11_10_i = e0 * e1 / np.sqrt(1 + (a4 - a2) * (a4 - a2))

                if np.isclose([e0, e1], 0, atol=1e-3).all():
                    w_2_11_00_i = 0
                    w_2_11_11_i = 0
                    w_2_11_01_i = 0
                    w_2_11_10_i = 0

                w_2_11_00_sum += w_2_11_00_i
                w_2_11_11_sum += w_2_11_11_i
                w_2_11_01_sum += w_2_11_01_i
                w_2_11_10_sum += w_2_11_10_i

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 4:
                a2 = (data[i + 1, j] - threshold) / (
                    data[i + 1, j] - data[i + 1, j + 1]
                )
                a3 = (data[i, j + 1] - threshold) / (
                    data[i, j + 1] - data[i + 1, j + 1]
                )

                df = 0.5 * (1 - a2) * (1 - a3)
                f = f + df

                du = np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                u = u + du

                dchi = 0.25
                chi = chi + dchi

                # identifying the sides!
                e0 = -1 * (1 - a3)
                e1 = -1 * (1 - a2)
                # updating the matrix elements
                w_2_11_00_i = (
                    e0 * e0 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                )
                w_2_11_11_i = (
                    e1 * e1 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                )
                w_2_11_01_i = (
                    e0 * e1 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                )
                w_2_11_10_i = (
                    e0 * e1 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                )

                if np.isclose([e0, e1], 0, atol=1e-3).all():
                    w_2_11_00_i = 0
                    w_2_11_11_i = 0
                    w_2_11_01_i = 0
                    w_2_11_10_i = 0

                w_2_11_00_sum += w_2_11_00_i
                w_2_11_11_sum += w_2_11_11_i
                w_2_11_01_sum += w_2_11_01_i
                w_2_11_10_sum += w_2_11_10_i

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 5:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a2 = (data[i + 1, j] - threshold) / (
                    data[i + 1, j] - data[i + 1, j + 1]
                )
                a3 = (data[i, j + 1] - threshold) / (
                    data[i, j + 1] - data[i + 1, j + 1]
                )
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])

                df = 1 - 0.5 * (1 - a1) * a2 - 0.5 * a3 * (1 - a4)
                f = f + df

                du = np.sqrt((1 - a1) * (1 - a1) + a2 * a2) + np.sqrt(
                    a3 * a3 + (1 - a4) * (1 - a4)
                )
                u = u + du

                dchi = 0.5
                chi = chi + dchi

                # identifying the sides!
                e0_1 = -1 * (1 - a1)
                e1_1 = a2
                e0_2 = a3
                e1_2 = -1 * (1 - a4)
                # updating the matrix elements
                w_2_11_00_i = e0_1 * e0_1 / np.sqrt(
                    (1 - a1) * (1 - a1) + a2 * a2
                ) + e0_2 * e0_2 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                w_2_11_11_i = e1_1 * e1_1 / np.sqrt(
                    (1 - a1) * (1 - a1) + a2 * a2
                ) + e1_2 * e1_2 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                w_2_11_01_i = e0_1 * e1_1 / np.sqrt(
                    (1 - a1) * (1 - a1) + a2 * a2
                ) + e0_2 * e1_2 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                w_2_11_10_i = e0_1 * e1_1 / np.sqrt(
                    (1 - a1) * (1 - a1) + a2 * a2
                ) + e0_2 * e1_2 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))

                if np.isclose([e0, e1], 0, atol=1e-3).all():
                    w_2_11_00_i = 0
                    w_2_11_11_i = 0
                    w_2_11_01_i = 0
                    w_2_11_10_i = 0

                w_2_11_00_sum += w_2_11_00_i
                w_2_11_11_sum += w_2_11_11_i
                w_2_11_01_sum += w_2_11_01_i
                w_2_11_10_sum += w_2_11_10_i

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 6:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a3 = (data[i, j + 1] - threshold) / (
                    data[i, j + 1] - data[i + 1, j + 1]
                )

                df = (1 - a3) + 0.5 * (a3 - a1)
                f = f + df

                du = np.sqrt(1 + (a3 - a1) * (a3 - a1))
                u = u + du

                dchi = 0

                # identifying the sides!
                e0 = a3 - a1
                e1 = -1
                # updating the matrix elements
                w_2_11_00_i = e0 * e0 / np.sqrt(1 + (a3 - a1) * (a3 - a1))
                w_2_11_11_i = e1 * e1 / np.sqrt(1 + (a3 - a1) * (a3 - a1))
                w_2_11_01_i = e0 * e1 / np.sqrt(1 + (a3 - a1) * (a3 - a1))
                w_2_11_10_i = e0 * e1 / np.sqrt(1 + (a3 - a1) * (a3 - a1))

                if np.isclose([e0, e1], 0, atol=1e-3).all():
                    w_2_11_00_i = 0
                    w_2_11_11_i = 0
                    w_2_11_01_i = 0
                    w_2_11_10_i = 0

                w_2_11_00_sum += w_2_11_00_i
                w_2_11_11_sum += w_2_11_11_i
                w_2_11_01_sum += w_2_11_01_i
                w_2_11_10_sum += w_2_11_10_i

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 7:
                a3 = (data[i, j + 1] - threshold) / (
                    data[i, j + 1] - data[i + 1, j + 1]
                )
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])
                df = 1 - 0.5 * a3 * (1 - a4)
                f = f + df

                du = np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                u = u + du

                dchi = -0.25
                chi = chi + dchi
                # identifying the sides!
                e0 = a3
                e1 = -1 * (1 - a4)
                # updating the matrix elements
                w_2_11_00_i = e0 * e0 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                w_2_11_11_i = e1 * e1 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                w_2_11_01_i = e0 * e1 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                w_2_11_10_i = e0 * e1 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))

                if np.isclose([e0, e1], 0, atol=1e-3).all():
                    w_2_11_00_i = 0
                    w_2_11_11_i = 0
                    w_2_11_01_i = 0
                    w_2_11_10_i = 0

                w_2_11_00_sum += w_2_11_00_i
                w_2_11_11_sum += w_2_11_11_i
                w_2_11_01_sum += w_2_11_01_i
                w_2_11_10_sum += w_2_11_10_i

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 8:
                a3 = (data[i, j + 1] - threshold) / (
                    data[i, j + 1] - data[i + 1, j + 1]
                )
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])

                df = 0.5 * a3 * (1 - a4)
                f = f + df

                du = np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                u = u + du

                dchi = 0.25
                chi = chi + dchi

                # identifying the sides!
                e0 = -a3
                e1 = 1 - a4
                # updating the matrix elements
                w_2_11_00_i = e0 * e0 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                w_2_11_11_i = e1 * e1 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                w_2_11_01_i = e0 * e1 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                w_2_11_10_i = e0 * e1 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))

                if np.isclose([e0, e1], 0, atol=1e-3).all():
                    w_2_11_00_i = 0
                    w_2_11_11_i = 0
                    w_2_11_01_i = 0
                    w_2_11_10_i = 0

                w_2_11_00_sum += w_2_11_00_i
                w_2_11_11_sum += w_2_11_11_i
                w_2_11_01_sum += w_2_11_01_i
                w_2_11_10_sum += w_2_11_10_i

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 9:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a3 = (data[i, j + 1] - threshold) / (
                    data[i, j + 1] - data[i + 1, j + 1]
                )

                df = a1 + 0.5 * (a3 - a1)
                f = f + df

                du = np.sqrt(1 + (a3 - a1) * (a3 - a1))
                u = u + du

                dchi = 0

                # identifying the sides!
                e0 = a1 - a3
                e1 = 1
                # updating the matrix elements
                w_2_11_00_i = e0 * e0 / np.sqrt(1 + (a3 - a1) * (a3 - a1))
                w_2_11_11_i = e1 * e1 / np.sqrt(1 + (a3 - a1) * (a3 - a1))
                w_2_11_01_i = e0 * e1 / np.sqrt(1 + (a3 - a1) * (a3 - a1))
                w_2_11_10_i = e0 * e1 / np.sqrt(1 + (a3 - a1) * (a3 - a1))

                if np.isclose([e0, e1], 0, atol=1e-3).all():
                    w_2_11_00_i = 0
                    w_2_11_11_i = 0
                    w_2_11_01_i = 0
                    w_2_11_10_i = 0

                w_2_11_00_sum += w_2_11_00_i
                w_2_11_11_sum += w_2_11_11_i
                w_2_11_01_sum += w_2_11_01_i
                w_2_11_10_sum += w_2_11_10_i

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 10:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a2 = (data[i + 1, j] - threshold) / (
                    data[i + 1, j] - data[i + 1, j + 1]
                )
                a3 = (data[i, j + 1] - threshold) / (
                    data[i, j + 1] - data[i + 1, j + 1]
                )
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])

                df = 1 - 0.5 * a1 * a4 + 0.5 * (1 - a2) * (1 - a3)
                f = f + df

                du = np.sqrt(a1 * a1 + a4 * a4) + np.sqrt(
                    (1 - a2) * (1 - a2) + (1 - a3) * (1 - a3)
                )
                u = u + du

                dchi = 0.5
                chi = chi + dchi

                # identifying the sides!
                e0_1 = -a1
                e1_1 = -a4
                e0_2 = 1 - a3
                e1_2 = 1 - a2
                # updating the matrix elements
                w_2_11_00_i = e0_1 * e0_1 / np.sqrt(
                    a1 * a1 + a4 * a4
                ) + e0_2 * e0_2 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                w_2_11_11_i = e1_1 * e1_1 / np.sqrt(
                    a1 * a1 + a4 * a4
                ) + e1_2 * e1_2 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                w_2_11_01_i = e0_1 * e1_1 / np.sqrt(
                    a1 * a1 + a4 * a4
                ) + e0_2 * e1_2 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                w_2_11_10_i = e0_1 * e1_1 / np.sqrt(
                    a1 * a1 + a4 * a4
                ) + e0_2 * e1_2 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))

                if np.isclose([e0, e1], 0, atol=1e-3).all():
                    w_2_11_00_i = 0
                    w_2_11_11_i = 0
                    w_2_11_01_i = 0
                    w_2_11_10_i = 0

                w_2_11_00_sum += w_2_11_00_i
                w_2_11_11_sum += w_2_11_11_i
                w_2_11_01_sum += w_2_11_01_i
                w_2_11_10_sum += w_2_11_10_i

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 11:
                a2 = (data[i + 1, j] - threshold) / (
                    data[i + 1, j] - data[i + 1, j + 1]
                )
                a3 = (data[i, j + 1] - threshold) / (
                    data[i, j + 1] - data[i + 1, j + 1]
                )

                df = 1 - 0.5 * (1 - a2) * (1 - a3)
                f = f + df

                du = np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                u = u + du

                dchi = -0.25
                chi = chi + dchi

                # identifying the sides!
                e0 = 1 - a3
                e1 = 1 - a2
                # updating the matrix elements
                w_2_11_00_i = (
                    e0 * e0 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                )
                w_2_11_11_i = (
                    e1 * e1 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                )
                w_2_11_01_i = (
                    e0 * e1 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                )
                w_2_11_10_i = (
                    e0 * e1 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                )

                if np.isclose([e0, e1], 0, atol=1e-3).all():
                    w_2_11_00_i = 0
                    w_2_11_11_i = 0
                    w_2_11_01_i = 0
                    w_2_11_10_i = 0

                w_2_11_00_sum += w_2_11_00_i
                w_2_11_11_sum += w_2_11_11_i
                w_2_11_01_sum += w_2_11_01_i
                w_2_11_10_sum += w_2_11_10_i

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                df_ALL[i, j] = df
                du_ALL[i, j] = du
                dchi_ALL[i, j] = dchi

            elif pattern == 12:
                a2 = (data[i + 1, j] - threshold) / (
                    data[i + 1, j] - data[i + 1, j + 1]
                )
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])

                df = (1 - a2) + 0.5 * (a2 - a4)
                f = f + df

                du = np.sqrt(1 + (a2 - a4) * (a2 - a4))
                u = u + du

                dchi = 0

                # identifying the sides!
                e0 = -1
                e1 = a2 - a4
                # updating the matrix elements
                w_2_11_00_i = e0 * e0 / np.sqrt(1 + (a2 - a4) * (a2 - a4))
                w_2_11_11_i = e1 * e1 / np.sqrt(1 + (a2 - a4) * (a2 - a4))
                w_2_11_01_i = e0 * e1 / np.sqrt(1 + (a2 - a4) * (a2 - a4))
                w_2_11_10_i = e0 * e1 / np.sqrt(1 + (a2 - a4) * (a2 - a4))

                if np.isclose([e0, e1], 0, atol=1e-3).all():
                    w_2_11_00_i = 0
                    w_2_11_11_i = 0
                    w_2_11_01_i = 0
                    w_2_11_10_i = 0

                w_2_11_00_sum += w_2_11_00_i
                w_2_11_11_sum += w_2_11_11_i
                w_2_11_01_sum += w_2_11_01_i
                w_2_11_10_sum += w_2_11_10_i

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                df_ALL[i, j] = df
                du_ALL[i, j] = du
                dchi_ALL[i, j] = dchi

            elif pattern == 13:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a2 = (data[i + 1, j] - threshold) / (
                    data[i + 1, j] - data[i + 1, j + 1]
                )

                df = 1 - 0.5 * (1 - a1) * a2
                f = f + df

                du = np.sqrt((1 - a1) * (1 - a1) + a2 * a2)
                u = u + du

                dchi = -0.25
                chi = chi + dchi

                # identifying the sides!
                e0 = -1 * (1 - a1)
                e1 = a2
                # updating the matrix elements

                w_2_11_00_i = e0 * e0 / np.sqrt((1 - a1) * (1 - a1) + a2 * a2)
                w_2_11_11_i = e1 * e1 / np.sqrt((1 - a1) * (1 - a1) + a2 * a2)
                w_2_11_01_i = e0 * e1 / np.sqrt((1 - a1) * (1 - a1) + a2 * a2)
                w_2_11_10_i = e0 * e1 / np.sqrt((1 - a1) * (1 - a1) + a2 * a2)

                if np.isclose([e0, e1], 0, atol=1e-3).all():
                    w_2_11_00_i = 0
                    w_2_11_11_i = 0
                    w_2_11_01_i = 0
                    w_2_11_10_i = 0

                w_2_11_00_sum += w_2_11_00_i
                w_2_11_11_sum += w_2_11_11_i
                w_2_11_01_sum += w_2_11_01_i
                w_2_11_10_sum += w_2_11_10_i

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                df_ALL[i, j] = df
                du_ALL[i, j] = du
                dchi_ALL[i, j] = dchi

            elif pattern == 14:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])

                df = 1 - 0.5 * a1 * a4
                f = f + df

                du = np.sqrt(a1 * a1 + a4 * a4)
                u = u + du

                dchi = -0.25
                chi = chi + dchi

                # identifying the sides!
                e0 = -a1
                e1 = -a4
                # updating the matrix elements
                w_2_11_00_i = e0 * e0 / np.sqrt(a1 * a1 + a4 * a4)
                w_2_11_11_i = e1 * e1 / np.sqrt(a1 * a1 + a4 * a4)
                w_2_11_01_i = e0 * e1 / np.sqrt(a1 * a1 + a4 * a4)
                w_2_11_10_i = e0 * e1 / np.sqrt(a1 * a1 + a4 * a4)

                if np.isclose([e0, e1], 0, atol=1e-3).all():
                    w_2_11_00_i = 0
                    w_2_11_11_i = 0
                    w_2_11_01_i = 0
                    w_2_11_10_i = 0

                w_2_11_00_sum += w_2_11_00_i
                w_2_11_11_sum += w_2_11_11_i
                w_2_11_01_sum += w_2_11_01_i
                w_2_11_10_sum += w_2_11_10_i

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                df_ALL[i, j] = df
                du_ALL[i, j] = du
                dchi_ALL[i, j] = dchi

            elif pattern == 15:
                # no contribution to boundary of figure from this pattern

                df = 1
                du = 0
                dchi = 0

                f += 1

                df_ALL[i, j] = df
                du_ALL[i, j] = du
                dchi_ALL[i, j] = dchi

    # now get the total matrix and normalize it by 2pi A factor!
    # f is the total area in this case

    w_2_11_matrix = np.array(
        [[w_2_11_00_sum, w_2_11_01_sum], [w_2_11_10_sum, w_2_11_11_sum]]
    )
    w_2_11_matrix = w_2_11_matrix / (2 * np.pi * (width - 1) * (height - 1))

    # the cross term components of this matrix are same so its a symmetric matrix
    # The existence of a preferred direction in the boundary will manifest as a
    # an unequality between the diagonal components of the matrix.

    if returnContributions:
        return (
            f,
            u,
            chi,
            w_2_11_matrix,
            pattern_storage,
            w_2_11_00_ALL,
            w_2_11_11_ALL,
            w_2_11_01_ALL,
            w_2_11_10_ALL,
            df_ALL,
            du_ALL,
            dchi_ALL,
        )
    else:
        return (
            f,
            u,
            chi,
            w_2_11_matrix,
            pattern_storage,
            w_2_11_00_ALL,
            w_2_11_11_ALL,
            w_2_11_01_ALL,
            w_2_11_10_ALL,
        )


def getPixelContributionsLegacy(data, threshold):
    """
    Function that computes the SMFs and TMFs for a data 2D array

    Parameters:
    ------------
    data = 2D array, the image in a 2d array format
    threshold = float, the threshold used to compute the MFs. Image above this threshold is considered in computing MFs

    """
    # getting the dimensions of the data to look at so we can loop over each cell in the image
    width = data.shape[0]
    height = data.shape[1]

    # looping over each square/cell in the image

    # I also wish to save in an array the pattern assigned to each cell for later use!
    # generate an empty array with one less length along each axis
    pattern_storage = np.ones((width - 1, height - 1)) * -99

    # we should store the W matrix components of all cells in an array so we can use them in a post process step later
    w_2_11_00_ALL = np.zeros((width - 1, height - 1))
    w_2_11_11_ALL = np.zeros((width - 1, height - 1))
    w_2_11_01_ALL = np.zeros((width - 1, height - 1))
    w_2_11_10_ALL = np.zeros((width - 1, height - 1))

    df_ALL = np.zeros((width - 1, height - 1))
    du_ALL = np.zeros((width - 1, height - 1))
    dchi_ALL = np.zeros((width - 1, height - 1))

    # looping over each pixel in rows and columns
    for i in range(width - 1):
        for j in range(height - 1):
            # we evaluate the pattern of the cell currently under consideration
            pattern = 0

            # NaN parsing

            if (
                (np.isnan(data[i, j]))
                or (np.isnan(data[i + 1, j]))
                or (np.isnan(data[i + 1, j + 1]))
                or (np.isnan(data[i, j + 1]))
                or (np.isinf(data[i, j]))
                or (np.isinf(data[i + 1, j]))
                or (np.isinf(data[i + 1, j + 1]))
                or (np.isinf(data[i, j + 1]))
            ):
                # This assumes that we're at the edge of the image, so we don't need to do anything
                # so we skip the cell (the alignment and anisotropy calculations break down severely too :/ )
                pass

            # doing the conditionals to evaluate what marching square pattern to choose for current cell
            if data[i, j] > threshold:
                pattern += 1
            if data[i + 1, j] > threshold:
                pattern += 2
            if data[i + 1, j + 1] > threshold:
                pattern += 4
            if data[i, j + 1] > threshold:
                pattern += 8
            # we save the pattern info to the pattern array
            pattern_storage[i, j] = pattern

            # now going to each kind of pattern
            # for each pattern we identify the edge length that contributes to the overall perimeter of the image
            # and other things that contribute to the other MFs and W matrix for TMFs
            # we will stick with the norm that e0 = {a1, a3} and e1 = {a2, a4} where {e0,e1} is an orthogonal basis.

            # Note that each pattern has a specific direction attached to it as well
            # We account for this direction in the sign of the edge components
            # Refer to Figure 7 of Schroeder Turk et al. 2010 paper on Minkowski Functionals for more info on this
            if pattern == 0:
                # no contribution to boundary of figure from this pattern
                pass
            elif pattern == 1:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, (j + 1)])

                df = 0.5 * a1 * a4
                du = np.sqrt(a1 * a1 + a4 * a4)
                dchi = 0.25
                # identifying the sides!
                e0 = +a1
                e1 = +a4
                # updating the matrix elements
                # need to multiply by the inverse edge length as well!

                w_2_11_00_i = e0 * e0 / np.sqrt(a1 * a1 + a4 * a4)
                w_2_11_11_i = e1 * e1 / np.sqrt(a1 * a1 + a4 * a4)
                w_2_11_01_i = e0 * e1 / np.sqrt(a1 * a1 + a4 * a4)
                w_2_11_10_i = e0 * e1 / np.sqrt(a1 * a1 + a4 * a4)

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 2:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a2 = (data[i + 1, j] - threshold) / (
                    data[i + 1, j] - data[i + 1, (j + 1)]
                )

                df = 0.5 * (1 - a1) * a2

                du = np.sqrt((1 - a1) * (1 - a1) + a2 * a2)

                dchi = 0.25

                # identifying the sides!
                e0 = 1 - a1
                e1 = -a2
                # updating the matrix elements
                w_2_11_00_i = e0 * e0 / np.sqrt((1 - a1) * (1 - a1) + a2 * a2)
                w_2_11_11_i = e1 * e1 / np.sqrt((1 - a1) * (1 - a1) + a2 * a2)
                w_2_11_01_i = e0 * e1 / np.sqrt((1 - a1) * (1 - a1) + a2 * a2)
                w_2_11_10_i = e0 * e1 / np.sqrt((1 - a1) * (1 - a1) + a2 * a2)

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 3:
                a2 = (data[i + 1, j] - threshold) / (
                    data[i + 1, j] - data[i + 1, (j + 1)]
                )
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, (j + 1)])

                df = +a2 + 0.5 * (a4 - a2)

                du = np.sqrt(1 + (a4 - a2) * (a4 - a2))

                dchi = 0

                # identifying the sides!
                e0 = 1
                e1 = a4 - a2
                # updating the matrix elements
                w_2_11_00_i = e0 * e0 / np.sqrt(1 + (a4 - a2) * (a4 - a2))
                w_2_11_11_i = e1 * e1 / np.sqrt(1 + (a4 - a2) * (a4 - a2))
                w_2_11_01_i = e0 * e1 / np.sqrt(1 + (a4 - a2) * (a4 - a2))
                w_2_11_10_i = e0 * e1 / np.sqrt(1 + (a4 - a2) * (a4 - a2))

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 4:
                a2 = (data[i + 1, j] - threshold) / (
                    data[i + 1, j] - data[i + 1, j + 1]
                )
                a3 = (data[i, j + 1] - threshold) / (
                    data[i, j + 1] - data[i + 1, j + 1]
                )

                df = 0.5 * (1 - a2) * (1 - a3)

                du = np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))

                dchi = 0.25

                # identifying the sides!
                e0 = -1 * (1 - a3)
                e1 = -1 * (1 - a2)
                # updating the matrix elements
                w_2_11_00_i = (
                    e0 * e0 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                )
                w_2_11_11_i = (
                    e1 * e1 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                )
                w_2_11_01_i = (
                    e0 * e1 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                )
                w_2_11_10_i = (
                    e0 * e1 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                )

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 5:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a2 = (data[i + 1, j] - threshold) / (
                    data[i + 1, j] - data[i + 1, j + 1]
                )
                a3 = (data[i, j + 1] - threshold) / (
                    data[i, j + 1] - data[i + 1, j + 1]
                )
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])

                df = 1 - 0.5 * (1 - a1) * a2 - 0.5 * a3 * (1 - a4)

                du = np.sqrt((1 - a1) * (1 - a1) + a2 * a2) + np.sqrt(
                    a3 * a3 + (1 - a4) * (1 - a4)
                )

                dchi = 0.5

                # identifying the sides!
                e0_1 = -1 * (1 - a1)
                e1_1 = a2
                e0_2 = a3
                e1_2 = -1 * (1 - a4)
                # updating the matrix elements
                w_2_11_00_i = e0_1 * e0_1 / np.sqrt(
                    (1 - a1) * (1 - a1) + a2 * a2
                ) + e0_2 * e0_2 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                w_2_11_11_i = e1_1 * e1_1 / np.sqrt(
                    (1 - a1) * (1 - a1) + a2 * a2
                ) + e1_2 * e1_2 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                w_2_11_01_i = e0_1 * e1_1 / np.sqrt(
                    (1 - a1) * (1 - a1) + a2 * a2
                ) + e0_2 * e1_2 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                w_2_11_10_i = e0_1 * e1_1 / np.sqrt(
                    (1 - a1) * (1 - a1) + a2 * a2
                ) + e0_2 * e1_2 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 6:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a3 = (data[i, j + 1] - threshold) / (
                    data[i, j + 1] - data[i + 1, j + 1]
                )

                df = (1 - a3) + 0.5 * (a3 - a1)

                du = np.sqrt(1 + (a3 - a1) * (a3 - a1))

                dchi = 0

                # identifying the sides!
                e0 = a3 - a1
                e1 = -1
                # updating the matrix elements
                w_2_11_00_i = e0 * e0 / np.sqrt(1 + (a3 - a1) * (a3 - a1))
                w_2_11_11_i = e1 * e1 / np.sqrt(1 + (a3 - a1) * (a3 - a1))
                w_2_11_01_i = e0 * e1 / np.sqrt(1 + (a3 - a1) * (a3 - a1))
                w_2_11_10_i = e0 * e1 / np.sqrt(1 + (a3 - a1) * (a3 - a1))

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 7:
                a3 = (data[i, j + 1] - threshold) / (
                    data[i, j + 1] - data[i + 1, j + 1]
                )
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])
                df = 1 - 0.5 * a3 * (1 - a4)

                du = np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))

                dchi = -0.25
                # identifying the sides!
                e0 = a3
                e1 = -1 * (1 - a4)
                # updating the matrix elements
                w_2_11_00_i = e0 * e0 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                w_2_11_11_i = e1 * e1 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                w_2_11_01_i = e0 * e1 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                w_2_11_10_i = e0 * e1 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 8:
                a3 = (data[i, j + 1] - threshold) / (
                    data[i, j + 1] - data[i + 1, j + 1]
                )
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])

                df = 0.5 * a3 * (1 - a4)

                du = np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))

                dchi = 0.25

                # identifying the sides!
                e0 = -a3
                e1 = 1 - a4
                # updating the matrix elements
                w_2_11_00_i = e0 * e0 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                w_2_11_11_i = e1 * e1 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                w_2_11_01_i = e0 * e1 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))
                w_2_11_10_i = e0 * e1 / np.sqrt(a3 * a3 + (1 - a4) * (1 - a4))

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 9:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a3 = (data[i, j + 1] - threshold) / (
                    data[i, j + 1] - data[i + 1, j + 1]
                )

                df = a1 + 0.5 * (a3 - a1)

                du = np.sqrt(1 + (a3 - a1) * (a3 - a1))

                dchi = 0

                # identifying the sides!
                e0 = a1 - a3
                e1 = 1
                # updating the matrix elements
                w_2_11_00_i = e0 * e0 / np.sqrt(1 + (a3 - a1) * (a3 - a1))
                w_2_11_11_i = e1 * e1 / np.sqrt(1 + (a3 - a1) * (a3 - a1))
                w_2_11_01_i = e0 * e1 / np.sqrt(1 + (a3 - a1) * (a3 - a1))
                w_2_11_10_i = e0 * e1 / np.sqrt(1 + (a3 - a1) * (a3 - a1))

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 10:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a2 = (data[i + 1, j] - threshold) / (
                    data[i + 1, j] - data[i + 1, j + 1]
                )
                a3 = (data[i, j + 1] - threshold) / (
                    data[i, j + 1] - data[i + 1, j + 1]
                )
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])

                df = 1 - 0.5 * a1 * a4 + 0.5 * (1 - a2) * (1 - a3)

                du = np.sqrt(a1 * a1 + a4 * a4) + np.sqrt(
                    (1 - a2) * (1 - a2) + (1 - a3) * (1 - a3)
                )

                dchi = 0.5

                # identifying the sides!
                e0_1 = -a1
                e1_1 = -a4
                e0_2 = 1 - a3
                e1_2 = 1 - a2
                # updating the matrix elements
                w_2_11_00_i = e0_1 * e0_1 / np.sqrt(
                    a1 * a1 + a4 * a4
                ) + e0_2 * e0_2 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                w_2_11_11_i = e1_1 * e1_1 / np.sqrt(
                    a1 * a1 + a4 * a4
                ) + e1_2 * e1_2 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                w_2_11_01_i = e0_1 * e1_1 / np.sqrt(
                    a1 * a1 + a4 * a4
                ) + e0_2 * e1_2 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                w_2_11_10_i = e0_1 * e1_1 / np.sqrt(
                    a1 * a1 + a4 * a4
                ) + e0_2 * e1_2 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                du_ALL[i, j] = du
                df_ALL[i, j] = df
                dchi_ALL[i, j] = dchi

            elif pattern == 11:
                a2 = (data[i + 1, j] - threshold) / (
                    data[i + 1, j] - data[i + 1, j + 1]
                )
                a3 = (data[i, j + 1] - threshold) / (
                    data[i, j + 1] - data[i + 1, j + 1]
                )

                df = 1 - 0.5 * (1 - a2) * (1 - a3)

                du = np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))

                dchi = -0.25

                # identifying the sides!
                e0 = 1 - a3
                e1 = 1 - a2
                # updating the matrix elements
                w_2_11_00_i = (
                    e0 * e0 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                )
                w_2_11_11_i = (
                    e1 * e1 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                )
                w_2_11_01_i = (
                    e0 * e1 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                )
                w_2_11_10_i = (
                    e0 * e1 / np.sqrt((1 - a2) * (1 - a2) + (1 - a3) * (1 - a3))
                )

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                df_ALL[i, j] = df
                du_ALL[i, j] = du
                dchi_ALL[i, j] = dchi

            elif pattern == 12:
                a2 = (data[i + 1, j] - threshold) / (
                    data[i + 1, j] - data[i + 1, j + 1]
                )
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])

                df = (1 - a2) + 0.5 * (a2 - a4)

                du = np.sqrt(1 + (a2 - a4) * (a2 - a4))

                dchi = 0

                # identifying the sides!
                e0 = -1
                e1 = a2 - a4
                # updating the matrix elements
                w_2_11_00_i = e0 * e0 / np.sqrt(1 + (a2 - a4) * (a2 - a4))
                w_2_11_11_i = e1 * e1 / np.sqrt(1 + (a2 - a4) * (a2 - a4))
                w_2_11_01_i = e0 * e1 / np.sqrt(1 + (a2 - a4) * (a2 - a4))
                w_2_11_10_i = e0 * e1 / np.sqrt(1 + (a2 - a4) * (a2 - a4))

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                df_ALL[i, j] = df
                du_ALL[i, j] = du
                dchi_ALL[i, j] = dchi

            elif pattern == 13:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a2 = (data[i + 1, j] - threshold) / (
                    data[i + 1, j] - data[i + 1, j + 1]
                )

                df = 1 - 0.5 * (1 - a1) * a2

                du = np.sqrt((1 - a1) * (1 - a1) + a2 * a2)

                dchi = -0.25

                # identifying the sides!
                e0 = -1 * (1 - a1)
                e1 = a2
                # updating the matrix elements

                w_2_11_00_i = e0 * e0 / np.sqrt((1 - a1) * (1 - a1) + a2 * a2)
                w_2_11_11_i = e1 * e1 / np.sqrt((1 - a1) * (1 - a1) + a2 * a2)
                w_2_11_01_i = e0 * e1 / np.sqrt((1 - a1) * (1 - a1) + a2 * a2)
                w_2_11_10_i = e0 * e1 / np.sqrt((1 - a1) * (1 - a1) + a2 * a2)

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                df_ALL[i, j] = df
                du_ALL[i, j] = du
                dchi_ALL[i, j] = dchi

            elif pattern == 14:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])

                df = 1 - 0.5 * a1 * a4

                du = np.sqrt(a1 * a1 + a4 * a4)

                dchi = -0.25

                # identifying the sides!
                e0 = -a1
                e1 = -a4
                # updating the matrix elements
                w_2_11_00_i = e0 * e0 / np.sqrt(a1 * a1 + a4 * a4)
                w_2_11_11_i = e1 * e1 / np.sqrt(a1 * a1 + a4 * a4)
                w_2_11_01_i = e0 * e1 / np.sqrt(a1 * a1 + a4 * a4)
                w_2_11_10_i = e0 * e1 / np.sqrt(a1 * a1 + a4 * a4)

                # we save these array contributions into array
                w_2_11_00_ALL[i, j] = w_2_11_00_i
                w_2_11_11_ALL[i, j] = w_2_11_11_i
                w_2_11_01_ALL[i, j] = w_2_11_01_i
                w_2_11_10_ALL[i, j] = w_2_11_10_i

                df_ALL[i, j] = df
                du_ALL[i, j] = du
                dchi_ALL[i, j] = dchi

            elif pattern == 15:
                # no contribution to boundary of figure from this pattern

                df = 1
                du = 0
                dchi = 0

                df_ALL[i, j] = df
                du_ALL[i, j] = du
                dchi_ALL[i, j] = dchi

    return (
        df_ALL,
        du_ALL,
        dchi_ALL,
        w_2_11_00_ALL,
        w_2_11_01_ALL,
        w_2_11_10_ALL,
        w_2_11_11_ALL,
    )


def get_struct_loc(labeled, comp_label):
    row_inds, col_inds = np.where(labeled == comp_label)
    return np.array([row_inds, col_inds])


def get_contour_locs(labeled, comp_label):
    """
    This function returns the coordinates of the cells in the contour grid that we will need to look at for "comp_label" isolated sub-polygon.

    It does not matter, it does not matter if I have a single point as one of my sub-polygons and if it is against the edge or something

    """

    # find the locations on grid for all points corresponding to this polygon
    row_inds, col_inds = np.where(labeled == comp_label)

    # print("Number of points in Comp %d before cuts = %d"%(comp_label, len(row_inds)))

    Npts = len(
        row_inds
    )  # this is the number of points identified as part of the sub-polygon

    # we write all the point co-ordinates in the shape (N,2)
    pts_i = np.vstack((row_inds, col_inds)).T
    # then i duplicate the above 4 time!
    pts_copy_i = np.repeat(pts_i, 4, axis=0)

    # these are the translational shifts we will perform
    ijm1 = np.zeros((Npts, 2))
    ijm1[:, 1] = -1

    im1jm1 = np.zeros((Npts, 2))
    im1jm1[:, 0] = -1
    im1jm1[:, 1] = -1

    im1j = np.zeros((Npts, 2))
    im1j[:, 0] = -1

    # and then add the above translation arrays to get all the points about the a grid point (i,j)
    # these shifts will give the array locations in the contour array not the grid array.
    # Note that the contour array has one less lenght and height compared to the grid array
    pts_copy_i[0::4] = pts_copy_i[0::4]
    pts_copy_i[1::4] = pts_copy_i[1::4] + ijm1
    pts_copy_i[2::4] = pts_copy_i[2::4] + im1jm1
    pts_copy_i[3::4] = pts_copy_i[3::4] + im1j

    # now we have an array of possible locations in contour array that we have to consider!
    # we need to remove the duplicate locations and also wrong locations (negative indices or outside the grid)

    pts_tups = [(pi[0], pi[1]) for pi in pts_copy_i]
    pts_tups = list(set(pts_tups))
    # and the convert back into array
    pts_f_i = np.array([[pif[0], pif[1]] for pif in pts_tups])

    # now that we have removed all the duplicates, let us remove the edge points and outside grid points

    # we get the maximum possible inds in the contour grid
    max_width = np.shape(labeled)[0] - 2
    max_height = np.shape(labeled)[1] - 2

    row_inds_f = pts_f_i[:, 0]
    col_inds_f = pts_f_i[:, 1]

    pts_ff_i = pts_f_i[
        (np.abs(row_inds_f) <= max_width)
        & (row_inds_f >= 0)
        & (col_inds_f >= 0)
        & (np.abs(col_inds_f) <= max_height)
    ]

    # print("Number of points in Comp %d after cuts = %d"%(comp_label, len(pts_ff_i)))

    # we are left with the final points that we will use to compute the matrix

    return pts_ff_i


def sub_polygon_tmf(
    data,
    threshold,
    w_2_11_00_ALL,
    w_2_11_11_ALL,
    w_2_11_01_ALL,
    w_2_11_10_ALL,
    do_print=True,
):
    """
    This function computes the tensor minkowski functions for the sub-polygons / isolated polygons in the image

    Note that marching squares has already been run on the entire image, and we will just accessing relevant indices
    to get the corresponding TMF calculations for single cell
    """

    # we first convert the data into a binary image depending on the threshold
    binary_img = np.copy(data)
    binary_img[data > threshold] = 1
    binary_img[data <= threshold] = 0

    # then we identify the sub-polygons or isolated polygons in the image
    # This defines the type of connection between nodes. Below mean all kinds of connections (also diagonals) are a connnection
    structure = np.ones((3, 3)).astype(int)
    labeled, ncomponents = label(binary_img, structure)
    if do_print == True:
        print("Threshold = %.3f, Number of Components = %d" % (threshold, ncomponents))

    # this list will store the individual eigen values of the polygon
    all_eigen_subpoly = []

    # print(np.shape(labeled), np.shape(w_2_11_00_ALL))

    for i in range(1, ncomponents + 1):
        # we are looping over each sub-polygon

        subpoly_pts = get_contour_locs(labeled, i)

        # print(subpoly_pts)

        row_inds_f = subpoly_pts[:, 0]
        col_inds_f = subpoly_pts[:, 1]

        # #find the locations on grid for all points corresponding to this polygon
        # row_inds, col_inds = np.where(labeled == i)

        # #we need to then remove the indices that are not along the edge for ease
        # row_inds_f = row_inds[(row_inds != max_row) & (col_inds != max_col)]
        # col_inds_f = col_inds[(row_inds != max_row) & (col_inds != max_col)]

        # now that we found the grid locations that do contribute to the W_2_11 matrix
        # let us find those corrsponding components
        # they should have the same coordinate location as the row_inds_f and col_inds_f!
        # we should check to confirm that we are not just reading 0 matrix values

        # For a given point, I basically need to check all 4 cells surrounding it
        # and not just the cell that is located diagonally down on right

        w_2_11_00_ALL_i = w_2_11_00_ALL[row_inds_f, col_inds_f]
        w_2_11_11_ALL_i = w_2_11_11_ALL[row_inds_f, col_inds_f]
        w_2_11_10_ALL_i = w_2_11_10_ALL[row_inds_f, col_inds_f]
        w_2_11_01_ALL_i = w_2_11_01_ALL[row_inds_f, col_inds_f]

        # the matrix components that are zero (which correspond to fully filled cells, will not add anything to the tot matrix) hence we do not need to worry about them

        # and then we do the computation!
        w_2_11_00_i = np.nansum(w_2_11_00_ALL_i)
        w_2_11_11_i = np.nansum(w_2_11_11_ALL_i)
        w_2_11_01_i = np.nansum(w_2_11_01_ALL_i)
        w_2_11_10_i = np.nansum(w_2_11_10_ALL_i)

        # then we compute the matrix out of this
        w_2_11_mat_i = np.array(
            [[w_2_11_00_i, w_2_11_01_i], [w_2_11_10_i, w_2_11_11_i]]
        )
        # then compute eigenvalues ratio for this matrix
        w_i, v_i = LA.eig(w_2_11_mat_i)

        # this may be a dumb fix for w_i = [0., 0.]...
        if np.max(w_i) == np.min(w_i):
            eigen_ratio = 1
        else:
            eigen_ratio = np.min(w_i) / np.max(w_i)

        all_eigen_subpoly.append(eigen_ratio)

    return all_eigen_subpoly


def get_marching_contour(data, threshold, weight=True, offset=[0, 0]):
    """
    This functions creates contour object for each cell by using the marching square algorithm

    To make our life easy, we will be assuming a grid laid out on integer numbers only

    offset is how we should shift the point co-ordinates, by default it is constructed on a [0,i],[0,j] kinda grid

    """
    width = data.shape[0]
    height = data.shape[1]

    # these list contains the hull objects and hull boundary points for the image
    all_hulls = []
    all_pts = []

    a1s = np.zeros((width - 1, height - 1))
    a2s = np.zeros((width - 1, height - 1))
    a3s = np.zeros((width - 1, height - 1))
    a4s = np.zeros((width - 1, height - 1))

    for i in range(width - 1):
        for j in range(height - 1):
            pattern = 0

            # doing the conditionals to evaluate what marching square pattern to choose for current cell
            if data[i, j] > threshold:
                pattern += 1
            if data[i + 1, j] > threshold:
                pattern += 2
            if data[i + 1, j + 1] > threshold:
                pattern += 4
            if data[i, j + 1] > threshold:
                pattern += 8

            # let us compute all the different kinds of possible points
            if weight == True:
                a1 = (data[i, j] - threshold) / (data[i, j] - data[i + 1, j])
                a2 = (data[i + 1, j] - threshold) / (
                    data[i + 1, j] - data[i + 1, (j + 1)]
                )
                a3 = (data[i, j + 1] - threshold) / (
                    data[i, j + 1] - data[i + 1, j + 1]
                )
                a4 = (data[i, j] - threshold) / (data[i, j] - data[i, j + 1])

                a1s[i, j] = a1
                a2s[i, j] = a2
                a3s[i, j] = a3
                a4s[i, j] = a4

            else:
                a1 = 0.5
                a2 = 0.5
                a3 = 0.5
                a4 = 0.5

            TL = [i - offset[0], j - offset[1]]  # top left corner of square
            TR = [i + 1 - offset[0], j - offset[1]]  # top right
            BL = [i - offset[0], j + 1 - offset[1]]  # bottom left
            BR = [i + 1 - offset[0], j + 1 - offset[1]]  # bottom left

            LM = [
                i - offset[0],
                j + a4 - offset[1],
            ]  # point on left hand side of square
            TM = [i + a1 - offset[0], j - offset[1]]  # point on top side
            BM = [i + a3 - offset[0], j + 1 - offset[1]]  # point on bottom side
            RM = [i + 1 - offset[0], j + a2 - offset[1]]  # point on right side

            # now going to each kind of pattern
            if pattern == 0:
                pass
            elif pattern == 1:
                pts_i = np.array([TL, LM, TM])
                hull_i = ConvexHull(pts_i)
                all_pts.append(pts_i)
                all_hulls.append(hull_i)

            elif pattern == 2:
                pts_i = np.array([TR, TM, RM])
                hull_i = ConvexHull(pts_i)
                all_pts.append(pts_i)
                all_hulls.append(hull_i)

            elif pattern == 3:
                pts_i = np.array([LM, TL, TR, RM])
                hull_i = ConvexHull(pts_i)
                all_pts.append(pts_i)
                all_hulls.append(hull_i)

            elif pattern == 4:
                pts_i = np.array([BR, RM, BM])
                hull_i = ConvexHull(pts_i)
                all_pts.append(pts_i)
                all_hulls.append(hull_i)

            elif pattern == 5:
                pts_i = np.array([LM, TL, TM, RM, BR, BM])
                hull_i = ConvexHull(pts_i)
                all_pts.append(pts_i)
                all_hulls.append(hull_i)

            elif pattern == 6:
                pts_i = np.array([TM, TR, BR, BM])
                hull_i = ConvexHull(pts_i)
                all_pts.append(pts_i)
                all_hulls.append(hull_i)

            elif pattern == 7:
                pts_i = np.array([LM, TL, TR, BR, BM])
                hull_i = ConvexHull(pts_i)
                all_pts.append(pts_i)
                all_hulls.append(hull_i)

            elif pattern == 8:
                pts_i = np.array([BL, BM, LM])
                hull_i = ConvexHull(pts_i)
                all_pts.append(pts_i)
                all_hulls.append(hull_i)

            elif pattern == 9:
                pts_i = np.array([TL, TM, BM, BL])
                hull_i = ConvexHull(pts_i)
                all_pts.append(pts_i)
                all_hulls.append(hull_i)

            elif pattern == 10:
                pts_i = np.array([BL, LM, TM, TR, RM, BM])
                hull_i = ConvexHull(pts_i)
                all_pts.append(pts_i)
                all_hulls.append(hull_i)

            elif pattern == 11:
                pts_i = np.array([TL, TR, RM, BM, BL])
                hull_i = ConvexHull(pts_i)
                all_pts.append(pts_i)
                all_hulls.append(hull_i)

            elif pattern == 12:
                pts_i = np.array([LM, RM, BR, BL])
                hull_i = ConvexHull(pts_i)
                all_pts.append(pts_i)
                all_hulls.append(hull_i)

            elif pattern == 13:
                pts_i = np.array([TL, TM, RM, BR, BL])
                hull_i = ConvexHull(pts_i)
                all_pts.append(pts_i)
                all_hulls.append(hull_i)

            elif pattern == 14:
                pts_i = np.array([TM, TR, BR, BL, LM])
                hull_i = ConvexHull(pts_i)
                all_pts.append(pts_i)
                all_hulls.append(hull_i)

            elif pattern == 15:
                pts_i = np.array([TL, TR, BR, BL])
                hull_i = ConvexHull(pts_i)
                all_pts.append(pts_i)
                all_hulls.append(hull_i)

    return all_hulls, all_pts, a1s, a2s, a3s, a4s
