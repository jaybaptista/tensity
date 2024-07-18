from scipy.ndimage import label
from tensity import getBinaryMap, sub_polygon_tmf, get_contour_locs, patternRepresentation, getPixelContributions
from tqdm import tqdm
import numpy as np
from scipy.spatial import ConvexHull

def peakThreshold(img,
                  peaks_x,
                  peaks_y,
                  peak_frac=.5):
    """
    Parameters
    ----------
    peaks_x : ndarray
        The x-coordinates of the peaks.
    peaks_y : ndarray
        The y-coordinates of the peaks.
    peak_frac : float
        The fraction of the maximum peak value that a peak must be in order to
        be considered a peak.

    Returns
    -------
    ndarray
        The x-coordinates of the peaks that meet the threshold.
    """

    structure = np.ones((3, 3)).astype(int)
    binary = []
    s = [] # surface area
    c = [] # circumference
    w_00 = []
    w_01 = []
    w_10 = []
    w_11 = []

    img = img.astype(np.float64)
    img[img < 0] = 0
    img[np.isnan(img)] = 0
    img[np.isinf(img)] = 0

    dfs, dus, dchis, W_00, W_01, W_10, W_11 = getPixelContributions(img, peak_thresh, False)

    for i in tqdm(range(len(peaks_x))):
        tmap = np.zeros_like(img)

        peak_value = img[peaks_y[i], peaks_x[i]]

        peak_thresh = peak_frac * peak_value
        bmap = getBinaryMap(img, peak_thresh)

        labeled, ncomponents = label(bmap, structure)
        target_coords = None

        target_comp = labeled[peaks_y[i], peaks_x[i]]

        subpoly_pts = get_contour_locs(labeled, target_comp)
        
        binary.append(subpoly_pts)

        s.append(np.sum(np.array(dfs)[subpoly_pts[:, 0], subpoly_pts[:, 1]]))
        c.append(np.sum(np.array(dus)[subpoly_pts[:, 0], subpoly_pts[:, 1]]))
        w_00.append(np.array(W_00))
        w_01.append(np.array(W_01))
        w_10.append(np.array(W_10))
        w_11.append(np.array(W_11))

    return binary, s, c, w_00, w_01, w_10, w_11

def get_marching_contour_VT(data, peak_x, peak_y, peak_frac=0.5, weight=True, offset=[0, 0]):
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

    threshold = peak_frac * data[peak_y, peak_x]

    labeled, ncomponents = label(getBinaryMap(data, threshold), np.ones((3, 3)).astype(int))
    target_comp = labeled[peaks_y, peaks_x]
    subpoly_pts = get_contour_locs(labeled, target_comp)
    # # this is the binary map that represents the component
    # tmap[subpoly_pts[:, 0], subpoly_pts[:, 1]] = 1

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


def getAngleVT(
    components, W_00, W_01, W_10, W_11, complement=False, eigenratio=False
):
    """
    This function takes in the precomputed structure components and the W tensor
    entries and creates the angle map for those structure components.
    """

    angle_map = np.zeros(len(components))

    if eigenratio:
        ratio_map = np.zeros(len(components))

    # we first convert the data into a binary image depending on the threshold
    # binary_img = getBinaryMap(data, threshold)

    # then we identify the sub-polygons or isolated polygons in the image
    # This defines the type of connection between nodes. Below mean all kinds of connections (also diagonals) are a connnection
    # structure = np.ones((3, 3)).astype(int)
    # labeled, ncomponents = label(binary_img, structure)

    # fix NaNs in the W matrix

    W_00[np.isnan(W_00)] = 0
    W_01[np.isnan(W_01)] = 0
    W_10[np.isnan(W_10)] = 0
    W_11[np.isnan(W_11)] = 0


    ncomponents = len(components)

    for i in range(0, ncomponents):
        # Get the points associated with the island

        # subpoly_pts = get_contour_locs(labeled, i)
        row_inds_f = components[i][:, 0]
        col_inds_f = components[i][:, 1]

        # Get the W matrix indices for those items
        W_00_i = W_00[i][row_inds_f, col_inds_f]
        W_01_i = W_01[i][row_inds_f, col_inds_f]
        W_10_i = W_10[i][row_inds_f, col_inds_f]
        W_11_i = W_11[i][row_inds_f, col_inds_f]

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
            ratio_map[i] = evals[0] / evals[1]

        angle_map[i] = angle

    if eigenratio:
        return angle_map, ratio_map
    else:
        return angle_map
    

class Peaks():
    def __init__(self, data, peaks_x, peaks_y, peak_frac=.5):
        # clean data

        data[data < 0] = 0
        data[np.isnan(data)] = 0
        data[np.isinf(data)] = 0
        self.data = data

        self.peaks_x = peaks_x
        self.peaks_y = peaks_y

        self.peak_frac = peak_frac
        self.components = None

        self.structure = np.ones((3, 3)).astype(int)

        print('Determining components...')
        self.getComponents()
        self.count = len(self.components)
        print('Components determined.')

        self.dfs = np.zeros([len(self.components), self.data.shape[0] - 1, self.data.shape[1] - 1])
        self.dus = np.zeros([len(self.components), self.data.shape[0] - 1, self.data.shape[1] - 1])
        self.dchis = np.zeros([len(self.components), self.data.shape[0] - 1, self.data.shape[1] - 1])

        # Tensorial Contributions
        self.W_00 = np.zeros([len(self.components), self.data.shape[0] - 1, self.data.shape[1] - 1])
        self.W_01 = np.zeros([len(self.components), self.data.shape[0] - 1, self.data.shape[1] - 1])
        self.W_10 = np.zeros([len(self.components), self.data.shape[0] - 1, self.data.shape[1] - 1])
        self.W_11 = np.zeros([len(self.components), self.data.shape[0] - 1, self.data.shape[1] - 1])

        self.a1 = np.zeros([len(self.components), self.data.shape[0] - 1, self.data.shape[1] - 1])
        self.a2 = np.zeros([len(self.components), self.data.shape[0] - 1, self.data.shape[1] - 1])
        self.a3 = np.zeros([len(self.components), self.data.shape[0] - 1, self.data.shape[1] - 1])
        self.a4 = np.zeros([len(self.components), self.data.shape[0] - 1, self.data.shape[1] - 1])

        self.functional()

    def getComponents(self):
        self.components = []
        self.peak_thresh = []
        self.patterns = []

        for i in tqdm(range(len(self.peaks_x))):
            tmap = np.zeros_like(self.data)
            peak_value = self.data[self.peaks_y[i], self.peaks_x[i]]
            peak_thresh = self.peak_frac * peak_value

            self.peak_thresh.append(peak_thresh)

            bmap = getBinaryMap(self.data, peak_thresh)

            labeled, ncomponents = label(bmap, self.structure)
            target_coords = None

            target_comp = labeled[self.peaks_y[i], self.peaks_x[i]]

            subpoly_pts = get_contour_locs(labeled, target_comp)

            # this is the binary map that represents the component
            tmap[subpoly_pts[:, 0], subpoly_pts[:, 1]] = 1

            self.components.append(subpoly_pts)

            patterns = (
                1 * tmap[:-1, :-1]
                + 2 * tmap[1:, :-1]
                + 4 * tmap[1:, 1:]
                + 8 * tmap[:-1, 1:]
            )

            self.patterns.append(patterns)
            
    def functional(self):
        for i in range(self.count):
            a1 = (self.data[:-1, :-1] - self.peak_thresh[i]) / (self.data[:-1, :-1] - self.data[1:, :-1])
            a2 = (self.data[1:, :-1] - self.peak_thresh[i]) / (self.data[1:, :-1] - self.data[1:, 1:])
            a3 = (self.data[:-1, 1:] - self.peak_thresh[i]) / (self.data[:-1, 1:] - self.data[1:, 1:])
            a4 = (self.data[:-1, :-1] - self.peak_thresh[i]) / (self.data[:-1, :-1] - self.data[:-1, 1:])
            
            dfs = np.zeros((self.data.shape[0] - 1, self.data.shape[1] - 1))
            dus = np.zeros((self.data.shape[0] - 1, self.data.shape[1] - 1))
            dchis = np.zeros((self.data.shape[0] - 1, self.data.shape[1] - 1))

            # Tensorial Contributions
            W_00 = np.zeros((self.data.shape[0] - 1, self.data.shape[1] - 1))
            W_01 = np.zeros((self.data.shape[0] - 1, self.data.shape[1] - 1))
            W_10 = np.zeros((self.data.shape[0] - 1, self.data.shape[1] - 1))
            W_11 = np.zeros((self.data.shape[0] - 1, self.data.shape[1] - 1))
            
            if 1 in self.patterns[i]:
                m = (self.patterns[i] == 1) & (a1 != 0) & (a4 != 0)
                dfs[m] = (0.5 * a1 * a4)[m]
                dus[m] = np.sqrt(a1**2 + a4**2)[m]
                dchis[m] = 0.25

                e0 = a1[m]
                e1 = a4[m]

                W_00[m] = (e0 * e0) / np.sqrt(a1**2 + a4**2)[m]
                W_01[m] = (e0 * e1) / np.sqrt(a1**2 + a4**2)[m]
                W_10[m] = (e0 * e1) / np.sqrt(a1**2 + a4**2)[m]
                W_11[m] = (e1 * e1) / np.sqrt(a1**2 + a4**2)[m]

            if 2 in self.patterns[i]:
                m = (self.patterns[i] == 2) & (a1 != 1) & (a2 != 0)
                dfs[m] = (0.5 * (1 - a1) * a2)[m]
                dus[m] = (np.sqrt((1 - a1) ** 2 + a2**2))[m]
                dchis[m] = 0.25

                e0 = (1 - a1)[m]
                e1 = (-a2)[m]

                W_00[m] = (e0 * e0) / np.sqrt((1 - a1) ** 2 + a2**2)[m]
                W_01[m] = (e0 * e1) / np.sqrt((1 - a1) ** 2 + a2**2)[m]
                W_10[m] = (e0 * e1) / np.sqrt((1 - a1) ** 2 + a2**2)[m]
                W_11[m] = (e1 * e1) / np.sqrt((1 - a1) ** 2 + a2**2)[m]

            if 3 in self.patterns[i]:
                m = self.patterns[i] == 3
                dfs[m] = (a2 + 0.5 * (a4 - a2))[m]
                dus[m] = np.sqrt(1 + (a4 - a2) ** 2)[m]

                e0 = 1
                e1 = (a4 - a2)[m]

                W_00[m] = (e0 * e0) / np.sqrt(1 + (a4 - a2) ** 2)[m]
                W_01[m] = (e0 * e1) / np.sqrt(1 + (a4 - a2) ** 2)[m]
                W_10[m] = (e0 * e1) / np.sqrt(1 + (a4 - a2) ** 2)[m]
                W_11[m] = (e1 * e1) / np.sqrt(1 + (a4 - a2) ** 2)[m]

            if 4 in self.patterns[i]:
                m = (self.patterns[i] == 4) & (a2 != 1) & (a3 != 1)
                dfs[m] = (0.5 * (1 - a2) * (1 - a3))[m]
                dus[m] = np.sqrt((1 - a2) ** 2 + (1 - a3) ** 2)[m]
                dchis[m] = 0.25

                e0 = (-1 * (1 - a3))[m]
                e1 = (-1 * (1 - a2))[m]

                W_00[m] = (e0 * e0) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]
                W_01[m] = (e0 * e1) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]
                W_10[m] = (e0 * e1) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]
                W_11[m] = (e1 * e1) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]

            if 5 in self.patterns[i]:
                m = (self.patterns[i] == 5)
                dfs[m] = (1 - 0.5 * (1 - a1) * a2 - 0.5 * a3 * (1 - a4))[m]
                dus[m] = (
                    np.sqrt((1 - a1) ** 2 + a2**2)[m] + np.sqrt(a3**2 + (1 - a4) ** 2)[m]
                )
                dchis[m] = 0.5


                m0_1 = (self.patterns[i] == 5) & (a1 != 1)
                m1_1 = (self.patterns[i] == 5) & (a2 != 0)

                m0_2 = (self.patterns[i] == 5) & (a3 != 0)
                m1_2 = (self.patterns[i] == 5) & (a4 != 1)

                m_mixed_1 = (self.patterns[i] == 5) & (a1 != 1) & (a2 != 0)
                m_mixed_2 = (self.patterns[i] == 5) & (a4 != 1) & (a3 != 0)

                e0_1 = -1 * (1 - a1)[m0_1]
                e1_1 = a2[m1_1]

                e0_2 = a3[m0_2]
                e1_2 = -1 * (1 - a4)[m1_2]

                e_mixed_1 = (-1 * (1 - a1) * a2)[m_mixed_1]
                e_mixed_2 = (a3 * -1 * (1 - a4))[m_mixed_2]

                W_00[m0_1] += (e0_1**2) / np.sqrt((1 - a1) ** 2 + a2**2)[m0_1]
                W_00[m0_2] += (e0_2**2) / np.sqrt(a3**2 + (1 - a4) ** 2)[m0_2]

                W_01[m_mixed_1] += (e_mixed_1) / np.sqrt((1 - a1) ** 2 + a2**2)[m_mixed_1]
                W_01[m_mixed_2] += (e_mixed_2) / np.sqrt(a3**2 + (1 - a4) ** 2)[m_mixed_2]

                W_10[m_mixed_1] += (e_mixed_1) / np.sqrt((1 - a1) ** 2 + a2**2)[m_mixed_1]
                W_10[m_mixed_2] += (e_mixed_2) / np.sqrt(a3**2 + (1 - a4) ** 2)[m_mixed_2]

                W_11[m1_1] += (e1_1**2) / np.sqrt((1 - a1) ** 2 + a2**2)[m1_1]
                W_11[m1_2] += (e1_2**2) / np.sqrt(a3**2 + (1 - a4) ** 2)[m1_2]


            if 6 in self.patterns[i]:
                m = self.patterns[i] == 6
                dfs[m] = ((1 - a3) + 0.5 * (a3 - a1))[m]
                dus[m] = np.sqrt(1 + (a3 - a1) ** 2)[m]
                dchis[m] = 0

                e0 = (a3 - a1)[m]
                e1 = -1

                W_00[m] = (e0 * e0) / np.sqrt(1 + (a3 - a1) ** 2)[m]
                W_01[m] = (e0 * e1) / np.sqrt(1 + (a3 - a1) ** 2)[m]
                W_10[m] = (e0 * e1) / np.sqrt(1 + (a3 - a1) ** 2)[m]
                W_11[m] = (e1 * e1) / np.sqrt(1 + (a3 - a1) ** 2)[m]

            if 7 in self.patterns[i]:
                m = (self.patterns[i] == 7) & (a3 != 0) & (a4 != 1)
                dfs[m] = (1 - 0.5 * a3 * (1 - a4))[m]
                dus[m] = np.sqrt(a3**2 + (1 - a4) ** 2)[m]
                dchis[m] = -0.25

                e0 = a3[m]
                e1 = -1 * (1 - a4)[m]

                W_00[m] = (e0 * e0) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]
                W_01[m] = (e0 * e1) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]
                W_10[m] = (e0 * e1) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]
                W_11[m] = (e1 * e1) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]

            if 8 in self.patterns[i]:
                m = (self.patterns[i] == 8) & (a3 != 0) & (a4 != 1)
                dfs[m] = (0.5 * a3 * (1 - a4))[m]
                dus[m] = np.sqrt(a3**2 + (1 - a4) ** 2)[m]
                dchis[m] = 0.25

                e0 = -a3[m]
                e1 = (1 - a4)[m]

                W_00[m] = (e0 * e0) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]
                W_01[m] = (e0 * e1) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]
                W_10[m] = (e0 * e1) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]
                W_11[m] = (e1 * e1) / np.sqrt(a3**2 + (1 - a4) ** 2)[m]

            if 9 in self.patterns[i]:
                m = self.patterns[i] == 9
                dfs[m] = (a1 + 0.5 * (a3 - a1))[m]
                dus[m] = np.sqrt(1 + (a3 - a1) ** 2)[m]

                e0 = (a1 - a3)[m]
                e1 = 1

                W_00[m] = (e0 * e0) / np.sqrt(1 + (a3 - a1) ** 2)[m]
                W_01[m] = (e0 * e1) / np.sqrt(1 + (a3 - a1) ** 2)[m]
                W_10[m] = (e0 * e1) / np.sqrt(1 + (a3 - a1) ** 2)[m]
                W_11[m] = (e1 * e1) / np.sqrt(1 + (a3 - a1) ** 2)[m]

            if 10 in self.patterns[i]:
                m = self.patterns[i] == 10
                dfs[m] = (1 - 0.5 * a1 * a4 + 0.5 * (1 - a2) * (1 - a3))[m]
                dus[m] = (np.sqrt(a1**2 + a4**2) + np.sqrt((1 - a2) ** 2 + (1 - a3) ** 2))[
                    m
                ]
                dchis[m] = 0.5

                m0_1 = (self.patterns[i] == 10) & (a1 != 0)
                m1_1 = (self.patterns[i] == 10) & (a4 != 0)

                m0_2 = (self.patterns[i] == 10) & (a3 != 1)
                m1_2 = (self.patterns[i] == 10) & (a2 != 1)

                m_mixed_1 = (self.patterns[i] == 10) & (a1 != 0) & (a4 != 0)
                m_mixed_2 = (self.patterns[i] == 10) & (a3 != 1) & (a2 != 1)

                e0_1 = -a1[m0_1]
                e1_1 = -a4[m1_1]

                e0_2 = (1 - a3)[m0_2]
                e1_2 = (1 - a2)[m1_2]

                e_mixed_1 = (a1 * a4)[m_mixed_1]
                e_mixed_2 = ((1 - a3) * (1 - a2))[m_mixed_2]

                W_00[m0_1] += (e0_1**2) / np.sqrt(a1**2 + a4**2)[m0_1]
                W_00[m0_2] += (e0_2**2) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m0_2]

                W_01[m_mixed_1] += e_mixed_1 / np.sqrt(a1**2 + a4**2)[m_mixed_1]
                W_01[m_mixed_2] += e_mixed_2 / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m_mixed_2]

                W_10[m_mixed_1] += e_mixed_1 / np.sqrt(a1**2 + a4**2)[m_mixed_1]
                W_10[m_mixed_2] += e_mixed_2 / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m_mixed_2]

                W_11[m1_1] += (e1_1**2) / np.sqrt(a1**2 + a4**2)[m1_1]
                W_11[m1_2] += (e1_2**2) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m1_2]

            if 11 in self.patterns[i]:
                m = (self.patterns[i] == 11) & (a3 != 1) & (a2 != 1)
                dfs[m] = (1 - 0.5 * (1 - a2) * (1 - a3))[m]
                dus[m] = np.sqrt((1 - a2) ** 2 + (1 - a3) ** 2)[m]
                dchis[m] = -0.25

                e0 = (1 - a3)[m]
                e1 = (1 - a2)[m]

                W_00[m] = (e0 * e0) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]
                W_01[m] = (e0 * e1) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]
                W_10[m] = (e0 * e1) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]
                W_11[m] = (e1 * e1) / np.sqrt((1 - a3) ** 2 + (1 - a2) ** 2)[m]

            if 12 in self.patterns[i]:
                m = self.patterns[i] == 12
                dfs[m] = ((1 - a2) + 0.5 * (a2 - a4))[m]
                dus[m] = np.sqrt(1 + (a2 - a4) ** 2)[m]

                e0 = -1
                e1 = (a2 - a4)[m]

                W_00[m] = (e0 * e0) / np.sqrt(1 + (a2 - a4) ** 2)[m]
                W_01[m] = (e0 * e1) / np.sqrt(1 + (a2 - a4) ** 2)[m]
                W_10[m] = (e0 * e1) / np.sqrt(1 + (a2 - a4) ** 2)[m]
                W_11[m] = (e1 * e1) / np.sqrt(1 + (a2 - a4) ** 2)[m]

            if 13 in self.patterns[i]:
                m = (self.patterns[i] == 13) & (a1 != 1) & (a2 != 0)
                dfs[m] = (1 - 0.5 * (1 - a1) * a2)[m]
                dus[m] = np.sqrt((1 - a1) ** 2 + a2**2)[m]
                dchis[m] = -0.25

                e0 = -(1 - a1)[m]
                e1 = a2[m]

                W_00[m] = (e0 * e0) / np.sqrt((1 - a1) ** 2 + a2**2)[m]
                W_01[m] = (e0 * e1) / np.sqrt((1 - a1) ** 2 + a2**2)[m]
                W_10[m] = (e0 * e1) / np.sqrt((1 - a1) ** 2 + a2**2)[m]
                W_11[m] = (e1 * e1) / np.sqrt((1 - a1) ** 2 + a2**2)[m]

            if 14 in self.patterns[i]:
                m = (self.patterns[i] == 14) & (a1 != 0) & (a4 != 0)
                dfs[m] = (1 - 0.5 * a1 * a4)[m]
                dus[m] = np.sqrt(a1**2 + a4**2)[m]
                dchis[m] = -0.25

                e0 = -a1[m]
                e1 = -a4[m]

                W_00[m] = (e0 * e0) / np.sqrt(a1**2 + a4**2)[m]
                W_01[m] = (e0 * e1) / np.sqrt(a1**2 + a4**2)[m]
                W_10[m] = (e0 * e1) / np.sqrt(a1**2 + a4**2)[m]
                W_11[m] = (e1 * e1) / np.sqrt(a1**2 + a4**2)[m]

            if 15 in self.patterns[i]:
                m = self.patterns[i] == 15
                dfs[m] = 1

            self.dfs[i] = dfs
            self.dus[i] = dus
            self.dchis[i] = dchis

            self.W_00[i] = W_00
            self.W_01[i] = W_01
            self.W_10[i] = W_10
            self.W_11[i] = W_11

            self.a1[i] = a1
            self.a2[i] = a2
            self.a3[i] = a3
            self.a4[i] = a4