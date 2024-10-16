import asdf
from astropy.io import fits
import numpy as np
from matplotlib.path import Path

import pyregion


def load_density(file, type="herschel"):
    
    fits_file = fits.open(file)
    nh2 = fits_file[0].data.copy()

    if type == "planck_herschel":
        tau_850 = nh2.copy()
        Ak = (2640 * tau_850) + 0.012

        nh2 = (Ak / 0.112) * 0.93e21
        nh2[np.isinf(nh2)] = np.nanmin(nh2[~np.isinf(nh2)])
        nh2 = np.log10(nh2)

        nh2[np.isinf(nh2)] = np.nanmin(nh2[~np.isinf(nh2)])
        nh2[np.isnan(nh2)] = np.nanmin(nh2)
        # the first element of the array is the legit file for this type of map

    elif type == "herschel":
        nh2 = np.log10(nh2)
        nh2[np.isinf(nh2)] = np.nanmin(nh2[~np.isinf(nh2)])
        nh2[np.isnan(nh2)] = np.nanmin(nh2)
        
    return nh2

def getHeader(file):
    
    fits_file = fits.open(file)
    header = fits_file[0].header
    return header

def getBitMaskFromCoordinates(coords, data):

    mask = np.zeros_like(data)
    X, Y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))
    points = np.vstack((X.flatten(), Y.flatten())).T
    p = Path(coords)
    mask[p.contains_points(points).reshape(data.shape)] = 1

    return mask