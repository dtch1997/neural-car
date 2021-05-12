import numpy as np

from .attrdict import *
from .viz import *

def rotate_by_angle(vec, th):
    """ Rotate a 2D vector by angle

    :param vec: np.array of shape (2,)
    :param th: angle in radians
    """
    M = np.array([[np.cos(th), -np.sin(th)],[np.sin(th), np.cos(th)]])
    return M @ vec