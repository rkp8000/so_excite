import numpy as np
import os
from scipy.linalg import toeplitz

cc = np.concatenate


class GenericFlexible(object):
    """Class for generic object."""
    
    def __init__(self, **kwargs):
        
        for k, v in kwargs.items():
            self.__dict__[k] = v
    
    
class Generic(GenericFlexible):
    """Generic object with fixed attributes."""
           
    def __setattr__(self, k, v):
        
        raise Exception('Attributes may only be set at instantiation.')
        
        
# FILE I/O

def save(save_file, obj):
    """
    Save a python object to a file using np.save.
    
    :param save_file: path to save file (should have .npy extension)
    :param obj: python object to save
    :return: path to saved file
    """
    if len(save_file) < 4 or save_file[-4:].lower() != '.npy':
        raise ValueError('Saved file must end with ".npy" extension.')
        
    # make sure save directory exists
    save_dir = os.path.dirname(save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
 
    # delete file if it already exists
    # if os.path.exists(save_file):
    #     os.remove(save_file)
    
    np.save(save_file, np.array([obj]))
    return save_file


def load(load_file):
    """
    Load a python object using np.load.
    
    :param load_file: path to file containing object
    :return: loaded python object
    """
    if load_file[-4:].lower() != '.npy':
        raise ValueError('Load file must end with ".npy"')
        
    return np.load(load_file)[0]


def save_time_file(save_file, ts):
    """
    Save a file containing a set of time stamps. Sampling frequency is computed
    from the mean time interval in ts.
    :param save_file: path of file to save
    :param ts: 1D timestamp array
    :return: path to saved file
    """
    data = {'timestamps': ts, 'fs': 1 / np.mean(np.diff(ts))}
    return save(save_file, data)


def load_time_file(time_file):
    """
    Return the timestamp array and sampling frequency from a timestamp file.
    :param time_file: path to file containing timestamp array
    :return: timestamp array, sampling frequency
    """
    data_t = load(time_file)

    for key in ('timestamps', 'fs'):
        if key not in data_t:
            raise KeyError(
                'Item with key "{}" not found in file '
                '"{}".'.format(key, time_file))

    return data_t['timestamps'], data_t['fs']


# GENERAL ARRAY MANIPULATION

def idx_closest(x, y):
    """Return the index of the element in y that is closest to x."""
    
    if isinstance(x, (int, float)):
        return np.argmin(np.abs(x - np.array(y)))
    else:
        x_, y_ = np.meshgrid(x, y)
        return np.argmin(np.abs(x_ - y_), 0)


def find_segs(x):
    """
    Return a list of index pairs corresponding to groups of data where x is True.
    :param x: 1D array
    :return: list of index pairs
    """
    assert x.dtype == bool

    # find the indices of starts and ends
    diff = np.diff(np.concatenate([[0], x, [0]]))

    starts = (diff == 1).nonzero()[0]
    ends = (diff == -1).nonzero()[0]

    return np.array([starts, ends]).T


# MATH

def sgmd(x):
    """Sigmoid (logistic) function."""
    return 1 / (1 + np.exp(-x))


def lognormal_mu_sig(mean, std):
    """Get log-normal params from mean and std."""
    if mean <= 0:
        raise ValueError('Mean must be > 0 for log-normal distribution')
    
    b = 1 + (std**2)/(mean**2)
    
    mu = np.log(mean/np.sqrt(b))
    sig = np.sqrt(np.log(b))
    
    return mu, sig


def running_mean(x, wdw):
    """
    Return a running average of the same length as x, 
    using a specified sliding window size. Window size
    must be an odd number.
    
    :param x: 1-D array
    :param wdw: sliding window size (odd integer)
    """
    if not x.ndim == 1:
        raise ValueError('Arg "x" must be 1-D array.')
        
    if not wdw % 2 == 1:
        wdw = wdw - 1
        
    # first row and last col
    r = cc([np.repeat(np.nan, int(np.floor(wdw/2))), x[:int(np.ceil(wdw/2))]])
    c = cc([x[int(np.floor(wdw/2)):], np.repeat(np.nan, int(np.floor(wdw/2)))])
    
    # matrix for averaging
    mat = np.fliplr(toeplitz(c, r[::-1]))
    
    return np.nanmean(mat, 1)


def angle_from_dx_dt_dy_dt(dx_dt, dy_dt, unit):
    """
    Return an angle between -180 and 180 the time derivatives of two curves
    moving through the x-y plane.
    
    :param dx_dt: time-derivative of x
    :param dy_dt: time-derivative of y
    :param unit: 'deg' for degrees or 'rad' for radians
    """
    
    theta = np.arctan(dy_dt/dx_dt) * 180 / np.pi
    
    if dx_dt < 0:
        theta += (np.sign(dy_dt) * 180)
        
    if unit.lower() == 'deg':
        return theta
    elif unit.lower() == 'rad':
        return theta * np.pi / 180
    else:
        raise Exception('Unit "{}" not recognized.'.format(unit))


# DOWNSAMPLING FUNCTIONS FOR ANIMATIONS

def downsample_spks(spks, num):
    """
    Downsample a vector spk train to have num equally spaced samples.
    The downsampled spk value at a given time point is True if any spk occurred
    in the window surrounding that time point, and zero otherwise.
    
    :param spks: 2-D logical array indicating spk times (rows are times, cols
        are neurons)
    :param num: number of time points in the resampled signal
    """
    window = len(spks) / num
    
    if window < 1:
        err_msg = ('Provided "num" value must be less than len(spks); '
                   'upsampling is not supported')
        raise ValueError(err_msg)
    
    # just use a loop for now
    
    spks_down = np.zeros((num, spks.shape[1]), dtype=bool)
    
    for f_ctr in range(num):
        
        # get start and end of window for this downsampled time point
        start = int(round(window * f_ctr))
        end = int(round(window * (f_ctr + 1)))
        
        # assign the downsampled value to True for each neuron in which
        # any spk occurred in this time window
        spks_down[f_ctr] = np.any(spks[start:end], axis=0)
    
    return spks_down


def downsample_ma(xs, num):
    """
    Downsample an array to have num equally spaced samples, where the downsampled
    value at each time point is a moving average of the values in the 
    corresponding window.
    
    :param xs: N-D array of values (1st dim is times, higher dims are variables)
    :param num: number of time points in the resampled signal
    """
    window = len(xs) / num
    
    if window < 1:
        err_msg = ('Provided "num" value must be less than len(xs); '
                   'upsampling is not supported')
        raise ValueError(err_msg)
    
    # just use a loop for now
    xs_down = np.nan * np.zeros((num,) + xs.shape[1:])
    
    for f_ctr in range(num):
        
        # get start and end of window for this downsampled time point
        start = int(round(window * f_ctr))
        end = int(round(window * (f_ctr + 1)))
        
        # assign the downsampled value to the average of the values in
        # the corresponding window
        xs_down[f_ctr] = np.mean(xs[start:end], axis=0)
    
    return xs_down