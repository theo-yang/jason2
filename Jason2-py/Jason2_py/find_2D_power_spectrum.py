# Imports
import numpy as np
from scipy import stats
from .helper import find_distance

def find_2D_power_spectrum(dataset,num_time_windows=5):
    """
    Performs two dimensional fourier transform on ssha dataset using Hann windows in time and space and 50 % overlap processing.
    
    Parameters 
    ----------
    dataset (nested dict): dictionary holding datasets of lat, lon, and ssha values, output of read_track() function.
    num_time_windows (int): # of overlapping Hann Windows (default = 5)
    
    Returns
    -------
    fft_2D (array): average power spectrum (m^2 cpkm^-1 cpd^-1) across datasets after applying Hann Windows (dimensions = time x distance)
    dx (float): distance (km) between measurements
    dt (float): time (days) between measruements
    nu_vector (array): vector of wavenumbers (1/km)
    f_vector (array): vector of frequencies (1/day)
    """  
    
    # construct matrix of cycle-mean subtracted ssha data
    cycle_list, n_cycles = (list(dataset.values()), len(dataset.keys()))
    n_points = len(list(dataset.values())[0]['ssha'])
    n_times = 2 * num_time_windows
    ssha_data = np.zeros((n_cycles,n_points))
    cycle_times = [0] * n_cycles
    dxs = [0] * n_cycles
    for cycle in range(n_cycles):
        data = list(dataset.values())[cycle]
        cycle_ssha = data['ssha'].data
        ssha_data[cycle][:] = cycle_ssha - np.mean(cycle_ssha)
        cycle_times[cycle] = data['time'][0]
        start = [data['lat'][0],data['lon'][0]]
        query_point = np.array([start] * n_points)
        track_points = np.array([data['lat'],data['lon']]).T
        dxs[cycle] = stats.mode(np.trunc(np.diff(find_distance(track_points,query_point)) * 10000) / 10000).mode[0]
    # Find dx, dt
    dx = stats.mode(dxs).mode[0]
    dt = stats.mode(np.trunc(np.diff(cycle_times) * 10000) / 10000).mode[0] / (3600 * 24)

    # apply Hann window in time and space
    n_indices = n_times + 1
    cycle_indices = np.arange(0,n_cycles,int(n_cycles / n_indices))
    window_size = cycle_indices[2] - cycle_indices[0] + 1
    hann_mat = np.matmul(np.hanning(window_size).reshape(window_size,1),np.hanning(n_points).reshape(1,n_points))
    fft_sum_array = np.zeros(np.shape(hann_mat))
    for i, cycle_idx in enumerate(cycle_indices[0:-2]):
        start = cycle_idx
        end = cycle_indices[i+2] + 1
        windowed_data = ssha_data[start:end,:] * hann_mat
        fft_sum_array += np.abs(np.fft.fftshift(np.fft.fft2(windowed_data))) ** 2
    
    # take average of FT
    fft_2D = fft_sum_array / (i * np.size(fft_sum_array))
                 
    # find wavenumber and frequency vectors
    f_vector = np.fft.fftshift(np.fft.fftfreq(window_size, d=dt))
    nu_vector = np.fft.fftshift(np.fft.fftfreq(n_points, d=dx))
    return fft_2D, dx, dt, nu_vector, f_vector