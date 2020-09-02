# Imports
import numpy as np

def find_square_error(f_range,nu_range,f_vector,nu_vector,power_spectrum,model_power_spectrum):
    """
    Finds square error between model and actual power spectra in specified wavenumber range (1/km) and frequency range (1/km)
    
    Parameters
    ----------
    f_range (float list): range of frequency (1/day) values to check error
    nu_range (float list): range of wavenumbers (1/km) values to check error
    nu_vector (np vector): vector of wavenumbers (1/km) corresponding to power spectra
    f_vector (np vector): vector of frequencies (1/day) corresponding to power specta
    power_spectrum (np array): power spectrum from JASON-2 data, i.e. fft_2D output from find_2D_power_spectrum()
    model_power_spectrum (np array): output power spectra from power_spectrum()
    
    Returns
    ------
    
    """
    
    # Slice by frequency and wavenumber range
    f_idx = np.where((f_vector >= f_range[0]) & (f_vector <= f_range[1]))
    nu_idx = np.where((nu_vector >= nu_range[0]) & (nu_vector <= nu_range[1]))
    f_mesh, nu_mesh  = np.meshgrid(f_idx, nu_idx, indexing='ij')
    
    return np.sum((power_spectrum[f_mesh,nu_mesh] - model_power_spectrum[f_mesh,nu_mesh]) ** 2)