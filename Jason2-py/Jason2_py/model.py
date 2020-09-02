"""
Functions for modeling the internal tide peaks on 2D power spectra from Jason-2 tracks:
- power_spectrum()
- mixed_power_spectrum()
- doppler_power_spectrum()
- fit_models()
- find_time_idx()
- find_avg_current_velocity()
"""
# Imports
import numpy as np
from numpy import sinc
from scipy.interpolate import RegularGridInterpolator

def power_spectrum(A,L,nu_vector,k,T,w,f_vector,dx,dt):
    """
    Computes model two dimensional power spectrum of ssha signal, s(x,t) = h(x,t) * w(x,t) * d(x,t)
    h(x,t) = A * sin (2 * pi (k * x + w * t))
    w(x,t) = Hann Window
    w(x,t) = 1/4 * (1 + cos(2 * pi * x / L)) * (1 + cos(2 * pi * t / T)) *
          rect(x / L) * rect(t / T) 
    d(x,t) = Signaling impulse train
    d(x,t) = III_dx(x) * III_dt(t) 
          (where III_i(y) is the Dirac Comb Function with period i)
    
    Parameters
    ----------
    A (float): Amplitude (m)
    L (float): Length of track (km)
    nu_vector (1xM np array): Vector of wavenumbers for evaluation (1/km), 
    k (float): Tidal wavenumber (1/km)
    T (float): Length of window interval (day)
    w (float): Alias frequency of tidal signal (1/day)
    f_vector (1xN np array): Vector of frequencies for evaluation (1/day)
    dx (float): Distance between consecutive measurements (km)
    dt (float): Time interval between consecutive measurements (day)
    
    Returns
    -------
    power (N x M np array): two dimensional wave power spectrum (m^2/km^2/day^2) computed in the range specified by k and f. 
    """
    
    # set dirac comb as sum from n = -50 to 50
    n_ = np.arange(-50,51)
    
    # create 3D mesh grid
    nu, f = np.meshgrid(nu_vector,f_vector)

    #solve for each n, wavenumber, and frequency
    sum1 = 0;
    sum2 = 0;
    sum3 = 0;
    sum4 = 0;

    for n in n_:
        sum1 += (L * sinc(L * (nu - n / dx + k)) + .5 * L * sinc(L * (nu - n / dx + k) - 1) + .5 * L * sinc(L * (nu - n / dx + k) + 1))
        sum2 += (L * sinc(L * (nu - n / dx - k)) + .5 * L * sinc(L * (nu - n / dx - k) - 1) + .5 * L * sinc(L * (nu - n / dx - k) + 1)) 
        sum3 += (T * sinc(T * (f - n / dt + w)) + .5 * T * sinc(T * (f - n / dt + w) - 1) + .5 * T * sinc(T * (f - n / dt + w) + 1))
        sum4 += (T * sinc(T * (f - n / dt - w)) + .5 * T * sinc(T * (f - n / dt - w) - 1) + .5 * T * sinc(T * (f - n / dt - w) + 1))

    return np.abs(1 / (8 * dx * dt) * A * 1j * (sum1 * sum3 - sum2 * sum4)) ** 2

def mixed_power_spectrum(A,L,nu_vector,k,T,w,f_vector,dx,dt):
    """
    Computes model two dimensional power spectrum of ssha signal with n tidal constituents: 
    
    s(x,t) = s_1(x,t,A_1,k_1,w_1) + ... + s_n(x,t,A_n,k_n,w_n)
    
    where each tidal constituent may have a different amplitude, wavenumber, and alias frequency.

    
    Parameters
    ----------
    A (list of floats): Amplitudes (m)
    L (float): Length of track (km)
    nu_vector (1xM np array): Vector of wavenumbers for evaluation (1/km), 
    k (list of floats): Tidal wavenumbers (1/km)
    T (float): Length of window interval (day)
    w (list of floats): Alias frequencies of tidal constituents (1/day)
    f_vector (1xN np array): Vector of frequencies for evaluation (1/day)
    dx (float): Distance between consecutive measurements (km)
    dt (float): Time interval between consecutive measurements (day)
    
    Returns
    -------
    power (N x M np array): two dimensional wave power spectrum (m^2/km^2/day^2) computed in the range specified by k and f. 
    """
    
    # Check that the number of Amplitudes, wavenumbers, and frequencies is consistent
    if not (len(A) == len(k) == len(w)):
        print('Lengths of amplitudes, wavenumbers, and frequencies not consistent!')
        return
    
    # find power spectrum
    power = np.zeros((len(f_vector),len(nu_vector)))
    for tide_num, amp in enumerate(A):
        power += power_spectrum(amp,L,nu_vector,k[tide_num],T,w[tide_num],f_vector,dx,dt)
    
    return power

def doppler_power_spectrum(eigenvector,current_data,ssha_data,args):
    """
    Finds the 2D power spectrum of tidal signal s(x,t) = A * sin(k * x + w_t * t) where
    w_t = (alias frequency) - (avg current velocity) • (tidal wavenumber). Simulations are 
    carried out for each cycle and averaged over the entire dataset.
   
    Parameters
    ----------
    - eigenvector (np array): values of the first baroclinic mode (2nd vertical mode, F_1)
    - current_data (dict): contains u and v current magnitudes from copernicus dataset
    - ssha_data (np array): track dataset from Jason-2, output from read_track()
    - args: tuple containing parameters for jason-2 track given below.
            - lat_range (list of floats): start and end latitudes (°N [-90, 90])
            - lon_range (list of floats): start and end longitudes (°E from [0, 360])
            - f_vector (np array): frequency values (1/day), from find_2D_power_spectrum() 
            - nu_vector (np array): wavenumber values (1/km), from find_2D_power_spectrum()
            - w0 (float): alias frequency (1/day)
            - L (float): length of track (km)
            - k (float): tidal wavenumber (1/km)
            - T (float): length of window (day)
            - dx (float): distance between measurmeants (km), from find_2D_power_spectrum()
            - dt (float): time between repeated measurments (day), from find_2D_power_spectrum()
            - fft_2D (np array): two dimensional power spectrum from JASON-2 data, from power_spectrum()
    Returns
    -------
    w_t (float list): doppler-shifted alias frequency for each cycle (1/day)
    doppler_shifted_model (np array): 2D power spectrum of doppler shifted model
    
    """
    
    # unpack args
    lat_range, lon_range, f_vector, nu_vector, w0, L, k, T, dx, dt, fft_2D = args
    current_lat = current_data['lat'].reshape(len(current_data['lat']))
    current_lon = current_data['lon'].reshape(len(current_data['lon']))
    
    # initialize arrays
    sim_data = np.zeros((len(ssha_data.keys()),len(f_vector),len(nu_vector)))
    w_t = [0] * len(ssha_data.keys())
    U = [0] * len(ssha_data.keys())
    
    # set velocity BCs for water column
    A = np.array([[-1, eigenvector[-1]],[-1,eigenvector[0]]])
    
    for cycle_num, cycle in enumerate(ssha_data.keys()):
        
        # find average u,v from day of Jason-2 pass
        day_idx = find_time_idx(current_data,ssha_data,cycle)
        u,v = find_avg_current_velocity(current_data, ssha_data,day_idx,cycle,current_lon,current_lat)
        
        # solve for depth-averaged current magnitude 
        mag = np.sqrt(u ** 2 + v ** 2)
        B = np.array([[mag],[0]])
        coeffs = np.linalg.solve(A,B)
        _U = np.mean(-coeffs[0] + coeffs[1] * eigenvector)
        U[cycle_num] = _U
        
        # find current angle relative to track
        omega = np.arctan(v/u) - np.arctan(np.diff(lat_range) / np.diff(lon_range))
        
        # apply doppler shift
        w_t[cycle_num] = w0 + _U * 24 * 3600 / 1000 * abs(k) * np.cos(omega)
        
        # Simulate 2D power spectrum, with amplitude at tidal wavenumber and doppler shifted alias frequency
        unit_amp = power_spectrum(1,L,nu_vector,k,T,w_t[cycle_num],f_vector,dx,dt)
        amp = fit_models([[k,w_t[cycle_num]]],nu_vector,f_vector,fft_2D, [unit_amp])
        sim_data[cycle_num] = amp[0] * unit_amp
        
    return w_t, np.mean(sim_data,axis=0) 

def fit_models(peaks,nu_vector,f_vector,power_spectrum, model_power_spectra):
    """
    For model 2D power spectrum:
    
    s(x,t) = A1 * s_1(x,t,A_1,k_1,w_1) + ... + A_n * s_n(x,t,A_n,k_n,w_n)
    
    finds coefficients [A_1,...,A_n] such that s(x,t) equals observed power spectrum at specified peaks in data
    
    Parameters
    ----------
    peaks (list of 2 element float lists): length n list of coordinate of peaks [tidal wavenumber (1/km), tidal frequency (1/day)] 
                                           for fitting.
    power_spectrum (np array): power spectrum from JASON-2 data, i.e. fft_2D output from find_2D_power_spectrum()
    model_power_spectrum (list of np arrays): length n output power spectra from power_spectrum()
    nu_vector (array): vector of wavenumbers (1/km)
    f_vector (array): vector of frequencies (1/day)
    
    Returns
    -------
    amplitudes (np vector): length n vector of coefficients 
    """
    
    # setup linear system of equations in matrix form, ax = b
    b = [0] * len(peaks)
    x = np.zeros((len(peaks),len(peaks)))
    for i, peak in enumerate(peaks):
        target_k = peak[0]
        target_f = peak[1]
        k_idx = (np.abs(nu_vector - target_k)).argmin()
        w_idx = (np.abs(f_vector - target_f)).argmin()
        b[i] = power_spectrum[w_idx][k_idx]
        for j, model in enumerate(model_power_spectra):
            x[i,j] = model[w_idx][k_idx]
    
    # solve linear equations and return coefficient
    return np.linalg.solve(x, b)

def find_time_idx(current_data,ssha_data,cycle):
    """
    Finds the day at which a jason-2 cycle pass occurs relative to start of Copernicus data collection
    
    Parameters
    ----------
    - current_data (dict): containing u and v current magnitudes from copernicus
    - ssha_data (np array): track dataset from Jason-2, i.e.  output from read_track()
    - cycle (str): 3 number digit of Jason-2 cycle
    - args: tuple containing parameters for jason-2 track given below.
            - lat_range (list of floats): start and end latitudes (°N, [-90, 90])
            - lon_range (list of floats): start and end longitudes (°E, [0, 360])
            - f_vector (np array): frequency values (1/day), from find_2D_power_spectrum() 
            - nu_vector (np array): wavenumber values (1/km), from find_2D_power_spectrum()
            - w0 (float): alias frequency (1/day)
            - L (float): length of track (km)
            - k (float): tidal wavenumber (1/km)
            - T (float): length of window (day)
            - dx (float): distance between measurmeants (km), from find_2D_power_spectrum()
            - dt (float): time between repeated measurments (day), from find_2D_power_spectrum()
            - fft_2D (np array): two dimensional power spectrum from JASON-2 data, from power_spectrum()

    Returns
    -------
    day_idxs (np vector):  index to extract current data
    """
    
    current_start = np.where(current_data['time'] == 21449)[0][0] # 2008-07-22
    jason_start = list(ssha_data.values())[0]['time'][0]
    day_idx = current_start + np.round((ssha_data[cycle]['time'][0] - jason_start) / (24 * 60 * 60))
    
    return int(day_idx)

def find_avg_current_velocity(current_data, ssha_data,day_idx,cycle,current_lon,current_lat):
    """
    Finds the average surface current velocities interpolated onto Jason-2 track points
   
    Parameters
    ----------
    - current_data (dict): contains u and v current magnitudes from copernicus dataset
    - ssha_data (np array): track dataset from Jason-2, output from read_track()
    - day_idx (int): index of current_data corresponding to a cycle pass in Jason-2
    - cycle (str): 3 number digit of Jason-2 cycle
    - current_lon (np array): longitudes from ECCO dataset
    - current_lat (np array): latitudes from ECCO dataset
    - args: tuple containing parameters for Jason-2 track given below.
            - lat_range (list of floats): start and end latitudes (°N, [-90, 90])
            - lon_range (list of floats): start and end longitudes (°E, [0, 360])
            - f_vector (np array): frequency values (1/day), from find_2D_power_spectrum() 
            - nu_vector (np array): wavenumber values (1/km), from find_2D_power_spectrum()
            - w0 (float): alias frequency (1/day)
            - L (float): length of track (km)
            - k (float): tidal wavenumber (1/km)
            - T (float): length of window (day)
            - dx (float): distance between measurmeants (km), from find_2D_power_spectrum()
            - dt (float): time between repeated measurments (day), from find_2D_power_spectrum()
            - fft_2D (np array): two dimensional power spectrum from Jason-2 data, from power_spectrum()

    Returns
    -------
    u (float): zonal mean flow (m/s)
    v (float): meridional mean flow (m/s)
    """
    
    # find average current velocity
    
    u = current_data['u'][day_idx]
    v = current_data['v'][day_idx]
    uF = RegularGridInterpolator((current_lat,current_lon),u)
    vF = RegularGridInterpolator((current_lat,current_lon),v)
    jason_coords = np.array([ssha_data[cycle]['lat'],ssha_data[cycle]['lon']]).T
    
    return np.mean(uF(jason_coords)), np.mean(vF(jason_coords))