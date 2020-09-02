import numpy as np
def find_N_from_DRHODR(lat_range,lon_range,drhodr):
    '''
    Finds the average Brunt Vaisala frequency in a specified grid given the stratification from ECCOv4 xarray

    Parameters
    ----------
        - start_coord (float list): starting and ending latitudes ([-90,90]  °N) of track
        - end_coord (float list): starting and ending longitudes ([-180,180] °E) of track
        - drhodr (xarray): dataset of stratification from ECCOv4

    Returns
    -------
         - N (np array): buoyancy frequency values at each depth of drhodr array in s^-1
    '''
        
    # Restrict ECCOv4 data to a grid whose diagonal are these two points
    idxs = np.where((drhodr.YC.values > lat_range[0]) & (drhodr.YC.values < lat_range[1]) & (drhodr.XC.values > lon_range[0]) & (drhodr.XC.values < lon_range[1]))
    drhodr_vals = drhodr.DRHODR[:,:,idxs[0],idxs[1],idxs[2]]
    drhodr_avg = np.mean(drhodr_vals,axis=[0,2,3,4]).values
    
    # Compute N from drhodr using MITgcm model paramters
    g = 9.81
    rho_0 = 999.8
    N = np.sqrt(- g / rho_0 * drhodr_avg)
    
    return N