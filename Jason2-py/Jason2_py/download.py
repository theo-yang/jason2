"""
Functions for downloading, reading Jason-2 ssha data and reading Copernicus Satellite Data:
- find_track()
- read_track()
- read_current_data()
"""

# Imports
import os
import numpy as np
from .helper import find_file_names, download_data_from_url, read_netcdf_file, find_distance, find_target_file


def find_track(start,cycle,output_directory=None,direction = 'N',supress_message=True): 
    """
    finds name of file containing track closest to start point in a specified cycle
    
    Parameters 
    ----------
    start (list): latitude ([-90,90]  °N) and longitude ([0,360] °E) of coordinate
    cycle (str): 3 number digit of cycle
    output_directory (str): directory in which data folder is found (default = cwd)
    direction (str): direction of track ('N' or 'S', default = 'N')
    supress_message (boolean): if True, don't print missing url messages (default = True)

    Returns
    -------
    filename_min_distance (str): file name of cycle which contains the coordinate closest to start point. Returns None
    if cycle does not exist.
    
    """
    
    # create data directory
    if output_directory != None:    
        out_dir = (output_directory + '\data')
    else:
        out_dir = os.getcwd() + '\data'
        
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # find names of all files in specified cycle
    parent_url = 'https://data.nodc.noaa.gov/jason2/gdr/gdr_ssha'
    cycle_url = parent_url + '/cycle' + cycle
    cycle_dir = out_dir + '\cycle' + cycle
    links = find_file_names(cycle_url,cycle_dir,cycle)
    if links == None:
        return
    
    # find and read file containing longitude and latitude closest to start
    if not os.path.exists(cycle_dir):
        os.makedirs(cycle_dir)
    minimum_distance = np.pi * 6371 # half circumference of Earth
    
    for i, filename in enumerate(links):
        url = cycle_url + '/' + filename
        path = cycle_dir + '\\' + filename
        download_data_from_url(url,path,supress_message)
        data_iter = read_netcdf_file(path,['lat','lon'])
        
        # check if directionality is correct
        if direction == 'N' and data_iter['lat'][0] - data_iter['lat'][1] > 0:
            continue
        if direction == 'S' and data_iter['lat'][0] - data_iter['lat'][1] < 0:
            continue
            
        # check if file has coordinate closer to start
        query_point = np.array([start] * len(data_iter['lat']))
        track_points = np.array([data_iter['lat'][:],data_iter['lon'][:]]).T
        min_dist_iter = min(find_distance(query_point,track_points))
        if min_dist_iter < minimum_distance:
            minimum_distance = min_dist_iter
            filenum_min_distance = i
    
    filename_min_distance = links[filenum_min_distance] 
    return filename_min_distance

def read_track(start,track_length,ref_cycle,target_cycle,reference_file,output_directory=None,supress_message=True): 
    """
    Reads JASON-2 Data latitude, longitude, ssha, and time data for track closest to specified point in target cycle
    
    Parameters 
    ----------
    start (list): latitude ([-90,90]]  degree N) and longitude ([0,360] degree E) of coordinate
    track_length (num): length of track (km)
    ref_cycle (str): 3 number digit of cycle used to reference where start coordinate is found
    target_cycle (str): 3 number digit of cycle for downloading data based on ref_cycle
    reference_file (str): name of nc file containing closest coordinate to start in reference cycle
    output_directory (str): directory in which data folder is found (default = current directory)
    supress_message (boolean): if True, don't print missing cycle messages (default = True)
    Returns
    -------
    data (dict): dictionary containing data for lat ([-90, 90] N), lon ([0, 360] E), ssha (m), and time (s) data 
    for full track in target cycle. Returns None if target datetime is missing from target cycle. 
    """
    
    # create output directory 
    if output_directory != None:    
        out_dir = output_directory + '\data' + '\cycle'
    else:
        out_dir = os.getcwd() + '\data' + '\cycle'
        
        
    # find time of closest coordinate in reference cycle
    ref_file_path = out_dir + ref_cycle + '\\' + reference_file
    ref_data = read_netcdf_file(ref_file_path,['lat','lon','time'])
    query_point = np.array([start] * len(ref_data['lat']))
    track_points = np.array([ref_data['lat'][:],ref_data['lon'][:]]).T
    min_idx = np.argmin(find_distance(query_point,track_points))
    ref_time = ref_data['time'][min_idx]
    
    # download and read data containing containing closest coordinate in specified cycle based on reference cycle
    target_cycle_dir = out_dir + target_cycle
    
    if not os.path.exists(target_cycle_dir): # check if downnload path already exists
        os.makedirs(target_cycle_dir)
    
    target_file = find_target_file(float(ref_time),ref_cycle,target_cycle,target_cycle_dir,supress_message)
    
    if target_file == None: # check if target file exists in JASON-2 Dataset
        return 
    
    target_url = 'https://data.nodc.noaa.gov/jason2/gdr/gdr_ssha' + '/cycle' + target_cycle + '/' + target_file
    target_path = target_cycle_dir + '\\' + target_file
    download_data_from_url(target_url,target_path)
    data = read_netcdf_file(target_path,['lat','lon','time','ssha'])
    
    # slice data to tracklength and direction specifications
    target_coords = np.array([data['lat'][:],data['lon'][:]]).T
    target_point = np.array([start] * len(data['lat']))
    start_idx = np.argmin(find_distance(target_point,target_coords))
    delta_idx = np.argmin(abs(track_length - find_distance(query_point[min_idx::,:],track_points[min_idx::,:])))
    end_idx = start_idx + delta_idx
    for key in data:
        data[key] = data[key][start_idx:end_idx+1]
    
    return data

def read_current_data(lat_range,lon_range,paths):
    """
    Extract current values from copernicus satellite daily gridded sea level data in a given coordinate grid. 
    See data source: https://cds.climate.copernicus.eu/cdsapp#!/dataset/satellite-sea-level-global?tab=overview

    Parameters
    ----------
        - paths (str list): directories where copernicus nc files are located
        - lat_range (float lst): [lat_min, lat_max] where latitude is between [89.875, -89.875] °N
        - lon_range (float lst):  [lon_min, lon_min] where longitude is between [0.125, 359.875] degrees °E
        
    Returns
    -------
        - time (K x 1 np vector): days since 1950-01-01 00:00:00 
        - latitude (N x 1 np vector): latitudes ([89.875, -89.875] °N) from copernicus dataset
        - longitude (M x 1 np vector): longitudes ([0.125, 359.875] degrees °E) from copernicus dataset
        - u (K x N x M np array): zonal current (m / s)
        - v (K x N x M np array): meridional current (m / s)
    """
    
    
    # find file names:
    all_files = []
    for path in paths:
        
        for file in os.listdir(path):
            
            # count nc files only
            if file.endswith(".nc"):
                all_files.append(os.path.join(path, file))
    
    # find lat and lon idxs
    ref_file = all_files[0]
    ref_data = read_netcdf_file(ref_file,['latitude','longitude'])
    lat_idxs = np.arange(np.argmax(ref_data['latitude'][:] >= lat_range[0]) - 1,np.argmax(ref_data['latitude'][:] >= lat_range[1]) + 1).tolist()
    lon_idxs = np.arange(np.argmax(ref_data['longitude'][:] >= lon_range[0]) - 1,np.argmax(ref_data['longitude'][:] >= lon_range[1]) + 1).tolist()

    # Initialize outputs
    time = np.zeros((len(all_files),1))
    latitude = np.array(ref_data['latitude'][lat_idxs]).reshape((len(ref_data['latitude'][lat_idxs]), 1))
    longitude = np.array(ref_data['longitude'][lon_idxs]).reshape((len(ref_data['longitude'][lon_idxs]), 1))
    u = np.zeros((len(time),len(latitude),len(longitude)))
    v = np.zeros(np.shape(u))
    
    # fill values
    for i, file in enumerate(all_files):
        data = read_netcdf_file(file,['time','ugos','vgos'])
        time[i] = int(data['time'][0])
        u[i,:,:] = np.array(data['ugos'][0,lat_idxs,lon_idxs])
        v[i,:,:] = np.array(data['vgos'][0,lat_idxs,lon_idxs])
        
    # return time sorted values
    return np.sort(time,0),latitude,longitude, u[time.argsort(axis=0)[:,0],:,:], v[time.argsort(axis=0)[:,0],:,:]