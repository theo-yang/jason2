"""
Functions for preprocessing raw ssha data from Jason-2:
- preprocess_dataset()
- check_track_start()
- check_masked_values()
- find_mean()
"""
# Imports
import numpy as np
from copy import deepcopy
from .helper import find_distance

def preprocess_dataset(dataset,start,distance_tol=15,masked_tol=20,supress_message=True):
    """
    Takes ssha dataset and removes datasets with incorrect starting point or too many missing values. The function fills 
    missing values with mean value at a given point across all cycles. Values missing from all cycles are replaced with 
    the mean ssha across each respective cycle.
    
    Parameters 
    ----------
    dataset (nested dict): dictionary holding datasets of lat, lon, and ssha values, output of read_track() function.
    start (list): latitude ([-90,90]]  degree N) and longitude ([0,360] degree E) of coordinate
    distance_tol (float): tolerance (in km) from target start coordinate (default = 10 km)
    masked_tol (float): tolerance (# NaN values) for amount of missing values in track function (default = 20 values)
    supress_message (boolean): if True, don't print removal messges (default = True)
    Returns
    -------
    preprocessed_data (nested dict): dictionary holding preprocessed data
    """  
    
    all_data = deepcopy(dataset)
    preprocessed_data = dict()
    for cycle, data in all_data.items():
        # check distance criteria
        distance, dist_bool = check_track_start(data,start,distance_tol) 
        
        # check missing val criteria
        n_masked, mask_bool = check_masked_values(data,masked_tol) # missing val criteria
        
        if dist_bool and mask_bool:
            preprocessed_data[cycle] = data
            ssha = preprocessed_data[cycle]['ssha']
            
        # error messages
        elif not supress_message:
            if dist_bool:
                print('cycle%s has %d missing values' % (cycle,n_masked))
            elif mask_bool:
                print('cycle%s is %f km away from target start' % (cycle,distance))
            else:
                print('cycle%s has %d missing values and is %f km away from target start' % (cycle,n_masked,distance))
    
    # get mean of remaining data
    mean_data = find_mean(preprocessed_data)
    
    # replace missing values with avg value at that point across all datasets
    for cycle, data in preprocessed_data.items():
        ssha = preprocessed_data[cycle]['ssha']
        ssha[ssha.mask & ~mean_data.mask] = mean_data[ssha.mask & ~mean_data.mask] 
        
        # replace remaining missing data with mean of ssha across full track
        data['ssha'][mean_data.mask] = data['ssha'].mean()
            
    return preprocessed_data
    
def check_track_start(data,start,distance_tol=15):
    """
    Checks if data is witihn tolerated distance from start coordinate
    
    Parameters 
    ----------
    data (dict): dictionary containing latitude and longitude values, output of read_track() function.
    start (list): latitude ([-90,90]]  degree N) and longitude ([0,360] degree E) of coordinate
    distance_tol (float): tolerance (in km) from target start coordinate (default = 10 km).
    
    Returns
    -------
    distance: distance (km) between start and first coordinate of data
    keep (boolean): True if distance < tol. False if distance > tol.
    """   
    
    data_start_coords = np.array([[data['lat'][0],data['lon'][0]]])
    start_coords = np.array([start])
    distance = find_distance(start_coords,data_start_coords)
    keep = distance < distance_tol
    
    return distance[0], keep[0]

def check_masked_values(data,masked_tol=20):
    """
    Checks if ssha data has fewer than tolerated number of NaN values
    
    Parameters 
    ----------
    data (dict): dictionary containing ssha values, output of read_track() function.
    masked_tol (float): tolerance (# NaN values) for amount of missing values in track function (default = 20 values)
    
    Returns
    -------
    n_masked_values (int): number of masked values in cycle
    keep (boolean): True if ssha data has <= tol number of missing values. False otherwise
    """   
    return np.sum(data['ssha'].mask), np.sum(data['ssha'].mask) <= masked_tol

def find_mean(dataset):
    """
    Finds mean of ssha across dataset
    
    Parameters 
    ----------
    dataset (nested dict): dictionary holding datasets of ssha values, output of read_track() function.
    
    Returns
    -------
    mean ssha (m) across dataset, ignoring masked values
    """   
    cycles = list(dataset.keys())
    return np.ma.array(tuple([dataset[cycle]['ssha'] for cycle in cycles])).mean(axis=0)