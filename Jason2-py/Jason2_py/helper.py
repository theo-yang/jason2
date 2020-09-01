"""
helper functions for Jason-2-py:
- find_file_names()
- download_data_from_url()
- download_data_from_url()
- read_netcdf_file()
- find_distance()
- find_target_file()
- find_alias_frq()
"""
# Imports
import numpy as np
import urllib
from netCDF4 import Dataset
from datetime import datetime,timedelta
from bs4 import BeautifulSoup
import os

def find_file_names(url,path,cycle,supress_message=True):
    """
    Finds list of filenames from a single JASON-2 cycle.
    
    Parameters
    ----------
    url (str): Full URL to check for files from JASON-2 gdr ssha
    path (str): path to check if file names exist already
    cycle (str): 3 number digit of cycle
    supress_message (boolean): if True, don't print missing cycle messages (default = True)
    
    Returns
    -------
    links (str list): list of nc files in jason2 dataset. returns None if cycle is missing 
    """
    
    parent_url = 'https://data.nodc.noaa.gov/jason2/gdr/gdr_ssha'
    url = parent_url + '/cycle' + cycle
    cycle_path = path + '\\filenames.txt'
        
    # Check if path already has txt file containing list of file names
    if os.path.exists(cycle_path):
        filenames_txt = open(cycle_path,"r") 
        links = filenames_txt.read().splitlines() 
        filenames_txt.close()
        
    else:
        # Check if URL exists:
        try:
            html_page = urllib.request.urlopen(url)
        except:
            if not supress_message:
                print('cycle %s does not exist'% cycle) 
            return 
        soup = BeautifulSoup(html_page,features="html.parser")
        links = []
        for link in soup.find_all('a', href=True):
            links.append(link['href'])
        links = links[5::]
        if not os.path.exists(path):
            os.mkdir(path)
        with open(cycle_path, "w") as filenames_txt:
            for link in links:
                filenames_txt.write("%s\n" % link)
            filenames_txt.close()
    return links

def download_data_from_url(url,path,supress_message=True):
    """
    Downloads netcdf data from URL
    
    Parameters
    ----------
    url (str): Full URL of file
    path (str): Path where the file is downloaded
    supress_message (boolean): if True, don't print missing url messages (default = True)
    
    Returns
    -------
    """
        
    # check if file already exists before downloading
    if not os.path.exists(path):
        try:
            urllib.request.urlretrieve(url,path)
        except:
            if not supress_message:
                print('%s does not exist' % url)
        
def read_netcdf_file(path,variables):
    """
    Reads variables from netcdf file
    
    Parameters
    ----------
    path (str): where the netcdf file is stored
    variables (str list): list of variables to save
    
    Returns
    -------
    dictionary containing data for each queried variable
    """
    
    dataset = Dataset(path, "r", format="NETCDF4")
    keys = variables
    vals = [dataset.variables[x] for x in variables]
    
    return dict(zip(keys,vals))

def find_distance(point1,point2):
    """
    Finds the distance between two coordinates positions on Earth
    
    Parameters
    ----------
    point1 (n x 2 numpy array): [[lat1, lon1], ...]
    point2 (n x 2 numpy array): [[lat2, lon2], ...]

    Returns
    -------
    distance (km) between points
    """
    
    p1 = point1; p2 = point2;
    R = 6371; # radius of earth
    pi = np.pi
    lat1 = pi / 180 * p1[:,0]; lat2 = pi / 180 * p2[:,0];
    lon1 = pi / 180 * p1[:,1]; lon2 = pi / 180 * p2[:,1];
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a),np.sqrt(1-a))
    
    return R * c

def find_target_file(start_time,start_cycle,target_cycle,path,supress_message=True):
    """
    Finds file name in target cycle containing repeated measurement at coordinate corresponding to start time based 
    on start cycle.  
    
    Parameters
    ----------
    start_time (float): seconds since 2000-01-01 00:00:00.0
    start_cycle (str): 3 number digit of start cycle
    target_cycle (str): 3 number digit of target cycle
    path (str): path where target filenames are to be found
    supress_message (boolean): if True, don't print missing datetime messages (default = True)

    Returns
    -------
    target_file (str): name of file in target_cycle containing pass over coordinates specified by start_day. Returns 
    None if target datetime is not in cycle.
    """
    
    # find datetime of measurement in target cycle
    reference_td = datetime(2000,1,1,0,0,0,0) 
    start_td = reference_td + timedelta(seconds=start_time)
    dt = timedelta(seconds=9.915634 * 24 * 60 * 60 * (int(target_cycle) - int(start_cycle)))
    target_td = start_td + dt
    
    # find file containing target datetime
    target_file = 'missing'
    url = 'https://data.nodc.noaa.gov/jason2/gdr/gdr_ssha' + '/cycle' + target_cycle
    filenames = find_file_names(url,path,target_cycle,supress_message)
    if filenames == None:
        return
    
    for name in filenames:
        start = datetime.strptime(name[20:35], "%Y%m%d_%H%M%S")
        stop = datetime.strptime(name[36:51], "%Y%m%d_%H%M%S")
        if start < target_td < stop:
            target_file = name
            break
    
    # if target datetime doesn't exists, return nothing
    if target_file == 'missing':
        if not supress_message:
            print('target datetime missing from cycle %s ' % target_cycle)
        return
    return target_file

def find_alias_frq(t_samp,t_sig):
    """
    Finds alias frequency (1/day) given a sampling period (t_sample) and signal period (t_signal)
    """
    n = 0
    af = abs(1 / t_sig - n / t_samp)
    while af > 1 / (2 * t_samp):
        n += 1
        af = abs( 1 / t_sig - n / t_samp)
    return af