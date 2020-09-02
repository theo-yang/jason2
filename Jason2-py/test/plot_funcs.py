"""
Plotting Functions for Jason2_py test notebook

Dependencies
------------
GeoViews 1.6.6
HoloViews 1.13.3
Bokeh 2.1.1
Cartopy
Numpy
"""
# Imports
import holoviews as hv
import cartopy.crs as ccrs
import geoviews as gv
import geoviews.feature as gf
import numpy as np
gv.extension('bokeh')
print('Plotting Modules:')
print('GeoViews {}'.format(gv.__version__))
print('HoloViews {}'.format(hv.__version__))


# plotting functions
def fft_2D_plt(dataset,title):
    """
    plot power 2D spectrum from dataset 
    """
    nu_vector = dataset['wavenumbers']
    f_vector = dataset['frequencies']
    fft_2D = dataset['fft 2D']
    return hv.Image((nu_vector,f_vector,np.log10(fft_2D)), 
                     vdims=hv.Dimension('z'), 
                     kdims=['cpkm','cpd']).opts(colorbar=True,
                                                  height=450,
                                                  width=450,
                                                  tools=['hover'],
                                                  cmap ='viridis',
                                                  title = title,
                                                  xticks=5,
                                                  yticks=5,
                                                  clim = (-3,-2),
                                                  colorbar_opts ={'width':45},
                                                  clabel= 'log₁₀(PSD)')

def track_plt(dataset,title, bounds=(-166,13.5,-154, 25)):
    """
    plot Jason-2 tracks from dataset with coordinates: N = [-90, 90], E = [-180, 180]
    """
    lon_range = (np.array(dataset['lon range']) % 180) - (np.array(dataset['lon range']) >= 180) * 180
    lat_range = np.array(dataset['lat range'])
    coast = gf.coastline.geoms('50m', bounds=bounds).opts(color='black',
                                                                        xticks=5,
                                                                        yticks=5,
                                                                        yformatter='%f °N',
                                                                        xformatter = '%f °E') 
    track = hv.Path((lon_range,lat_range)).opts(color = 'black')
    return (coast * track ).opts(projection=ccrs.PlateCarree(),
                                 title = title,
                                 height=200,
                                 width=200)


def line_plt(dataset,dimension,val,title=None):
    """
    if "f" : power vs frq at tidal wavenumber
    if "nu" : power vs wavenumber at tidal alias frequency
    val (float) : value wavenumber (if dimension == "f") or frequency (if dimension == "nu") at which to plot cross secion
    """
    nu_vector = dataset['wavenumbers']
    f_vector = dataset['frequencies']
    
    if dimension == "f":
        idx = (np.abs(nu_vector - val)).argmin()
        data = dataset['fft 2D']
        vector = f_vector
        kdims=['cpd','m² cpkm⁻¹ cpd⁻¹']
        if title == None:
            title = 'wavenumber = %f cpkm'%nu_vector[idx]
    
    if dimension == "nu":
        idx = (np.abs(f_vector - val)).argmin()
        data = dataset['fft 2D'].T
        vector = nu_vector
        kdims = ['cpkm','m² cpkm⁻¹ cpd⁻¹']
        if title == None:
            title = 'frequency = %f cpd'%f_vector[idx]
            
    return hv.Curve((vector,data[:,idx]), 
                    kdims=kdims).opts(
                    tools=['hover'],
                    height=275,
                    width=275,
                    xticks=5,
                    yticks=5,
                    fontsize={'title':10},
                    title = title)

def U_plt(current_dataset,ssha_dataset,title,bounds=(-166,13.5,-154, 25)):
    """
    plot current velocities magntiudes averaged from 2008 to 2016 and Jason-2 tracks from dataset
    """
    # plot coastline
    coast = gf.coastline.geoms('50m', bounds=bounds).opts(color='black',
                                                                        xticks=5,
                                                                        yticks=5,
                                                                        yformatter='%f °N',
                                                                        xformatter = '%f °E') 
    # plot surface velocity magnitudes
    lats = current_dataset['lat'].reshape(len(current_dataset['lat']))
    _lons = current_dataset['lon'].reshape(len(current_dataset['lon']))
    lons = (_lons % 180) - (_lons >= 180) * 180
                                 
    u = np.mean(current_dataset['u'],axis=0)
    v = np.mean(current_dataset['v'],axis=0)
    magnitude = np.sqrt(u ** 2 + v ** 2)
    vel = hv.Image((lons,lats,magnitude)).opts(colorbar=True,
                                             cmap ='viridis',
                                             clabel='m/s',
                                             projection=ccrs.PlateCarree(),
                                             tools=['hover'])
    # plot tracks
    lon_range = (np.array(ssha_dataset['lon range']) % 180) - (np.array(ssha_dataset['lon range']) >= 180) * 180
    lat_range = ssha_dataset['lat range']
    track = hv.Path((lon_range ,lat_range)).opts(color = 'black')
    
    return (coast * vel * track).opts(projection=ccrs.PlateCarree(),
                                 title = title,
                                 height=300,
                                 width=500)