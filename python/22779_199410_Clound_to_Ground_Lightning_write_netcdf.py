import os
import numpy as np
import pandas as pd
import xarray as xr

old_path = '/run/media/jsignell/WRF/Data/LIGHT/Data_1991-2009/'
new_path = '/run/media/jsignell/WRF/Data/LIGHT/raw/'
out_path = '/home/jsignell/erddapData/Cloud_to_Ground_Lightning/US/'

f = open('messed_up_old_files.txt')
l = f.readlines()
l=[fname.strip() for fname in l]
f.close()
l.sort()

def fname_to_ncfile(fname, old=False, new=True):
    if new:
        tstr = '{y}-{doy}'.format(y=fname[6:10], doy=fname[11:14])
        ncfile = str(pd.datetime.strptime(tstr, '%Y-%j').date()).replace('-','_')+'.nc'
        return(ncfile)

[d for d in pd.date_range('1991-01-01','2015-09-30').astype(str) if (d+'.nc').replace('-','_') not in os.listdir(out_path)]

d = {}
for y in range(1991, 2016):
    d.update({y: len([f for f in os.listdir(out_path) if str(y) in f])})
d

import os
out_path = '/home/jsignell/erddapData/Cloud_to_Ground_Lightning/'
little = []
for fname in os.listdir(out_path):
    if os.stat(out_path+fname).st_size <8000:
        little.append(fname)

import pandas as pd
new_path = '/run/media/jsignell/WRF/Data/LIGHT/raw/'
fnames = []
for fname in os.listdir(new_path):
    for l in little:
        t = pd.Timestamp(l.partition('.')[0].replace('_','-'))
        if '{y}.{doy:03d}'.format(y=t.year, doy=t.dayofyear) in fname:
            fnames.append(fname)

fname = l[4]

df = pd.read_csv(old_path+fname, delim_whitespace=True, header=None, names=['D', 'T','lat','lon','amplitude','strokes'])

df['T'][7430175]

df = df.drop(7430175)

s = pd.to_datetime(df['D']+' '+df['T'], errors='coerce')

df = df[['time', 'lat', 'lon', 'amplitude', 'strokes']]

df.head()

df[df.time.isnull()]

df['strokes'] = df['strokes'].astype(int)

df.strokes[df.strokes == '503/12/08']

df = df.drop(724017)

days = np.unique(df.time.apply(lambda x: x.date()))
for day in days:
    df0 = df[(df.time >= day) & (df.time < day+pd.DateOffset(1))]
    df0 = df0.reset_index()
    df0.index.name = 'record'
    write_day(df0, out_path)
    print day

import os
import numpy as np
import pandas as pd
import xarray as xr

new_path = '/run/media/jsignell/WRF/Data/LIGHT/raw/'
out_path = '/home/jsignell/erddapData/Cloud_to_Ground_Lightning/'

def new_files(path, fname, out_path):
    df = pd.read_csv(path+fname, delim_whitespace=True, header=None, parse_dates={'time':[0,1]})
    df = df.drop(5, axis=1)
        
    df.columns = ['time', 'lat', 'lon', 'amplitude','strokes',
                  'semimajor','semiminor','ratio','angle','chi_squared','nsensors','cloud_ground']
    df.index.name = 'record'
    
    attrs = {'semimajor': {'long_name': 'Semimajor Axis of 50% probability ellipse for each flash',
                           'units': 'km'},
             'semiminor': {'long_name': 'Semiminor Axis of 50% probability ellipse for each flash',
                           'units': 'km'},
             'ratio': {'long_name': 'Ratio of Semimajor to Semiminor'},
             'angle': {'long_name': 'Angle of 50% probability ellipse from North',
                       'units': 'Deg'},
             'chi_squared': {'long_name': 'Chi-squared value of statistical calculation'},
             'nsensors': {'long_name': 'Number of sensors reporting the flash'},
             'cloud_ground': {'long_name': 'Cloud_to_Ground or In_Cloud Discriminator'}}


    ds = df.to_xarray()
    ds.set_coords(['time','lat','lon'], inplace=True)
    if df.shape[0] < 5:
        chunk=1
    else:
        chunk = min(df.shape[0]/5, 1000)
    for k, v in attrs.items():
        ds[k].attrs.update(v)
        if k == 'cloud_ground':
            ds[k].encoding.update({'dtype': 'S1'})
        elif k == 'nsensors':
            ds[k].encoding.update({'dtype': np.int32, 'chunksizes':(chunk,),'zlib': True})
        else:
            ds[k].encoding.update({'dtype': np.double,'chunksizes':(chunk,),'zlib': True})

    ds.amplitude.attrs.update({'units': 'kA',
                               'long_name': 'Polarity and strength of strike'})
    ds.amplitude.encoding.update({'dtype': np.double,'chunksizes':(chunk,),'zlib': True})
    ds.strokes.attrs.update({'long_name': 'multiplicity of flash'})
    ds.strokes.encoding.update({'dtype': np.int32,'chunksizes':(chunk,),'zlib': True})
    ds.lat.attrs.update({'units': 'degrees_north',
                         'axis': 'Y',
                         'long_name': 'latitude',
                         'standard_name': 'latitude'})
    ds.lat.encoding.update({'dtype': np.double,'chunksizes':(chunk,),'zlib': True})
    ds.lon.attrs.update({'units': 'degrees_east',
                         'axis': 'X',
                         'long_name': 'longitude',
                         'standard_name': 'longitude'})
    ds.lon.encoding.update({'dtype': np.double,'chunksizes':(chunk,),'zlib': True})
    ds.time.encoding.update({'units':'seconds since 1970-01-01', 
                             'calendar':'gregorian',
                             'dtype': np.double,'chunksizes':(chunk,),'zlib': True})

    ds.attrs.update({ 'title': 'Cloud to Ground Lightning',
                      'institution': 'Data from NLDN, hosted by Princeton University',
                      'references': 'https://ghrc.nsstc.nasa.gov/uso/ds_docs/vaiconus/vaiconus_dataset.html',
                      'featureType': 'point',
                      'Conventions': 'CF-1.6',
                      'history': 'Created by Princeton University Hydrometeorology Group at {now} '.format(now=pd.datetime.now()),
                      'author': 'jsignell@princeton.edu',
                      'keywords': 'lightning'})

    date = df.time[len(df.index)/2]
    ds.to_netcdf('{out_path}{y}_{m:02d}_{d:02d}.nc'.format(out_path=out_path, y=date.year, m=date.month, d=date.day), 
                 format='netCDF4', engine='netcdf4')

for fname in fnames:
    try:
        new_files(new_path, fname, out_path)
        print fname
    except:
        f = open('messed_up_new_files.txt', 'a')
        f.write(fname+'\n')
        f.close()

import os
import numpy as np
import pandas as pd
import xarray as xr

old_path = '/run/media/jsignell/WRF/Data/LIGHT/Data_1991-2009/'
out_path = '/home/jsignell/erddapData/Cloud_to_Ground_Lightning/'
    
def write_day(df, out_path):

    ds = df.drop('index', axis=1).to_xarray()
    ds.set_coords(['time','lat','lon'], inplace=True)
    
    ds.amplitude.attrs.update({'units': 'kA',
                               'long_name': 'Polarity and strength of strike'})
    ds.amplitude.encoding.update({'dtype': np.double})
    ds.strokes.attrs.update({'long_name': 'multiplicity of flash'})
    ds.strokes.encoding.update({'dtype': np.int32})
    ds.lat.attrs.update({'units': 'degrees_north',
                         'axis': 'Y',
                         'long_name': 'latitude',
                         'standard_name': 'latitude'})
    ds.lat.encoding.update({'dtype': np.double})
    ds.lon.attrs.update({'units': 'degrees_east',
                         'axis': 'X',
                         'long_name': 'longitude',
                         'standard_name': 'longitude'})
    ds.lon.encoding.update({'dtype': np.double})
    ds.time.encoding.update({'units':'seconds since 1970-01-01', 
                             'calendar':'gregorian',
                             'dtype': np.double})

    ds.attrs.update({ 'title': 'Cloud to Ground Lightning',
                      'institution': 'Data from NLDN, hosted by Princeton University',
                      'references': 'https://ghrc.nsstc.nasa.gov/uso/ds_docs/vaiconus/vaiconus_dataset.html',
                      'featureType': 'point',
                      'Conventions': 'CF-1.6',
                      'history': 'Created by Princeton University Hydrometeorology Group at {now} '.format(now=pd.datetime.now()),
                      'author': 'jsignell@princeton.edu',
                      'keywords': 'lightning'})


    date = df.time[len(df.index)/2]
    
    ds.to_netcdf('{out_path}{y}_{m:02d}_{d:02d}.nc'.format(out_path=out_path, y=date.year, m=date.month, d=date.day), 
                 format='netCDF4', engine='netcdf4')

def old_files(path, fname, out_path):
    df = pd.read_csv(path+fname, delim_whitespace=True, header=None, names=['D', 'T','lat','lon','amplitude','strokes'],
                     parse_dates={'time':[0,1]})
    
    days = np.unique(df.time.apply(lambda x: x.date()))
    for day in days:
        df0 = df[(df.time >= day) & (df.time < day+pd.DateOffset(1))]
        df0 = df0.reset_index()
        df0.index.name = 'record'
        write_day(df0, out_path)
        
'''
for fname in os.listdir(old_path):
    try:
        old_files(old_path, fname, out_path)
    except:
        f = open('messed_up_old_files.txt', 'a')
        f.write(fname+'\n')
        f.close()
'''

if df0.shape[0] >1000:
    chunks={'chunksizes':(1000,),'zlib': True}
else:
    chunks={}
for v in ds.data_vars.keys()+ds.coords.keys():
    if v =='strokes':
        continue
    ds[v].encoding.update(chunks)

