import pathlib
import datetime

# data
import netCDF4
import pandas
import numpy as np

# plots
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean.cm

# stats
import statsmodels.stats.outliers_influence
import statsmodels.sandbox.regression.predstd
import statsmodels.graphics.regressionplots
import statsmodels.regression
import statsmodels.tsa.seasonal


# interaction
import tqdm
from IPython.display import YouTubeVideo, display

get_ipython().magic('matplotlib inline')

# this opens the netCDF file. 
path = pathlib.Path('../../../data/sat/dt_global_merged_msla_h_merged_nc4.nc')
ds = netCDF4.Dataset(str(path))
# and reads all the relevant variable
# Sea-level anomaly
sla = ds.variables['sla'][:]
# time 
time = netCDF4.num2date(ds.variables['time'][:], ds.variables['time'].units)
# location
lon = ds.variables['lon'][:]
lat = ds.variables['lat'][:]

# compute days, seconds * (hours/second)  * (days / hour) -> days
days = np.array([t.timestamp() * 1/3600.0 * 1/24.0 for t in time]) 
# compute relative to first measurement
days = days - datetime.datetime(1970, 1, 1, 0, 0).timestamp()
years = days/365.25

YouTubeVideo('XU0CZlbr4yY')

fig, ax = plt.subplots(figsize=(13, 8))
# plot on a dark background
ax.set_facecolor((0.2, 0.2, 0.2))
ax.set_xlabel('longitude')
ax.set_ylabel('latitude')
# split up by longitude=200 and merge back
sl = np.ma.hstack([sla[0,:,800:], sla[0,:,:800]])
# show the sea-level anamolies
im = ax.imshow(sl, cmap=cmocean.cm.delta_r, origin='top', vmin=-0.5, vmax=0.5, extent=(lon[800] - 360, lon[800], lat[0], lat[-1]))
# room for a colormap
divider = make_axes_locatable(ax)
# append the colormap axis
cax = divider.append_axes("right", size="5%", pad=0.05)
# show the colormap
plt.colorbar(im, cax=cax, label='sea surface height [m]')
# squeeze toegether
fig.tight_layout()
# inlay for timeseries, transparent background
ax_in = plt.axes([0.65, 0.2, .2, .2], facecolor=(0.8, 0.8, 0.8, 0.5))
series = np.mean(sla, (1, 2))
# plot the line
ax_in.plot(time, series)
ax_in.xaxis.set_visible(False)
ax_in.yaxis.set_visible(False)
# add the moving dot
dot, = ax_in.plot(time[0], series[0], 'ro')

# export to movie
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(
    title='{} {}'.format(ds.variables['sla'].long_name, ds.variables['sla'].units), 
    artist='Fedor Baart',
    comment='Sea-level rise over the period 1993-2017'
)
writer = FFMpegWriter(fps=15, metadata=metadata)
with writer.saving(fig, "avisosla.mp4", 100):
    for i in range(time.shape[0]):
        # update the data and title
        ax.set_title(time[i])
        sl = np.ma.hstack([sla[i,:,800:], sla[i,:,:800]])
        im.set_data(sl)
        dot.set_data(time[i], series[i])
        # snapshot
        writer.grab_frame()

# mean sealevel in m -> cm
mean_sla = np.mean(sla, axis=(1, 2)) * 100.0

# convert times to year
# create a linear model sla ~ constant + time
exog_linear = statsmodels.regression.linear_model.add_constant(years)
# create a quadratic model sla ~ constant + time + time^2
exog_quadratic = statsmodels.regression.linear_model.add_constant(np.c_[years, years**2])

linear_model = statsmodels.regression.linear_model.OLS(mean_sla, exog=exog_linear)
linear_fit = linear_model.fit()
quadratic_model = statsmodels.regression.linear_model.OLS(mean_sla, exog=exog_quadratic)
quadratic_fit = quadratic_model.fit()



if (linear_fit.aic < quadratic_fit.aic):
    print('The linear model is a higher quality model (smaller AIC) than the quadratic model.')
else:
    print('The quadratic model is a higher quality model (smaller AIC) than the linear model.')
if (quadratic_fit.pvalues[2] < 0.05):
    print('The quadratic term is bigger than we would have expected under the assumption that there was no quadraticness.')
else:
    print('Under the assumption that there is no quadraticness, we would have expected a quadratic term as big as we have seen.')
    
# choose the model, prefer the most parsimonuous when in doubt.
if  (linear_fit.aic < quadratic_fit.aic) or quadratic_fit.pvalues[2] >= 0.05:
    display(quadratic_fit.summary(title='Quadratic model (not used)'));    
    display(linear_fit.summary(title='Linear model (used)'))
    print('The linear model is preferred as the quadratic model is not both significant and of higher quality.')
    fit = linear_fit
    model = linear_model
else:
    display(linear_fit.summary(title='Linear model (not used)'))
    display(quadratic_fit.summary(title='Quadratic model (used)'));    
    print('The quadratic model is preferred as it is both significantly better and of higher quality.')
    fit = quadratic_fit
    model = quadratic_model

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_title('Global sea-level rise')
ax.plot(time, mean_sla)
ax.set_xlabel('Time')
ax.set_ylabel('Sea-level anomaly, global average [cm]')


# add the prediction interval
prstd, iv_l, iv_u = statsmodels.sandbox.regression.predstd.wls_prediction_std(fit)
# plot the prediction itnervals
for i in np.linspace(0.1, 1.0, num=10):
    ax.fill_between(
        time, 
        (iv_l - fit.fittedvalues)*i + fit.fittedvalues, 
        (iv_u - fit.fittedvalues)*i + fit.fittedvalues, 
        alpha=0.1,
        color='green'
    )
# get the confidence interval for the fitted line  from the outlier table
table = statsmodels.stats.outliers_influence.summary_table(fit)
st, data, ss2 = table
predict_mean_ci_low, predict_mean_ci_upp = data[:,4:6].T

# plot the confidence intervals
for i in np.linspace(0.1, 1.0, num=10):
    ax.fill_between(
        time, 
        (predict_mean_ci_low - fit.fittedvalues)*i + fit.fittedvalues, 
        (predict_mean_ci_upp - fit.fittedvalues)*i + fit.fittedvalues, 
        alpha=0.1,
        color='black'
    )
    ax.plot(time, fit.fittedvalues);


index = pandas.date_range(time[0], periods=len(time), freq='M')
df = pandas.DataFrame(index=index, data=dict(ssh=mean_sla))
model = statsmodels.tsa.seasonal.seasonal_decompose(df.ssh)

fig = model.plot()
fig.set_size_inches(8, 5)
fig.axes[0].set_title('Seasonal decomposition of sea-level anomalies');

# this allows us to create a trend without the seasonal effect
fig, ax = plt.subplots()
ax.plot(time, model.trend + model.resid)
ax.set_xlabel('time')
ax.set_ylabel('deseasonalized sea-level anomaly [cm]');

fig, ax = plt.subplots(figsize=(13,8))
# mean, m -> cm
mean_lat_sla = sla.mean(axis=2) * 100

# make sure we use a symetric range, because we're using a divergent colorscale
vmax = np.abs(mean_lat_sla).max()
pc = ax.pcolor(time, lat, mean_lat_sla.T, cmap=cmocean.cm.delta_r, vmin=-vmax, vmax=vmax)
ax.set_xlabel('time')
ax.set_ylabel('latitude [degrees N]')
ax.set_title('Sea-level anomaly [cm] as a function of time and latitude')
plt.colorbar(pc, ax=ax);

def fit(years, sla):
    # no const (fit through 0)
    exog = statsmodels.regression.linear_model.add_constant(years)
    linear_model = statsmodels.regression.linear_model.OLS(sla, exog)
    fit = linear_model.fit()
    const, trend = fit.params
    # cm/year * 100/1 year/century) -> cm / century
    return trend * 100

def make_latitude_plot(years, lat, mean_lat_sla):
    # compute trend per latitude
    trends = np.array([fit(years, series) for series in mean_lat_sla.T])

    # create a figure for zoomed out and zoomed in plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    # global trend
    ax = axes[0]
    ax.plot(trends, lat, alpha=0.4)
    ax.set_ylim(-80, 80)
    ax.set_xlim(0, 40)
    ax.set_xlabel('slr [cm/century]')
    ax.set_ylabel('latitude [degrees]')

    # pick a location in NL
    (idx,) = np.nonzero(lat == 52.125)
    ax.plot(trends[idx], 52.125, 'r+')
    text = "Scheveningen: %.1f" % (trends[idx], )
    ax.annotate(text, (trends[idx], 53.125))
    ax.set_title('Sea-level rise [1993-2016] averaged per latitude');

    # same plot, zoomed in to NL
    ax = axes[1]
    ax.plot(trends, lat, alpha=0.4, label='sea-level rise [cm/century]')
    ax.set_ylim(51, 54)
    ax.set_xlim(10, 30)
    ax.set_xlabel('slr [cm/century]')
    ax.set_ylabel('latitude [degrees]')
    ax.set_title('Sea-level rise [1993-2016] averaged per latitude')
    ax.legend(loc='best')
    locations = [
        (53.375, 'Schiermonnikoog'),
        (52.625, 'Egmond aan Zee'),
        (51.375, 'Vlissingen')
    ]    
    for lat_i, label in locations: 
        (idx,) = np.nonzero(lat == lat_i)
        ax.plot(trends[idx], lat_i, 'r+')
        text = "%s: %.1f" % (label, trends[idx])
        ax.annotate(text, (trends[idx], lat_i))  
    return fig, ax

fig, ax = make_latitude_plot(years, lat, mean_lat_sla)

index = pandas.date_range(time[0], periods=len(time), freq='M')

def decompose(series):
    df = pandas.DataFrame(index=index, data=dict(ssh=series))
    model = statsmodels.tsa.seasonal.seasonal_decompose(df.ssh)
    return model.trend 
decompositions = []
for series in mean_lat_sla[:, 90:-90].T:
    decomposed = decompose(series)
    decompositions.append(np.array(decomposed))
    
    
    

import matplotlib.cm
fig, ax = plt.subplots(figsize=(13, 8))
vmax = np.vstack(decompositions).max()

im = ax.imshow(np.vstack(decompositions), aspect=0.5, cmap=cmocean.cm.deep_r, origin='top') 

xlabels = [
    time[loc].strftime('%Y-%m') if (loc >= 0 and loc < 300) else '' 
    for loc 
    in ax.xaxis.get_ticklocs().astype('int')
]
_ = ax.xaxis.set_ticklabels(xlabels)
ylabels = [
    "%.0f" % (lat[90:-90][loc], ) if loc < 540 else ''
    for loc 
    in ax.yaxis.get_ticklocs().astype('int')
]
y = ax.yaxis.set_ticklabels(ylabels)

ax.set_title('Sea level (monthly averaged, std trend) over latitude')
ax.set_xlabel('time')
ax.set_ylabel('latitude')
plt.colorbar(im, ax=ax);

