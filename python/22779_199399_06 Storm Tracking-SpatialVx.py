import pandas as pd
from pointprocess import *
from lightning_setup import *
get_ipython().magic('matplotlib inline')

c = Region(city=cities['cedar'])
c.define_grid()

#choose an interesting day
t = '2012-08-19'

# get grid slices for that day optionally filtering out cloud to cloud lightning
box, tr = c.get_daily_grid_slices(t, filter_CG=dict(method='less_than', amax=-10), base=12)

# initialixe databox object
db = c.to_databox(box, tr[0:-1])

p = db.get_features()
computed = pd.HDFStore('cedar/features.h5')
computed['features_1km5min_thresh01_sigma3_minarea4_const5_{t}'.format(t=t)] = p
computed.close()

p = db.add_buffer(p)
computed.open()
computed['features_1km5min_thresh01_sigma3_minarea4_const5_buffered_{t}'.format(t=t)] = p
computed.close()

computed = pd.HDFStore('cedar/features.h5')
p = computed['features_1km5min_thresh01_sigma3_minarea4_const5_{t}'.format(t=t)]
computed.close()

ft = Features(p,db)

feature_locations(ft.titanize(), paths=True)

plt.figure(figsize=(7,7))
ft.windrose();

from scipy.ndimage.filters import gaussian_filter
from matplotlib import animation
from JSAnimation import IPython_display

cmap = cmap=plt.get_cmap('gnuplot_r', 5)
cmap.set_under('None')
gauss2d = np.array([gaussian_filter(box[i,:,:], 3) for i in range(box.shape[0])])

it0 = 72
by = 1
fig = plt.figure(figsize=(12, 8))

ax2 = background(plt.subplot(1, 1, 1, projection=ccrs.PlateCarree()))
im2, ax2 = c.plot_grid(gauss2d[it0], vmin=0.0001, vmax=.05, cmap=cmap, cbar=True, ax=ax2)

def init():
    im2.set_data(gauss2d[it0])
    return im2, 

def animate(i):
    im2.set_data(gauss2d[it0+i*by])
    try:
        ax2.scatter(p[tr[it0+i*by],:,'centroidX'],p[tr[it0+i*by],:,'centroidY'], 
                    c='green', edgecolors='None', s=50)
    except:
        pass
    fig.suptitle("it={i} time={t}".format(i=it0+i*by, t=tr[it0+i*by]), fontsize=18)
    return im2,  

animation.FuncAnimation(fig, animate, init_func=init, blit=True, frames=100, interval=100)

c = Region(city=cities['cedar'])
c.get_top(10)

c.define_grid(60)
box, tr = c.get_daily_grid_slices('2014-09-26')

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,4))

axes[0].plot(np.sum(box, axis=(1,2)))
axes[0].set_title("Flattened t axis");

axes[1].plot(np.sum(box, axis=(0,2)))
axes[1].set_title("Flattened y axis")

axes[2].plot(np.sum(box, axis=(0,1)))
axes[2].set_title("Flattened x axis");

c.define_grid(nbins=200, extents=[c.gridx[5], c.gridx[25], c.gridy[15], c.gridy[35]])
box, tr = c.get_grid_slices('2014-09-26', freq='5min')
box = box[100:250,:,:]
tr = tr[100:250]

from rpy2 import robjects 
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
pandas2ri.activate()
SpatialVx = importr('SpatialVx')
rsummary = robjects.r.summary

def import_r_tools(filename='r-tools.R'):
    import os
    from rpy2.robjects import pandas2ri, r, globalenv
    from rpy2.robjects.packages import STAP
    pandas2ri.activate()
    #path = os.path.dirname(os.path.realpath(__file__))
    path = './'
    with open(os.path.join(path,filename), 'r') as f:
        string = f.read()
    rfuncs = STAP(string, "rfuncs")
    return rfuncs

def dotvars(**kwargs):
    res = {}
    for k, v in kwargs.items():
        res[k.replace('_', '.')] = v
    return res

r_tools = import_r_tools()

d = {}
X, Y = np.meshgrid(c.gridx[0:-1], c.gridy[0:-1])
ll = np.array([X.flatten('F'), Y.flatten('F')]).T
for i in range(box.shape[0]-1):
    hold = SpatialVx.make_SpatialVx(box[i,:,:], box[i+1,:,:], loc=ll)
    look = r_tools.FeatureFinder_gaussian(hold, smoothpar=3,nx=199, ny=199, thresh=.01, **(dotvars(min_size=4)))
    try:
        x = rsummary(look, silent=True)[0]
    except:
        continue
    px = pandas2ri.ri2py(x)
    df0 = pd.DataFrame(px, columns=['centroidX', 'centroidY', 'area', 'OrientationAngle', 
                                  'AspectRatio', 'Intensity0.25', 'Intensity0.9'])
    df0['Observed'] = list(df0.index+1)
    m = SpatialVx.centmatch(look, criteria=3, const=5)
    p = pandas2ri.ri2py(m[12])
    df1 = pd.DataFrame(p, columns=['Forecast', 'Observed'])
    l = SpatialVx.FeatureMatchAnalyzer(m)
    try:
        p = pandas2ri.ri2py(rsummary(l, silent=True))
    except:
        continue
    df2 = pd.DataFrame(p, columns=['Partial Hausdorff Distance','Mean Error Distance','Mean Square Error Distance',
                                  'Pratts Figure of Merit','Minimum Separation Distance', 'Centroid Distance',
                                  'Angle Difference','Area Ratio','Intersection Area','Bearing', 'Baddeleys Delta Metric',
                                  'Hausdorff Distance'])
    df3 = df1.join(df2)

    d.update({tr[i]: pd.merge(df0, df3, how='outer')})
p =pd.Panel(d)

p =pd.Panel(d)
p

