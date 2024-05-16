from plotly.offline import plot, iplot
import cufflinks as cf, plotly.plotly as py, json, pandas as pd, numpy as np
import ClassifierCapsuleClass as ClfClass, ClassiferHelperAPI as CH
import RegressionCapsuleClass as RgrClass
import plotly.graph_objs as go
from collections import Counter
cf.go_offline()
import folium
from folium.plugins import MarkerCluster
from folium import plugins
import importlib
importlib.reload(CH)

with open("../data/Flickr_EXIF_full.json" , "r") as exif_fl:
    exif_d = json.load(exif_fl)

df = pd.DataFrame.from_dict(exif_d).transpose()
df['datetime'] = pd.to_datetime(df['date'])
df['date'] = df['datetime'].dt.date

df['date'] = pd.to_datetime(df.date)
df['year_month'] = df.date.dt.strftime("%m-%Y")
df['year'] = df.date.dt.strftime("%Y")
df['month'] = df.date.dt.strftime("%m")
df['week'] = df.date.dt.week

df = df[(df['year'] > '1970')]
df.sort_values(by='date', inplace=True)

df['year_month'].iplot(kind='histogram')

df['year'].iplot(kind='histogram')

df['month'].iplot(kind='histogram')

df_non_zero = df[(df['lat'] != 0) & (df['long'] != 0)][['lat', 'long']]

map_loc = folium.Map(location=[38.64264252590279, -51.622090714285676],tiles='Stamen Terrain',zoom_start=2)

recds = df_non_zero.to_records()
for i in range(0,len(recds)):
    folium.Marker([recds[i][1],recds[i][2]],
              icon=folium.Icon(color='green',icon='info-sign'), popup=recds[i][0]
              ).add_to(map_loc)
    
map_loc.save(outfile='../FinalResults/FlickrLocations.html')

locs = [(recd[1],recd[2]) for recd in recds]
heatmap_map = folium.Map(location=[38.64264252590279, -51.622090714285676],tiles='Stamen Terrain', zoom_start=2)
hm = plugins.HeatMap(locs)
heatmap_map.add_children(hm)

heatmap_map.save("../FinalResults/heatMap_Flickr.html")

df_new = df.groupby(['year','week']).count()['date']
df_dict = df_new.to_dict()
df_tups = [(' wk#'.join(map(str,key)), df_dict[key]) for key in df_dict.keys()]
df_tups = sorted(df_tups, key=lambda x : (x[0], x[1]))
x = ["'"+tup[0][2:] for tup in df_tups]
y = [tup[1] for tup in df_tups]
trace1 = go.Bar(
            x = x,
            y = y
        )

data = [trace1]
layout = go.Layout(
    xaxis=dict(tickangle=45)
)
fig = dict(data=data, layout=layout)
py.iplot(fig)

df_train.iplot(kind='histogram',histnorm='probability')

clfArgs = {'dummy' : {'strategy' : 'most_frequent'},
            'bayesian' : {'fit_prior' : True},
            'logistic' : {'penalty' : 'l2'},
            'svm' : {'kernel' : 'rbf','probability' : True},
            'dtree' : {'criterion' : 'entropy'},
            'random_forests' : {'n_estimators' : 10 },
            'ada_boost' : {'n_estimators' : 50 }}

regrArgs = {'linear' : {'fit_intercept' : True},
            'ridge' : {'fit_intercept' : True},
            'lasso' : {'fit_intercept' : True},
            'elastic_net' : {'fit_intercept' : True},
            'svr' : {'fit_intercept' : True},
            'dtree_regressor' : {'fit_intercept' : True}}
# ['dummy', 'bayesian', 'logistic', 'svm', 'dtree', 'random_forests', 'ada_boost']:
# ['linear', 'ridge', 'lasso', 'svr', 'dtree_regressor', 'elastic_net']:
for rgrMeth in ['ada_boost']:
    train_data_fl = "/tmp/training_fl.csv"
    test_data_fl = "/tmp/training_fl.csv"
    obj, results = CH.trainTestClf(train_data_fl, test_data_fl, rgrMeth, 'beauty', None, clfArgs)

    
    df_train = pd.DataFrame(list(results.items()), columns=['GID', "Probability"])
    df_train.index = df_train.GID
    df_train.drop(['GID'],1,inplace=True)
    
    test_data_fl = "/tmp/testing_fl.csv"
    obj, results = CH.trainTestClf(train_data_fl, test_data_fl, rgrMeth, 'beauty', None, clfArgs)
    df_test = pd.DataFrame(list(results.items()), columns=['GID', "Probability"])

    df_test.index = df_test.GID
    df_test.drop(['GID'],1,inplace=True)
    
    test_data_fl = "/tmp/testing_fl_bing.csv"
    obj, results = CH.trainTestClf(train_data_fl, test_data_fl, rgrMeth, 'beauty', None, clfArgs)
    df_test2 = pd.DataFrame(list(results.items()), columns=['GID', "Probability"])

    df_test2.index = df_test2.GID
    df_test2.drop(['GID'],1,inplace=True)
    
    trace1 = go.Histogram(
        x=df_train['Probability'],
        opacity=0.75,
        histnorm='probability',
        name='Pred. probability - Training',
        marker=dict(
            color='grey')
    )
    trace2 = go.Histogram(
        x=df_test['Probability'],
        opacity=0.75,
        histnorm='probability',
        name='Pred. probability - Flickr',
        marker=dict(
            color='blue')
    )
    trace3 = go.Histogram(
        x=df_test2['Probability'],
        opacity=0.75,
        histnorm='probability',
        name='Pred. probability - Bing',
        marker=dict(
            color='lightgreen')
    )

    data = [trace1, trace3, trace2]

    layout = go.Layout(
        title='PDF %s' %rgrMeth,
        xaxis=dict(
            title='Share rate'
        ),
        yaxis=dict(
            title='P(X)'
        ),
        barmode='overlay'
    )

    fig = go.Figure(data=data, layout=layout)
    f = py.iplot(fig)
    print(f.embed_code)

'''
'symmetry',
 'hsv_itten_std_v',
 'arousal',
 'contrast',
 'pleasure',
 'hsv_itten_std_h',
 'hsv_itten_std_s',
 'dominance'

'''

with open("../data/GZC_beauty_features.json", "r") as fl1:
    gzc_bty = json.load(fl1)
    
with open("../data/ggr_beauty_features.json") as fl2:
    ggr_bty = json.load(fl2)
    
with open("../data/Flickr_Beauty_features.json") as fl3:
    flickr_zbra_bty = json.load(fl3)
    
with open("../data/Flickr_Bty_Giraffe.json") as fl4:
    flickr_giraffe_bty = json.load(fl4)

def build_box_plot(ftr, names, *datasets):
    traces = []
    i = 0
    for dataset in datasets:
        ftrset = [dataset[img][ftr] for img in dataset.keys()]
        traces.append(go.Box(x=ftrset,name=names[i]))
        i += 1
    return traces

layout = go.Layout(title = "Symmetry")
data = build_box_plot('symmetry',['GZC', 'GGR', "Flickr Zebra", "Flickr Giraffe"], gzc_bty, ggr_bty, flickr_zbra_bty, flickr_giraffe_bty)
fig = go.Figure(data=data,layout=layout)
iplot(fig)

layout = go.Layout(title = "contrast")
data = build_box_plot('contrast',['GZC', 'GGR', "Flickr Zebra", "Flickr Giraffe"], gzc_bty, ggr_bty, flickr_zbra_bty, flickr_giraffe_bty)
fig = go.Figure(data=data,layout=layout)
iplot(fig)

layout = go.Layout(title = "hsv_itten_std_h")
data = build_box_plot('hsv_itten_std_h',['GZC', 'GGR', "Flickr Zebra", "Flickr Giraffe"], gzc_bty, ggr_bty, flickr_zbra_bty, flickr_giraffe_bty)
fig = go.Figure(data=data,layout=layout)
iplot(fig)

layout = go.Layout(title = "hsv_itten_std_s")
data = build_box_plot('hsv_itten_std_s',['GZC', 'GGR', "Flickr Zebra", "Flickr Giraffe"], gzc_bty, ggr_bty, flickr_zbra_bty, flickr_giraffe_bty)
fig = go.Figure(data=data,layout=layout)
iplot(fig)

layout = go.Layout(title = "hsv_itten_std_v")
data = build_box_plot('hsv_itten_std_v',['GZC', 'GGR', "Flickr Zebra", "Flickr Giraffe"], gzc_bty, ggr_bty, flickr_zbra_bty, flickr_giraffe_bty)
fig = go.Figure(data=data,layout=layout)
iplot(fig)

layout = go.Layout(title = "arousal")
data = build_box_plot('arousal',['GZC', 'GGR', "Flickr Zebra", "Flickr Giraffe"], gzc_bty, ggr_bty, flickr_zbra_bty, flickr_giraffe_bty)
fig = go.Figure(data=data,layout=layout)
iplot(fig)

layout = go.Layout(title = "dominance")
data = build_box_plot('dominance',['GZC', 'GGR', "Flickr Zebra", "Flickr Giraffe"], gzc_bty, ggr_bty, flickr_zbra_bty, flickr_giraffe_bty)
fig = go.Figure(data=data,layout=layout)
iplot(fig)

layout = go.Layout(title = "pleasure")
data = build_box_plot('pleasure',['GZC', 'GGR', "Flickr Zebra", "Flickr Giraffe"], gzc_bty, ggr_bty, flickr_zbra_bty, flickr_giraffe_bty)
fig = go.Figure(data=data,layout=layout)
iplot(fig)



