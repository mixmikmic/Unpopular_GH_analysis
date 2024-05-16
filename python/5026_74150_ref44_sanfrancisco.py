from pymongo import MongoClient
from collections import defaultdict
import scipy.io as sio
import scipy.sparse as sp
import gensim
import pandas as pd
import sys
sys.path.append('..')
import persistent as p
from scipy.spatial import ConvexHull
import shapely.geometry as geom
import folium

cl = MongoClient()
db = cl.combined
fullcity, city = 'San Francisco', 'sanfrancisco'
scaler = p.load_var('../sandbox/paper-models/{}.scaler'.format(city))
venue_infos = {}
for venue in db.venues.find({ "bboxCity": fullcity}):
    venue_infos[venue['_id']] = (venue['coordinates'], venue['name'], None if len(venue['categories']) == 0 else venue['categories'][0])

user_visits = defaultdict(lambda : defaultdict(int))
venue_visitors = defaultdict(lambda : defaultdict(int))

for vid in venue_infos:
    for checkin in  db.checkins.find({"venueId": vid}):
        uid = checkin['foursquareUserId']
        user_visits[uid][vid] += 1
        venue_visitors[vid][uid] += 1

user_visits = {us: {vn: c for vn, c in usval.items()} for us, usval in user_visits.items()}
venue_visitors = {us: {vn: c for vn, c in usval.items()} for us, usval in venue_visitors.items()}

still_work_todo = True
print(len(user_visits), len(venue_visitors))
while still_work_todo:
    users_to_remove = set()
    new_users = {}
    for u, vst in user_visits.items():
        rvst = {vid: num for vid, num in vst.items() if vid in venue_visitors}
        num_visit = len(rvst.values())
        if num_visit < 5:
            users_to_remove.add(u)
        else:
            new_users[u] = dict(rvst)
    #print(len(users_to_remove), len(new_users))
    venues_to_remove = set()
    new_venues = {}
    for u, vst in venue_visitors.items():
        rvst = {user: num for user, num in vst.items() if user in new_users}
        num_visit = sum(rvst.values())
        if num_visit < 10:
            venues_to_remove.add(u)
        else:
            new_venues[u] = dict(rvst)
    #print(len(venues_to_remove), len(new_venues))
    user_visits = new_users
    venue_visitors = new_venues
    still_work_todo = len(users_to_remove) > 0 or len(venues_to_remove) > 0
print('At the end of the pruning, we are left with {} unique users who have made {} checkins in {} unique venues'.format(len(user_visits), sum(map(len, user_visits.values())), len(venue_visitors)))

sorted_venues = {v: i for i, v in enumerate(sorted(venue_visitors))}
sorted_users = sorted(user_visits) 
write_doc = lambda vst: [v for v, c in vst.items() for _ in range(c)]

texts = [write_doc(user_visits[uid]) for uid in sorted_users]

dictionary = gensim.corpora.Dictionary(texts)
print(dictionary)
dictionary.filter_extremes(no_below=0, no_above=.08, keep_n=None)
print(dictionary)
corpus = [dictionary.doc2bow(text) for text in texts]
gensim.corpora.MmCorpus.serialize('{}_corpus.mm'.format(city), corpus)
dictionary.save('{}_venues_dict.dict'.format(city))

get_ipython().magic('time model = gensim.models.ldamulticore.LdaMulticore(corpus, id2word=dictionary, workers=10, num_topics=20)')

venues_per_topic = 10
num_topics = 5
index_name = [['Topic {}'.format(i+1) for i in range(num_topics) for _ in range(venues_per_topic) ],
          [str(_) for i in range(num_topics) for _ in range(venues_per_topic) ]]

res =[]

for topic in range(num_topics):
    for vidx, weight in model.get_topic_terms(topic, venues_per_topic):
        vid = dictionary.id2token[vidx]
        name = venue_infos[vid][1]
        link = 'https://foursquare.com/v/'+vid
        cat = venue_infos[vid][2]['name']
        res.append([name, cat, weight, link])

pd.DataFrame(res, index=index_name, columns=['Venue', 'Category', 'Weight', 'URL'])

weights = np.array([[_[1] for _ in model.get_topic_terms(i, 500000)] for i in range(20)])

top_venues_per_topic = (weights.cumsum(-1)>.15).argmax(1)
top_venues_per_topic

mf = folium.Map(location=[37.76,-122.47])
feats=[]
for topic, num_venues in enumerate(top_venues_per_topic):
    pts=[]
    for vidx, _ in model.get_topic_terms(topic, num_venues):
        vid = dictionary.id2token[vidx]
        pts.append(venue_infos[vid][0])
    pts = np.array(pts)

    spts = scaler.transform(pts)

    hull = pts[ConvexHull(spts).vertices, :]

    geojson_geo = geom.mapping(geom.Polygon(hull))
    feats.append({ "type": "Feature", "geometry": geojson_geo, "properties": {"fill": "#BB900B"}})

_=folium.GeoJson({"type": "FeatureCollection", "features": feats},
                 style_function=lambda x: {
                     'opacity': 0.2,
                     'fillColor': x['properties']['fill'],
                 }).add_to(mf)

mf

from IPython.display import Image

Image('sf_ref44_static.png')

