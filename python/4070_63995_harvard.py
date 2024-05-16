get_ipython().magic('load harvard.py')

get_ipython().system('wget {RSS_URL} -O events.rss')

rss = feedparser.parse('./events.rss')
rss.feed.title

events = parse_rss(rss)

HTML(events[1]['description'])

import json
with open('harvard_university.json', 'w') as outfile:
  json.dump(events, outfile)

