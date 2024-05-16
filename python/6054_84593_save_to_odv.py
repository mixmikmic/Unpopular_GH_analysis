from biofloat import ArgoData, converters
from os.path import join, expanduser
ad = ArgoData(cache_file=join(expanduser('~'),'6881StnP_5903891.hdf'), verbosity=2)

wmo_list = ad.get_cache_file_all_wmo_list()
df = ad.get_float_dataframe(wmo_list)

df.head()

corr_df = df.dropna().copy()
corr_df['DOXY_ADJUSTED'] *= 1.12
corr_df.head()

converters.to_odv(corr_df, '6881StnP_5903891.txt')

from IPython.display import Image
Image('../doc/screenshots/Screen_Shot_2015-11-25_at_1.42.00_PM.png')

