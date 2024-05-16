import pandas as pd

path = '/Users/smmaurer/Desktop/MTC BAUS data/2015_09_01_bayarea_v3.h5'
with pd.HDFStore(path) as hdf:
    print hdf.keys()

parcels = pd.read_hdf(path, 'parcels')

parcels.index.name

parcels.count()

parcels.geom_id.nunique()



path = '/Users/smmaurer/Desktop/MTC BAUS data/02_01_2016_parcels_geography.csv'
geodf = pd.read_csv(path, index_col="geom_id", dtype={'jurisdiction': 'str'})

geodf.index.name

geodf.count()



path = '/Users/smmaurer/Dropbox/Git-rMBP/ual/bayarea_urbansim/data/census_id_to_name.csv'
namedf = pd.read_csv(path)

namedf['jurisdiction_id'] = namedf.census_id

namedf.index.name

namedf.count()



parcels['geom_id'].reset_index().describe()

merged = pd.merge(parcels['geom_id'].reset_index(), 
                  geodf['jurisdiction_id'].reset_index(), 
                  how='left', on='geom_id')

merged = pd.merge(merged, namedf[['jurisdiction_id', 'name10']], 
                  how='left', on='jurisdiction_id').set_index('parcel_id')

print merged.head()

merged.count()

merged.geom_id.nunique()

merged.describe()

merged.to_csv('parcel_jurisdictions_v1.csv')









