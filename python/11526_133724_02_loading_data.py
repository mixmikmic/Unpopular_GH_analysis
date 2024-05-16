import datacube
dc = datacube.Datacube(app='load-data-example')

data = dc.load(product='ls5_nbar_albers', x=(149.25, 149.5), y=(-36.25, -36.5),
               time=('2008-01-01', '2009-01-01'))

data

data = dc.load(product='ls5_nbar_albers', x=(1543137.5, 1569137.5), y=(-4065537.5, -4096037.5),
               time=('2008-01-01', '2009-01-01'), crs='EPSG:3577')

data

data = dc.load(product='ls5_nbar_albers', x=(149.25, 149.5), y=(-36.25, -36.5),
               time=('2008-01-01', '2009-01-01'), measurements=['red', 'nir'])

data

help(dc.load)



