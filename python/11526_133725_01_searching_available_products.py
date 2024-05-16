import datacube
dc = datacube.Datacube(app='list-available-products-example')

dc.list_products()

data = dc.load(product='bom_rainfall_grids', x=(149.0, 150.0), y=(-36.0, -37.0),
               time=('2000-01-01', '2001-01-01'))

data



