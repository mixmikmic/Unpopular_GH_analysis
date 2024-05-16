import mpl_toolkits.basemap.pyproj as pyproj # Import the pyproj module
# Define a projection with Proj4 notation, in this case an Icelandic grid
isn2004=pyproj.Proj("+proj=lcc +lat_1=64.25 +lat_2=65.75 +lat_0=65 +lon_0=-19 +x_0=1700000 +y_0=300000 +no_defs +a=6378137 +rf=298.257222101 +to_meter=1")
 # Define some common projections using EPSG codes
wgs84=pyproj.Proj("+init=EPSG:4326") # LatLon with WGS84 datum used by GPS units and Google Earth
wgs84b = pyproj.Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
nad83=pyproj.Proj("+init=EPSG:4269")
nad83b=pyproj.Proj("+proj=lcc +lat_1=42.68333333333333 +lat_2=41.71666666666667 +lat_0=41 +lon_0=-71.5 +x_0=200000 +y_0=750000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs +preserve_units")
nad83c=pyproj.Proj("+proj=lcc +lat_1=42.68333333333333 +lat_2=41.71666666666667 +lat_0=41 +lon_0=-71.5 +x_0=200000 +y_0=750000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +to_meter=0 +no_defs +preserve_units")
pyproj.transform(wgs84,nad83c,-70.483290627,41.766497572)

