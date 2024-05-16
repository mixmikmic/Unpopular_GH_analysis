import numpy as np

# CACO 30 March 2016 results
    
# OPUS solution results, now based on the "precise" results
dzO = 0.015
dxdyO = np.sqrt( 0.002**2 + 0.005**2)

# Uncertainty in survey results - this is the RMS differences between reference markers and stake measurements
# for 5, 5, and 6 stakes on RM1, RM2, and RM3.
dzS = 0.007
dxdyS = np.sqrt(0.0144**2+0.0223**2)

dxdySO=np.sqrt(dxdyO**2+dxdyS**2)
dzSO=np.sqrt(dzO**2+dzS**2)

# Marker error from Photoscan
dzP = 0.0086
dxdyP = np.sqrt(0.0141**2+0.0122**2)

# Some kind of error estimate from Photoscan
# Pixel size on ground: 7.19 cm/pix in DEM, 5 cm/pix in ortho
# Reprojection error is 0.3 pix...assuming pix = 4 cm, then
rp_error = 0.3 # pix
px_size = 4.    # cm/pix
rp_error_m = rp_error*px_size/100.

print "OPUS solution = ",dxdyO, dzO
print "stake errors = ",dxdyS, dzS
print "combined survey errors = ",dxdySO,dzSO
print "marker error in Photoscan = ",dxdyP,dzP
print "sum of GCP undertainty = ",np.sqrt(dxdySO**2+dxdyP**2), np.sqrt(dzSO**2+dzP**2) 
print "rp_error_m = ",rp_error_m
print "sum of these = ",np.sqrt(dxdySO**2+dxdyP**2+rp_error_m**2), np.sqrt(dzSO**2+dzP**2+rp_error_m**2) 



