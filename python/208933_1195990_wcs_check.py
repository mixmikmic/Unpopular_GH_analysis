import numpy as np

import lsst.daf.persistence as dafPersist
import lsst.afw.geom as afwGeom

butler = dafPersist.Butler('/home/shared/twinkles/output_data_v2')

subset = butler.subset('src')
dataid = subset.cache[4] # Random choice
my_src = butler.get('src', dataId=dataid)
my_calexp = butler.get('calexp', dataId=dataid)
my_wcs = my_calexp.getWcs()

# Pick a bright star that was using to calibrate photometry
selection = my_src['calib_photometry_used']
index = np.argmax(np.ma.masked_array(my_src.getPsfFlux(), ~selection))

ra_target, dec_target = my_src['coord_ra'][index], my_src['coord_dec'][index] # Radians
radec = afwGeom.SpherePoint(ra_target, dec_target, afwGeom.radians)

xy = my_wcs.skyToPixel(radec)

print('%10s%15.6f%15.6f'%('sdss:', 
                          my_src['base_SdssCentroid_x'][index], 
                          my_src['base_SdssCentroid_y'][index]))
print('%10s%15.6f%15.6f'%('naive:', 
                          my_src['base_NaiveCentroid_x'][index], 
                          my_src['base_NaiveCentroid_y'][index])) 
print('%10s%15.6f%15.6f'%('gauss:', 
                          my_src['base_GaussianCentroid_x'][index], 
                          my_src['base_GaussianCentroid_y'][index])) 
print('%10s%15.6f%15.6f'%('radec:', 
                          xy.getX(), 
                          xy.getY()))

