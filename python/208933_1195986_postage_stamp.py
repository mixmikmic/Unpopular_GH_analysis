import numpy as np

import lsst.daf.persistence as dafPersist
import lsst.afw.geom as afwGeom
import lsst.afw.coord as afwCoord
import lsst.afw.image as afwImage

from astropy.visualization import ZScaleInterval

# Set plotting defaults
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 8)
zscale = ZScaleInterval()

butler = dafPersist.Butler('/home/shared/twinkles/output_data_v2')

# Display the available keys
print(butler.getKeys('calexp'))
#print(dir(butler))
#butler.queryMetadata('calexp', butler.getKeys('calexp')) # Warning may not return in same order as getKeys()

# Count the number of images in each filter
# Is queryMetadata faster than subset?
visit_array, band_array = map(np.array, zip(*butler.queryMetadata('calexp', ['visit', 'filter'])))
for band in np.unique(band_array):
    print(band, np.sum(band_array == band))

subset = butler.subset('src')
dataid = subset.cache[4] # A random choice of image
#print(subset.cache[4])
#print(dir(butler))
#help(butler.get)
#my_src = butler.get('src', dataId={'visit': 234})
my_src = butler.get('src', dataId=dataid)
my_calexp = butler.get('calexp', dataId=dataid)
my_wcs = my_calexp.getWcs()
my_calib = my_calexp.getCalib()
my_calib.setThrowOnNegativeFlux(False) # For magnitudes

#my_src.schema.getNames()
#my_src.schema # To see slots

# This is just a demonstration of the slot functionality
np.testing.assert_equal(my_calib.getMagnitude(my_src['base_PsfFlux_flux']),
                        my_calib.getMagnitude(my_src.getPsfFlux()))

psf_mag = my_calib.getMagnitude(my_src.getPsfFlux())
#psf_mag = np.where(np.isfinite(psf_mag), psf_mag, 9999.) # Don't set sentinel values!

cm_mag = my_calib.getMagnitude(my_src.getModelFlux())
cm_mag = np.where(np.isfinite(cm_mag), cm_mag, 9999.) # Don't set sentinel values

# If you have nan or inf values in your array, use the range argument to avoid searching for min and max
plt.figure()
#plt.yscale('log', nonposy='clip')
plt.hist(psf_mag, bins=np.arange(15., 26., 0.25), range=(15., 26.))
#plt.hist(np.nan_to_num(psf_mag), bins=np.arange(15., 26., 0.25)) # Alertnative
plt.xlabel('PSF Magnitude')
plt.ylabel('Counts')

plt.figure()
plt.scatter(psf_mag, psf_mag - cm_mag, c=my_src['base_ClassificationExtendedness_value'], cmap='coolwarm')
plt.colorbar().set_label('Classification')
plt.xlim(15., 26.)
plt.ylim(-1., 2.)
plt.xlabel('PSF Magnitude')
plt.ylabel('PSF - Model Magnitude')

# Pick a bright star candidates
#mask = ~np.isfinite(psf_mag) | (my_src['base_ClassificationExtendedness_value'] == 1)
#index = np.argmin(np.ma.masked_array(psf_mag, mask))

# Pick a bright star that was using to calibrate photometry
selection = my_src['calib_photometry_used']
index = np.argmin(np.ma.masked_array(psf_mag, ~selection))

print(psf_mag[index])
print(index)

ra_target, dec_target = my_src['coord_ra'][index], my_src['coord_dec'][index] # Radians
#print(dir(afwCoord))
#print(dir(afwGeom))
#print(help(afwGeom.Point2D))
#coord = afwCoord.Coord(ra_target * afwGeom.degrees, dec_target * afwGeom.degrees)
#coord = afwGeom.Point2D(ra_target * afwGeom.degrees, dec_target * afwGeom.degrees)
radec = afwGeom.SpherePoint(ra_target, dec_target, afwGeom.radians) # Is this really the preferred way to do this?

#xy = afwGeom.PointI(my_wcs.skyToPixel(radec)) # This converts to integer
#xy = afwGeom.Point2D(my_wcs.skyToPixel(radec))
xy = my_wcs.skyToPixel(radec)
print(my_src['base_SdssCentroid_x'][index], my_src['base_SdssCentroid_y'][index])
print(xy.getX(), xy.getY())
#print(xy)
#dir(my_wcs)
#xy = my_wcs.skyToPixel(radec)

print(my_wcs.skyToPixel(radec).getX())
print(my_src.getX()[index])

# Equivalence check
assert my_src.getX()[index] == my_src['base_SdssCentroid_x'][index]

# Trying to isolate some behavior here that I don't understand
ra_target, dec_target = my_src['coord_ra'][index], my_src['coord_dec'][index] # Radians
radec = afwGeom.SpherePoint(ra_target, dec_target, afwGeom.radians)
xy = my_wcs.skyToPixel(radec)
print(my_wcs.skyToPixel(radec).getX())
print(my_src.getX()[index])

# Probably this cell should go away
#my_calexp = butler.get('calexp', dataId={'visit': 234})
#subset = butler.subset('md')
#subset = butler.subset('wcs')
#subset.cache
#my_wcs = butler.get('wcs', dataId={'visit': 234})

cutoutSize = afwGeom.ExtentI(100, 100)
#my_wcs.skyToPixel(coord)
xy = afwGeom.PointI(my_wcs.skyToPixel(radec))
bbox = afwGeom.BoxI(xy - cutoutSize//2, cutoutSize)
#print(bbox)
#print(dir(my_calexp))
#print(help(butler.get))
#my_calexp.getBBox()

# Full frame image
image = butler.get('calexp', immediate=True, dataId=dataid) #.getMaskedImage()

# Postage stamp image only
cutout_image = butler.get('calexp_sub', bbox=bbox, immediate=True, dataId=dataid).getMaskedImage()

#import lsst.afw.display as afwDisplay
#print(dir(afwDisplay))

vmin, vmax = zscale.get_limits(image.image.array)
plt.imshow(image.image.array, vmin=vmin, vmax=vmax, cmap='binary')
plt.colorbar()
#dir(xy)
plt.scatter(xy.getX(), xy.getY(), color='none', edgecolor='red', s=200)
#my_calexp.image.array

# Demonstration of equivalency
my_calexp_cutout = my_calexp.Factory(my_calexp, bbox, afwImage.LOCAL)
assert np.all(my_calexp_cutout.image.array == cutout_image.image.array)

print(cutout_image.getDimensions())
vmin, vmax = zscale.get_limits(cutout_image.image.array)
plt.imshow(cutout_image.image.array, vmin=vmin, vmax=vmax, cmap='binary')

# Does the cutout_image have a wcs? It does not appear to...
plt.scatter(xy.getX() - cutout_image.getX0(), xy.getY() - cutout_image.getY0(), c='none', edgecolor='red', s=200)



