from lsst.daf.persistence import Butler

butler = Butler('/home/shared/twinkles/output_data_v2')

subset = butler.subset('calexp', filter='i')

dataId = subset.cache[6]

my_calexp = butler.get('calexp', **dataId)

psf = my_calexp.getPsf()

shape = psf.computeShape()

shape.getDeterminantRadius()

from lsst.afw.geom import Point2D
point = Point2D(50.1, 160.2)

shape = psf.computeShape(point)

shape.getDeterminantRadius()

shapes = []
for x in range(100):
    for y in range(100):
        point = Point2D(x*40., y*40.)
        shapes.append(psf.computeShape(point).getDeterminantRadius())

from matplotlib import pylab as plt

plt.hist(shapes)

src = butler.get('src', **dataId)

import numpy
star_idx = numpy.where(src.get('base_ClassificationExtendedness_value') == 0)

import math
star_shapes = []
psf_shapes = []
for x, y in zip(src.getX()[star_idx[0]], src.getY()[star_idx[0]]):
    point = Point2D(x, y)
    psf_shapes.append(psf.computeShape(point).getDeterminantRadius())
for xx, yy in zip(src.get('base_SdssShape_psf_xx')[star_idx[0]], src.get('base_SdssShape_psf_yy')[star_idx[0]]):
    star_shapes.append(math.sqrt(xx + yy))
star_shapes = numpy.array(star_shapes)
psf_shapes = numpy.array(psf_shapes)

good_star_idx = numpy.where(numpy.isfinite(star_shapes))[0]

plt.hist(psf_shapes[good_star_idx])

plt.hist(star_shapes[good_star_idx])

plt.scatter(star_shapes[good_star_idx], star_shapes[good_star_idx]/psf_shapes[good_star_idx]*(math.sqrt(2)/2), alpha=0.5)
plt.ylim(1.00075, 1.00175)



