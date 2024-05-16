from __future__ import print_function
# Need to import everything before getting help!
import lsst.sims.maf
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.stackers as stackers
import lsst.sims.maf.plots as plots

# Show the list of metrics
metrics.BaseMetric.help(doc=False)

# Show the help of a given metric
help(metrics.TransientMetric)

# If you have an object, help works on it too!
metric = metrics.CountMetric('expMJD')
help(metric)

# Show the list of slicers
slicers.BaseSlicer.help(doc=False)

# Show help of a given slicer
help(slicers.HealpixSlicer)

stackers.BaseStacker.help(doc=False)

# Show help of a given stacker
help(stackers.RandomDitherFieldPerNightStacker)

# See the plots available.
import inspect
vals = inspect.getmembers(plots, inspect.isclass)
for v in vals:
    print(v[0])

# Show the help of a given plots class
help(plots.HealpixSkyMap)



