import numpy as np 
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles

# List of provided metrics
metrics.BaseMetric.list(doc=False)

from lsst.sims.maf.metrics import BaseMetric

class Coaddm5Metric(BaseMetric):
    """Calculate the coadded m5 value at this gridpoint."""
    def __init__(self, m5Col = 'fiveSigmaDepth', metricName='CoaddM5', **kwargs):
        """Instantiate metric.
        m5col = the column name of the individual visit m5 data."""
        self.m5col = m5col
        super(Coaddm5Metric, self).__init__(col=m5Col, metricName=metricName, **kwargs)
    def run(self, dataSlice, slicePoint=None):
        return 1.25 * np.log10(np.sum(10.**(.8*dataSlice[self.m5col])))

# Import BaseMetric, or have it available to inherit from
from lsst.sims.maf.metrics import BaseMetric

# Define our class, inheriting from BaseMetric
class OurPercentileMetric(BaseMetric):
    # Add a doc string to describe the metric.
    """
    Calculate the percentile value of a data column
    """
    # Add our "__init__" method to instantiate the class.
    # We will make the 'percentile' value an additional value to be set by the user.
    # **kwargs allows additional values to be passed to the BaseMetric that you 
    #     may not have been using here and don't want to bother with. 
    def __init__(self, colname, percentile, **kwargs):
        # Set the values we want to keep for our class.
        self.colname = colname
        self.percentile = percentile
        # Now we have to call the BaseMetric's __init__ method, to get the "framework" part set up.
        # We currently do this using 'super', which just calls BaseMetric's method.
        # The call to super just basically looks like this .. you must pass the columns you need, and the kwargs.
        super(OurPercentileMetric, self).__init__(col=colname, **kwargs)
        
    # Now write out "run" method, the part that does the metric calculation.
    def run(self, dataSlice, slicePoint=None):
        # for this calculation, I'll just call numpy's percentile function.
        result = np.percentile(dataSlice[self.colname], self.percentile)
        return result

metric = OurPercentileMetric('airmass', 20)
slicer = slicers.HealpixSlicer(nside=64)
sqlconstraint = 'filter = "r" and night<365'
myBundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint)

opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
bgroup = metricBundles.MetricBundleGroup({0: myBundle}, opsdb, outDir='newmetric_test', resultsDb=None)
bgroup.runAll()

myBundle.setPlotDict({'colorMin':1.0, 'colorMax':1.8})
bgroup.plotAll(closefigs=False)



