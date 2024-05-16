#import general things
import numpy as np 
from scipy.stats import chi2
from scipy.interpolate import UnivariateSpline, interp1d
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

#import lsst things
import lsst.sims.maf.db as db
import lsst.sims.maf.metrics as metrics
import lsst.sims.maf.slicers as slicers
import lsst.sims.maf.metricBundles as metricBundles
from lsst.sims.maf.metrics import BaseMetric

# 5 sigma depth
m5 = np.random.rand(200)*2.+23. #200 observations with 5sigma depths uniformily distributed between mag 23 and 25
N = len(m5) # number of visits
N

mref = 18.

sigma = 0.2 * (10.**(-0.2*m5)) * (10.**(0.2*mref)) # Flux standard error for star of magnitude mref
print np.mean(sigma)

signal = 0.01 # 1% intrinsic standard deviation

#Let's try directly simulating a bunch of standard deviations, both with and without signal.
numruns = 10000 # simulate numruns realizations of these observations

noiseonlyvar = np.zeros(numruns) # hold the measured noise-only variances 
noiseandsignalvar = np.zeros(numruns) # hold the measured noise-plus-signal variances 
signalonlyvar = np.zeros(numruns) #temporary for testing

#Simulate the measured variances
for i in np.arange(numruns):
    scatter = np.random.randn(N)*sigma # random realization of the Gaussian error distributions
    sig = np.random.randn(N)*signal #random realization of the underlying variable signal
    noiseonlyvar[i] = np.var(scatter) # store the noise-only variance
    signalonlyvar[i] = np.var(sig) # store the signal-only variance
    noiseandsignalvar[i] = np.var(sig+scatter) # store the noise-plus-signal variance
    
#plot the histograms of measured variances:
plt.hist(noiseonlyvar,bins=50,label='noise only',color='blue')
plt.hist(noiseandsignalvar,bins=50,label='signal plus noise',color='green')
plt.hist(signalonlyvar,bins=50,label='signal only',color='black')
plt.xlabel('measured variance')
plt.ylabel('# instances')
plt.xlim(0.00,0.0005)
plt.legend()
plt.show()

#Plot the cumulative density functions
plt.plot(np.sort(noiseonlyvar),np.arange(numruns)/float(numruns),label='noise only',color='blue') # noise only
plt.plot(np.sort(signalonlyvar),np.arange(numruns)/float(numruns),label='signal only',color='black') # signal only
plt.plot(np.sort(noiseandsignalvar),1.-np.arange(numruns)/float(numruns),label='1-(signal+noise)',color='green') # 1-(signal+noise)
plt.xlabel('measured variance')
plt.ylabel('cumulative density')
plt.xlim(0.00,0.0005)
plt.legend()
plt.show()

completeness = .95 # fraction of variable sources recovered
contamination = .05 # fraction of non-variable sources that are misclassified

#%%timeit #This is the computationally expensive part, but hopefully not too bad.
#1 loops, best of 3: <300 ms per loop

mag = np.arange(16,np.mean(m5),0.5) #magnitudes to be sampled
res = np.zeros(mag.shape) #hold the distance between the completeness and contamination goals.

noiseonlyvar = np.zeros(numruns) # hold the measured noise-only variances 

#Calculate the variance at a reference magnitude and scale from that
m0=20.
sigmaref = 0.2 * (10.**(-0.2*m5)) * (10.**(0.2*m0))

#run the simulations
#Simulate the measured noise-only variances at a reference magnitude
for i in np.arange(numruns):
    scatter = np.random.randn(N)*sigmaref # random realization of the Gaussian error distributions
    noiseonlyvar[i] = np.var(scatter) # store the noise-only variance

#Since we are treating the underlying signal being representable by a fixed-width gaussian,
#its variance pdf is a Chi-squared distribution with the degrees of freedom = visits.
#Since variances add, the variance pdfs convolve.

#We'll use the cdf of the noise-only variances because it's easier to interpolate
noisesorted = np.sort(noiseonlyvar)
interpnoisecdf = UnivariateSpline(noisesorted,np.arange(numruns)/float(numruns),k=1,s=0) #linear

#We need a binned, signal-only variance probability distribution function for numerical convolution
numsignalsamples = 50
xsig = np.linspace(chi2.ppf(0.001, N),chi2.ppf(0.999, N),numsignalsamples)
signalpdf = chi2.pdf(xsig, N)
#correct x to the proper variance scale
xsig = (signal**2.)*xsig/N
pdfstepsize = xsig[1]-xsig[0]
#Since everything is going to use this stepsize down the line,
#normalize so the pdf integrates to 1 when summed (no factor of stepsize needed)
signalpdf /= np.sum(signalpdf)

#run through the sample magnitudes, calculate distance between cont and comp thresholds
for i,mref in enumerate(mag): #i counts and mref is the currently sampled magnitude
    #Scale factor from m0
    scalefact = 10.**(0.4*(mref-m0))
    
    #Calculate the desired contamination threshold
    contthresh = np.percentile(noiseonlyvar,100.-100.*contamination)*scalefact
    
    #Realize the noise CDF at the required stepsize
    xnoise = np.arange(noisesorted[0]*scalefact,noisesorted[-1]*scalefact,pdfstepsize)
    noisecdf = interpnoisecdf(xnoise/scalefact)
    noisepdf = (noisecdf[1:]-noisecdf[:-1]) #turn into a noise pdf
    noisepdf /= np.sum(noisepdf)
    xnoise = (xnoise[1:]+xnoise[:-1])/2. #from cdf to pdf conversion
    
    #calculate and plot the convolution = signal+noise variance dist.
    convolution=0
    if len(noisepdf) > len(signalpdf):
        convolution = np.convolve(noisepdf,signalpdf)
    else: 
        convolution = np.convolve(signalpdf,noisepdf)
    xconvolved = xsig[0]+xnoise[0]+np.arange(len(convolution))*pdfstepsize
    
    #calculate the completeness threshold
    combinedcdf = np.cumsum(convolution)
    findcompthresh = interp1d(combinedcdf,xconvolved)
    compthresh = findcompthresh(1.-completeness)
    
    #Plot the pdfs for demonstration purposes
    plt.plot(xsig, signalpdf, label="signal",c='b')
    plt.plot(xnoise,noisepdf,  label="noise",c='r')
    plt.plot([contthresh,contthresh],[0,noisepdf[np.argmin(np.abs(xnoise-contthresh))]],
             'r--',label='cont thresh')
    plt.plot(xconvolved,convolution, 
             'g-',label="sig+noise")
    plt.plot([compthresh,compthresh],[0,convolution[np.argmin(np.abs(xconvolved-compthresh))]],
             'g--',label='comp thresh')
    plt.xlabel('variance')
    plt.ylabel('pdf')
    plt.title('mag = '+str(mref))
    plt.legend()
    plt.show()

    res[i] = compthresh - contthresh

#Plot the results:
plt.scatter(mag,res)
#Interpolate with a cubic spline
f1 = UnivariateSpline(mag,res,k=1,s=0)
#Find the closest approach to zero to the desired resolution
magres = 0.01 #magnitude resolution
magsamples = np.arange(16,np.mean(m5),magres) #sample the magnitude range at this resolution
plt.plot(magsamples,f1(magsamples)) #Plot the interpolated values
plt.xlabel('magnitude')
plt.ylabel('completeness - contamination')
plt.ylim(-0.0005,0.0001)

#Find the closest approach to zero
vardepth = magsamples[np.argmin(np.abs(f1(magsamples)))]
plt.plot([16,np.mean(m5)],[0,0],'--',color='red') #plot zero
#that's the final result
print vardepth
plt.plot([vardepth,vardepth],[-1,1],'--',color='red') #Plot resulting variability depth

plt.show()

#Calculate the survey depth there a variable star can be reliably identified through a comparsion
#of the measured variance to the measurement uncertainty.

class VarDepth(BaseMetric):
    def __init__(self, m5Col = 'fiveSigmaDepth', 
                 metricName='variability depth', 
                 completeness = .95, contamination = .05, 
                 numruns = 10000, signal = 0.01,
                 magres = 0.01, **kwargs):
        """
        Instantiate metric.
        
        :m5col: the column name of the individual visit m5 data.
        :completeness: fractional desired completeness of recovered variable sample.
        :contamination: fractional allowed incompleteness of recovered nonvariables.
        :numruns: number of simulated realizations of noise (most computationally espensive part).
        :signal: sqrt total pulsational power meant to be recovered.
        :magres: desired resolution of variability depth result."""
        self.m5col = m5Col
        self.completeness = completeness
        self.contamination = contamination
        self.numruns = numruns 
        self.signal = signal
        self.magres = magres
        super(VarDepth, self).__init__(col=m5Col, metricName=metricName, **kwargs)
    def run(self, dataSlice, slicePoint=None):
        #Get the visit information
        m5 = dataSlice[self.m5col]
        #Number of visits
        N = len(m5)
        
        #magnitudes to be sampled
        mag = np.arange(16,np.mean(m5),0.5) 
        #hold the distance between the completeness and contamination goals.
        res = np.zeros(mag.shape) 
        #make them nans for now
        res[:] = np.nan 

        #hold the measured noise-only variances 
        noiseonlyvar = np.zeros(self.numruns)

        #Calculate the variance at a reference magnitude and scale from that
        m0=20.
        sigmaref = 0.2 * (10.**(-0.2*m5)) * (10.**(0.2*m0))

        #run the simulations
        #Simulate the measured noise-only variances at a reference magnitude
        for i in np.arange(self.numruns):
            # random realization of the Gaussian error distributions
            scatter = np.random.randn(N)*sigmaref 
            noiseonlyvar[i] = np.var(scatter) # store the noise-only variance
            
        #Since we are treating the underlying signal being representable by a 
        #fixed-width gaussian, its variance pdf is a Chi-squared distribution 
        #with the degrees of freedom = visits. Since variances add, the variance 
        #pdfs convolve. The cumulative distribution function of the sum of two 
        #random deviates is the convolution of one pdf with a cdf. 

        #We'll consider the cdf of the noise-only variances because it's easier 
        #to interpolate
        noisesorted = np.sort(noiseonlyvar)
        #linear interpolation
        interpnoisecdf = UnivariateSpline(noisesorted,np.arange(self.numruns)/float(self.numruns),k=1,s=0)

        #We need a binned, signal-only variance probability distribution function for numerical convolution
        numsignalsamples = 100
        xsig = np.linspace(chi2.ppf(0.001, N),chi2.ppf(0.999, N),numsignalsamples)
        signalpdf = chi2.pdf(xsig, N)
        #correct x to the proper variance scale
        xsig = (self.signal**2.)*xsig/N
        pdfstepsize = xsig[1]-xsig[0]
        #Since everything is going to use this stepsize down the line,
        #normalize so the pdf integrates to 1 when summed (no factor of stepsize needed)
        signalpdf /= np.sum(signalpdf)

        #run through the sample magnitudes, calculate distance between cont 
        #and comp thresholds.
        #run until solution found.
        solutionfound=False
        
        for i,mref in enumerate(mag): 
            #i counts and mref is the currently sampled magnitude
            #Scale factor from m0
            scalefact = 10.**(0.4*(mref-m0))

            #Calculate the desired contamination threshold
            contthresh = np.percentile(noiseonlyvar,100.-100.*self.contamination)*scalefact

            #Realize the noise CDF at the required stepsize
            xnoise = np.arange(noisesorted[0]*scalefact,noisesorted[-1]*scalefact,pdfstepsize)
            
            #Only do calculation if near the solution:
            if (len(xnoise) > numsignalsamples/10) and (not solutionfound):
                noisecdf = interpnoisecdf(xnoise/scalefact)
                noisepdf = (noisecdf[1:]-noisecdf[:-1]) #turn into a noise pdf
                noisepdf /= np.sum(noisepdf)
                xnoise = (xnoise[1:]+xnoise[:-1])/2. #from cdf to pdf conversion

                #calculate and plot the convolution = signal+noise variance dist.
                convolution=0
                if len(noisepdf) > len(signalpdf):
                    convolution = np.convolve(noisepdf,signalpdf)
                else: 
                    convolution = np.convolve(signalpdf,noisepdf)
                xconvolved = xsig[0]+xnoise[0]+np.arange(len(convolution))*pdfstepsize

                #calculate the completeness threshold
                combinedcdf = np.cumsum(convolution)
                findcompthresh = UnivariateSpline(combinedcdf,xconvolved,k=1,s=0)
                compthresh = findcompthresh(1.-self.completeness)

                res[i] = compthresh - contthresh
                if res[i] < 0: solutionfound = True
        
        #interpolate for where the thresholds coincide
        #print res
        if np.sum(np.isfinite(res)) > 1:
            f1 = UnivariateSpline(mag[np.isfinite(res)],res[np.isfinite(res)],k=1,s=0)
            #sample the magnitude range at given resolution
            magsamples = np.arange(16,np.mean(m5),self.magres) 
            vardepth = magsamples[np.argmin(np.abs(f1(magsamples)))]
            return vardepth
        else:
            return min(mag)-1

#And test it out:
metric = VarDepth('fiveSigmaDepth',numruns=100) #Note: default numruns=10000 takes way too long.
slicer = slicers.HealpixSlicer(nside=64)
sqlconstraint = 'filter = "r"'
myBundle = metricBundles.MetricBundle(metric, slicer, sqlconstraint)

#Run it:
opsdb = db.OpsimDatabase('enigma_1189_sqlite.db')
bgroup = metricBundles.MetricBundleGroup({0: myBundle}, opsdb, outDir='newmetric_test', resultsDb=None)
bgroup.runAll()

myBundle.setPlotDict({'colorMin':16.1, 'colorMax':20.5})
bgroup.plotAll(closefigs=False,dpi=600,figformat='png')



