import pyspark
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD
from pyspark.mllib.tree import DecisionTree

sc = pyspark.SparkContext()

# Check that Spark is working
largeRange = sc.parallelize(range(0,10000,2),5)
reduceTest = largeRange.reduce(lambda a,b: a+b)
filterReduceTest = largeRange.filter(lambda x:x%7 ==0).sum()
print('largeRange:',largeRange)
print('reduceTest:',reduceTest)
print('filterRduceTest:',filterReduceTest)

# check loading data with sc.textFile
import os.path
baseDir = os.path.join('data\MNIST')
fileName = baseDir + '\Train-28x28_cntk_text.txt'

rawData = sc.textFile(fileName)
TrainNumber = rawData.count()
print(TrainNumber)

assert TrainNumber == 60000

# Test Compare with hash
# Check our testing library/package
# This should print '1 test passed.' on two lines
from test_helper import Test
twelve = 12
Test.assertEquals(twelve, 12, 'twelve should equal 12')
#Test.assertEqualsHashed(twelve,'7b52009b64fd0a2a49e6d8a939753077792b0554','twelve, once hashed, should equal the hashed value of 12' )

# Test Compare lists 
# This should print '1 test paseed.'
unsortedList = [(5,'b'),(5,'a'),(4,'c'),(3,'a')]
Test.assertEquals(sorted(unsortedList),[(3,'a'),(4,'c'),(5,'a'),(5,'b')],
                 "unsortedList doesn't sort properly")

# Check matplotlib plotting
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import log

# function for generating plot layout
def preparePlot(xticks, yticks, figsize=(10.5,6), hideLabels=False, gridColor='#999999', gridWidth=1.0):
    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.axes.tick_params(labelcolor='#999999',labelsize='10')
    for axis, ticks in [(ax.get_xaxis(), xticks),(ax.get_yaxis(),yticks)]:
        axis.set_ticks_position('none')
        axis.set_ticks(ticks)
        axis.label.set_color('#999999')
        if hideLabels: axis.set_ticklabels([])
    plt.grid(color=gridColor, lineWidth=gridWidth, linestyle='-')
    map(lambda position: ax.spines[position].set_visible(False),['bottom','top','left','right'])
    return fig, ax

# generate layout and plot data
x = range(1,50)
y = [log(x1 ** 2) for x1 in x]
fig, ax = preparePlot(range(5,60,10),range(0,12,1))
plt.scatter(x,y,s=14**2, c='#d6ebf2', edgecolors='#8cbfd0',alpha=0.75)
ax.set_xlabel(r'$range(1, 50)$'), ax.set_ylabel(r'$\log_e(x^2)$')



