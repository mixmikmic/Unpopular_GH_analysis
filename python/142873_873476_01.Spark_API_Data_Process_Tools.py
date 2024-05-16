import pyspark
sc=pyspark.SparkContext()

# parallelize creates an RDD from the passed object
x = sc.parallelize([1,2,3])
y = x.map(lambda x: (x,x**2))

# collect copies RDD elements to a list on the driver
print(x.collect())
print(y.collect())

x = sc.parallelize([1,2,3])
y = x.flatMap(lambda x: (x,100*x,x**2))
print(x.collect())
print(y.collect())

x = sc.parallelize([1,2,3],2)
def f(iterator): yield sum(iterator)
y = x.mapPartitions(f)
# glom() falttens elements on the same partition
print(x.glom().collect())
print(y.glom().collect())

x = sc.parallelize([1,2,3],2)
def f(partitionIndex,iterator): yield (partitionIndex, sum(iterator))
y = x.mapPartitionsWithIndex(f)
print(x.glom().collect())
print(y.glom().collect())

y = x.getNumPartitions()
print(x.glom().collect())
print(y)

x = sc.parallelize([1,2,3])
y = x.filter(lambda x:x%2==1) # filter out elements
print(x.collect())
print(y.collect())

x = sc.parallelize(['A','A','B'])
y = x.distinct()
print(x.collect())
print(y.collect())

x = sc.parallelize(range(7))
ylist = [x.sample(withReplacement=False,fraction=0.5) for i in range(5)]
print('x = '+str(x.collect()))
for cnt, y in zip(range(len(ylist)),ylist):
    print('sample: '+str(cnt)+' y = '+str(y.collect()))

x = sc.parallelize(range(7))
ylist = [x.takeSample(withReplacement=False,num=3) for i in range(5)]
print('x = '+str(x.collect()))
for cnt, y in zip(range(len(ylist)),ylist):
    print('sample: '+str(cnt)+' y = '+str(y))

x = sc.parallelize(['A','A','B'])
y = sc.parallelize(['D','C','A'])
z = x.union(y)
print(x.collect())
print(y.collect())
print(z.collect())

x = sc.parallelize(['A','A','B'])
y = sc.parallelize(['A','C','D'])
z = x.intersection(y)
print(x.collect())
print(y.collect())
print(z.collect())

x = sc.parallelize([('B',1),('A',2),('C',3)])
y = x.sortByKey()
print(x.collect())
print(y.collect())



