lines = sc.textFile('text_file.md')
print lines.first()
print lines.count()

# wordcount in a single line
wdct = lines.flatMap(lambda line: line.split()).countByValue()
print wdct.items()[:10]

num_chars = lines.map(lambda line: len(line))
first_word = lines.filter(lambda line: len(line.split()) > 2).map(lambda line: line.lower().split()[0])

# make a pair RDD
pairs_num = num_chars.map(lambda x: (x, x**2))
pairs_wds = first_word.map(lambda word: (word, 1))
print pairs_num.take(5)
print pairs_wds.take(5)

# single-line word count (the lambda function says what to do with the values)
# the value type must the same as original type
wc = pairs_wds.reduceByKey(lambda x, y: x + y)
print wc.filter(lambda p: p[1] > 1).collect()

# group by key then convert the pyspark.resultiterable.ResultIterable to a Python list using mapValues
print pairs_num.groupByKey().mapValues(list).take(10)

# mapValue will apply a function to each value without altering the key
# the partition of the return RDD (this is a transformation, not action)
# will be the same of the original partition
pairs_num.mapValues(lambda x: -x).take(5)

pairs_num.flatMapValues(lambda x: range(x)).take(5)

# revisit map and flatmap
print 'map', num_chars.map(lambda x: x / 2).take(4)
print 'map', num_chars.map(lambda x: (x, x)).take(4)
print 'flatmap', num_chars.flatMap(lambda x: (x, x)).take(4)

wc.keys().take(10)

# values
wc.values().take(10)

# here we create a new collection of pairs using existing data
repeat = sc.parallelize([(w, c) for w, c, in zip(wc.keys().collect(), wc.values().collect())])
print repeat.count()
print repeat.first()

wc.sortByKey().take(10)

wc.sortByKey(ascending=False, keyfunc=lambda x: len(x)).take(10)

# check for duplicates (distinct works on RDDs and pair RDDs)
print pairs_wds.count()
print pairs_wds.distinct().count()

# this should give an empty list since both RDDs are equal
print wc.subtract(repeat).collect()

a = sc.parallelize([(1, 2), (3, 4), (3, 6)])
b = sc.parallelize([(3, 9)])

# remove elements with a key present in the 2nd RDD
a.subtractByKey(b).collect()

# inner join
a.join(b).collect()

# inner join
b.join(a).collect()

# rightOuterJoin
a.rightOuterJoin(b).collect()

# rightOuterJoin
b.rightOuterJoin(a).collect()

# leftOuterJoin
a.leftOuterJoin(b).collect()

# cogroup gives the keys and a list of corresponding values
a.cogroup(b).mapValues(lambda value: [item for val in value for item in val]).collect()

# combine per key is the most general aggregation function that most
# other functions are built on; like aggregate the return type can
# different from the original type
print pairs_num.take(10)
print pairs_num.keys().count(), pairs_num.keys().distinct().count()

pairs_num.combineByKey(createCombiner=(lambda x: (x, 1)),
                       mergeValue=(lambda x, y: (x[0] + y, x[1] + 1)),
                       mergeCombiners=(lambda x, y: (x[0] + y[0], x[1] + y[1]))).collectAsMap()

# the number of partitions the RDD exists on
pairs_num.getNumPartitions()

pairs_num.countByKey().items()[:10]

print pairs_num.lookup(14)
print pairs_num.lookup(17)

