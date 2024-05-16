lines = sc.textFile('text_file.md', use_unicode=False)

lines.take(5)

py_lines = lines.filter(lambda line: 'Python' in line and 'Java' not in line)
jv_lines = lines.filter(lambda line: 'Java' in line and 'Python' not in line)
print py_lines.union(jv_lines).count(), lines.filter(lambda line: 'Python' in line or 'Java' in line).count()

class ScalaFinder(object):
    def __init__(self, keyword):
        self.keyword = keyword
    def printLines(self, RDD):
        for line in RDD.collect():
            if (self.keyword in line):
                print line

sf = ScalaFinder('and')
sf.printLines(lines)

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
plt.style.use('halverson')

num_chars = lines.map(lambda line: len(line))
left, count = num_chars.histogram(range(0, 120, 10))

plt.bar(left[:-1], count, width=10)
plt.xlabel('Number of words per line')
plt.ylabel('Count')

# flatMap flattens the iterators returned to it
num_chars_sq = lines.flatMap(lambda line: (len(line), len(line)**2))
for item in num_chars_sq.collect():
    if (item > 7000): print item

words = lines.flatMap(lambda line: line.split())
print len(words.collect()), len(words.distinct().collect())

print sorted(words.countByValue().items(), key=lambda (u, v): v, reverse=True)[:10]

# this transformation goes as the square of the number of items
cartProd = lines.cartesian(num_chars)
print cartProd.count(), cartProd.first()

# all-with-all between two RDDs which can be the same
samp_cartProd = cartProd.sample(False, 0.001, seed=0)
print samp_cartProd.collect()

max_item = num_chars.reduce(lambda x, y: x if x > y else y)
print max_item, num_chars.max()

total = num_chars.reduce(lambda x, y: x + y)
print total

total_with_fold = num_chars.fold(0, lambda x, y: x + y)
print total_with_fold

# the first argument is the zero value for the two operations: the first being the elements on a partition
# and the second being the zero value for the combination of the results
sumCount = num_chars.aggregate((0, 0),
                               (lambda acc, value: (acc[0] + value, acc[1] + 1)),
                               (lambda acc1, acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])))
print float(sumCount[0]) / sumCount[1]

# foreach method does not have a return
print lines.first()
lines.foreach(lambda line: line.lower())
print lines.first()
lines = lines.map(lambda line: line.lower())
print lines.first()

num_chars.variance()

# fast re-use is possible through persist with different storage levels
from pyspark import StorageLevel
lines.persist(StorageLevel.MEMORY_ONLY)
print lines.top(5)

