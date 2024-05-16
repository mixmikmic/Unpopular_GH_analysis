lines = sc.textFile('text_file.md')
lines.take(5)

pcount = sc.accumulator(0)
tcount = sc.accumulator(0)

def countPython(line):
    global pcount, tcount
    if ('python' in line.lower()): pcount += 1
    if ('the' in line.lower()): tcount += 1

lines.foreach(countPython)
print lines.first()
print pcount.value, tcount.value

keywords = ['code', 'hardware', 'causality', 'engine', 'computer']
ret = sc.broadcast(keywords)

out = lines.filter(lambda x: any([keyword in x for keyword in keywords]))
print out.count()

print out.collect()

lens = lines.map(lambda x: len(x))
print lens.take(3)

def combineCtrs(c1, c2):
    return (c1[0] + c2[0], c1[1] + c2[1])

def partitionCounters(nums):
    sumCount = [0, 0]
    for num in nums:
        sumCount[0] += num
        sumCount[1] += 1
    return [sumCount]

def fastAvg(nums):
    sumCount = nums.mapPartitions(partitionCounters).reduce(combineCtrs)
    return sumCount[0] / float(sumCount[1])

fastAvg(lens)

lens.mean()

pairs = lens.map(lambda x: (x, 1))
# pairs.mean() this line fails because not a numeric RDD

stats = lens.stats()
mu = stats.mean()
print mu

lines.filter(lambda x: len(x) > 0).reduce(lambda x,y: x[0] + y[1])

lines.getNumPartitions()

