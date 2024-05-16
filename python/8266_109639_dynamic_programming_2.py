# This is solved most efficiently at query time by doing some pre-processing.  
# When the method is configured, we compute a Summed-Area-Table.

def getSafeMatValue(m, x, y):
    r = len(m)
    c = len(m[0])
    if y >= 0 and y < r and x >=0 and x < c:
        return m[y][x]
    else:
        return 0
        
def buildSAT(m):
    r = len(m)
    c = len(m[0])
    SAT = [[0 for x in range(c)] for y in range(r)]
    for x in range(c):
        for y in range(r):
            SAT[y][x] = getSafeMatValue(m, x, y)
            + getSafeMatValue(SAT,x-1,y) 
            + getSafeMatValue(SAT,x,y-1) 
            - getSafeMatValue(SAT,x-1,y-1)
    return SAT

def config_sum_matrix_sum(m):
    SAT = buildSAT(m)
    def sumMatrixSum(x1, y1, x2, y2):
        s = getSafeMatValue(SAT, x2, y2)
        + getSafeMatValue(SAT, x1, y1)
        - getSafeMatValue(SAT,y1-1,x2) 
        - getSafeMatValue(SAT,y2,x1-1)
        return s
    return sumMatrixSum




def getMinSubArray(A, q):
    if q >= len(A):
        return sum(A)
    
    n = len(A)
    minArray = A[:q]
    lastArray = A[:q]
    lastSum = sum(minArray)
    minSum = sum(minArray)
    
    for i in range(1, n - q + 1):
        rightIdx = i + q
        newSum = lastSum + A[rightIdx] - lastArray[0]
        newSubArray = lastArray[1:].append(A[rightIdx])
        if (newSum < minSum):
            minArray = nextSubArray
            minSum = newSum
        
        lastSum = newSum
        lastArray = newSubArray
    
    return (lastSum, lastArray)
        



