def remove_item_from_array(arr, item):
    k = 0
    l = len(arr)
    for idx, j in enumerate(arr):
        if j == item:
            k += 1
        else:
            arr[idx - k] = j
    
    return arr[0:(l-k)]


arr1 = [0,1,2,3,0,4,5,6,7,0,8,0,9]
arr2 = [9,2,3,4,1,5,5,6,0,0,0,3,0]
arr3 = [0,0,0,0,0,0,0,1,0]

print(remove_item_from_array(arr1,0))
print(remove_item_from_array(arr2,0))
print(remove_item_from_array(arr3,0))
print(remove_item_from_array(arr3,1))

def test_cell(y_i, x_i, s, mat):
    return (y_i > 0) and (x_i > 0) and (y_i < len(max)) and (x_i < len(mat[0])) and (s[0] == mat[y_i][x_i])

def strDFS(y, x, s, mat):
    if len(s) == 0:
        return True
    # up
    elif test_cell(y-1, x, s, mat):
        strDFS(y-1, x, s[1:], mat)
    # down
    elif test_cell(y+1, x, s, mat):
        strDFS(y+1, x, s[1:], mat)
    # left
    elif test_cell(y, x-1, s, mat):
        strDFS(y, x-1, s[1:], mat)
    # right
    elif test_cell(y, x+1, s, mat):
        strDFS(y, x+1, s[1:], mat)
    else:
        return False

def find_string_in_char_matrix(mat, s):
    found = []
    if not ((len(s) == 0) or (len(mat) == 0)):
        for i in range(len(mat[0])):
            for j in range(len(mat)):
                if (mat[j][i] == s[0]) and strDFS(j, i, s[1:], mat):
                    found.append((j,i))
    return found

def search_in_rotation(arr, lP, rP, item):
    if lP > rP:
        return None
    elif item == arr[lP]:
        return lP
    elif item == arr[rP]:
        return rP
    
    mP = lP + ((rP - lP) // 2)
    if arr[mP] == item:
        return mP
    
    if arr[lP] < arr[mP]:
        # in first increasing array
        if item > arr[lP] and item < arr[mP]:
            # search left
            return search_in_rotation(arr, lP+1, mP-1, item)
        else:
            # search right
            return search_in_rotation(arr, mP+1, rP-1, item)
    else:
        # in the second increasing array
        if item > arr[mP] and item < arr[rP]:
            # search right
            return search_in_rotation(arr, mP+1, rP-1, item)
        else:
            # search left
            return search_in_rotation(arr, lP+1, mP-1, item)

def binary_search_rotated_array(array, item):
    l = len(array)
    if l == 0:
        return None
    elif l == 1 and array[0] == item:
        return 0
    
    return search_in_rotation(array, 0, len(array)-1, item)
        
    
a = list(range(20))
a = a[5:] + a[:5]
print(a)
print(binary_search_rotated_array(a, 15))

def is_panadrome(s):
    if len(s) == 1 or len(s) == 2:
        return True
    else:
        m = len(s) // 2
        # O(n)
        return s[:m] == s[-m:][::-1]

def min_palandrome_splits(string):
    if len(string) == 0:
        return -1
    elif is_panadrome(string):
        return 0
    else:
        # the last element contains the minimum, worst case min_splits == n
        splits = list(range(1,len(string)+1))
        
        for i in range(len(string)):
            if is_panadrome(string[:i]):
                splits[i] = 0
            
            for j in range(0,i):
                if is_panadrome(string[j+1:i]) and splits[i] > splits[j] + 1:
                    splits[i] == splits[j] + 1
        
    return splits[-1]

s = "madamifmadam"
print(is_panadrome(s))
print(min_palandrome_splits(s))

    

class TreeNode:
    
    def __init__(self, key, left=None, right=None):
        self.left = left
        self.right = right
        self.key = key
    
    def is_leaf(self):
        if self.left or self.right:
            return False
        else:
            return True


tree1 = TreeNode('A', 
                TreeNode('B',
                         TreeNode('D'),
                         TreeNode('E')
                        ),
                TreeNode('C')
                )


tree2 = TreeNode('A', TreeNode('D'), TreeNode('B'))

leaves = [[],[]]

def collectLeaves(tree1, tree2):
    
    # pre-order dfs
    def dfs(node, tree_num):
        if node.is_leaf():
            leaves[tree_num].append(node.key)
        if node.left:
            dfs(node.left, tree_num)
        else:
            leaves[tree_num].append(None)
        if node.right:
            dfs(node.right, tree_num)
        else:
            leaves[tree_num].append(None)
    
    dfs(tree1, 0)
    dfs(tree2, 1)

    

def compare_leaves(leaves):
    if len(leaves[0]) > len(leaves[1]):
        for i in range(len(leaves[0])):
            l1 = leaves[0][i]
            if i > len(leaves[1])-1:
                l2 = None
            else:
                l2 = leaves[1][i]
            if not l1 == l2:
                return (l1,l2)
    else:
        for i in range(len(leaves[1])):
            l2 = leaves[1][i]
            if i > len(leaves[0])-1:
                l1 = None
            else:
                l1 = leaves[0][i]
            if not l1 == l2:
                return (l1,l2)
    
    return (None, None)
# there is a weird bug in traversing the second tree, it goes right then left
collectLeaves(tree1, tree2)
print(leaves)
compare_leaves(leaves)

def sort_squared(arr):
    negs = []
    output = []
    for i in arr:
        j = i**2
        if i < 0:
            negs.append(j)
        else:
            if len(negs) > 0 and j >= negs[-1]:
                output.append(negs.pop())
            output.append(j)
    # if there are negatives left
    
    for j in negs[::-1]:
        output.append(j)
    
    return output


a = list(range(-10,10))
sort_squared(a)

def compute_processing_time(tasks, k):
    if k == 0:
        return len(tasks)
    if len(tasks) == 1:
        return 1
    
    time = 0
    last_time = {}
    
    for task in tasks:
        if task in last_time and (time - last_time[task] < k):
            time = last_time[task] + k
        time += 1
        last_time[task] = time
    
    return time

tasks1 = ['A','B','C','D'] # k = 3, 4
tasks2 = ['A','B','A','C'] # k = 3, 6
tasks3 = ['A','A','A','A'] # k = 4, 16

print(compute_processing_time(tasks1, 3))
print(compute_processing_time(tasks2, 3))
print(compute_processing_time(tasks3, 4))

class BinToTsvConverter:

    def __init__(self):
        """
        initializes the class
        """
    
    def convert(self,filename):
        """
        converts the file and writes filename.tsv
        """

def read_calculate_latency_bandwidth(filenames):
    
    connections_total = 0
    latency_total = 0
    bandwidth_total = 0
    rows_total = 0
    
    converter = BinToTsvConverter()
    
    for filename in filenames:
        converter.convert(filename)
        with open("{}.tsv".format(filename)) as f:
            for line in f.readlines():
                connections, latency, bandwidth = line.split("\t")
                connections_total += connections
                latency_total += latency
                bandwidth_total += bandwidth
                rows_total += 1
    
    return (latency_total / rows_total, bandwidth_total)


# map-reduce style

def read_file_calc_stats(filename):
    converter = BinToTsvConverter()
    converter.convert(filename)
    
    connections_total = 0
    latency_total = 0
    bandwidth_total = 0
    rows_total = 0
    
    with open("{}.tsv".format(filename)) as f:
        for line in f.readlines():
            connections, latency, bandwidth = line.split("\t")
            connections_total += connections
            latency_total += latency
            bandwidth_total += bandwidth
            rows_total += 1
    
    return (connections_total, latency_total, bandwidth_total, rows_total)

def combine_stats(t1, t2):
    return (t1[0] + t2[0], t1[1] + t2[1], t1[2] + t2[2], t1[3] + t2[3])

def compute_latency_bandwidth(summary_stats):
    return (summary_stats[1] / summary_stats[3], summary_stats[2])

# compute_latency_bandwidth(reduce(combine_stats, map(read_file_calc_stats, filenames)))

def find_zero_triples(arr):
    s = set(arr)
    l = list(s)
    output = set()
    
    for i, a in enumerate(l[:-2]):
        for b in l[i+1:]:
            c = -(a+b)
            if not (c == a) and not (c == b) and c in s:
                # these aren't unique solutions
                output.add(",".join(sorted([a,b,c])))
                
    return output

a = list(range(-10,10))
find_zero_triples(a)



