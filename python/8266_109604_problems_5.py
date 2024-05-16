from random import shuffle

def find_missing_number(arr):
    n = len(arr) + 1 # O(1)
    expected_sum = int(n * (n + 1) / 2)
    s = sum(arr) # O(n)
    return expected_sum - s

arr = list(range(1,11))
shuffle(arr)
print(arr.pop(4))
print(find_missing_number(arr))

def find_second_most_frequent(arr):
    counts_map = {arr[0] : 1}
    first_most = arr[0]
    idx = 1
    while arr[idx] == first_most:
        idx += 1
        counts_map[arr[idx]] += 1
    
    counts_map[arr[idx]] = 1
    second_most = arr[idx]
    idx += 1
    while arr[idx] == second_most:
        idx += 1
        counts_map[arr[idx]] += 1
    
    if counts_map[second_most] > counts_map[first_most]:
        second_most, first_most = first_most, second_most
    
    n = len(arr)
    # now we walk the rest of the array
    while idx < n:
        i = arr[idx]
        
        # increment
        if i in counts_map:
            counts_map[i] += 1
        else:
            counts_map[i] = 1
        
        # check if it was the second most, and if it's now the first most
        if i == second_most and counts_map[i] > counts_map[first_most]:
            # swap them
            first_most, second_most = second_most, first_most
        
        # check if it's now the second most
        elif not i == first_most and counts_map[i] > counts_map[second_most]:
            # replace the second most
            second_most = i
        idx += 1
    
    if counts_map[second_most] == counts_map[first_most]:
        return (first_most, second_most)
    else:
        return second_most

    
arr1 = ['a','b','c','a','a','a','b','b','b','c','d','d','d','e','e','e','e','e','e','e','e']
print(find_second_most_frequent(arr1))
arr2 = ['a','b','c','a','a','a','b','a','a','a','a','b','b','c','d','d','d','e','e','e','e','e','e','e','e']
print(find_second_most_frequent(arr2))

INT_MAX = 10000000000
INT_MIN =-10000000000
 
# A binary tree node
class Node:
 
    # Constructor to create a new node
    def __init__(self, val, left = None, right = None):
        self.val = val 
        self.left = left
        self.right = right


def is_BST(root, mi, ma):
    
    def is_BST_rec(node, mi, ma):
        if node is None:
            return True
        
        if node.val < mi or node.val > ma:
            return False
        
        return is_BST_rec(node.left, mi, node.val - 1) and is_BST_rec(node.right, node.val + 1, ma)

    
    return is_BST_rec(root, mi, ma)

root = Node(4,
           Node(2, 
                Node(1), 
                Node(3)
               ),
           Node(5)
           )

print(is_BST(root, INT_MIN, INT_MAX))

class Node:
    
    def __init__(self, val, children = []):
        self.val = val
        self.children = children

def get_depth_difference(root, val1, val2):
    
    depth1 = None
    depth2 = None
    depth = 0
    
    def dfs_rec(node, val1, val2, depth1, depth2, cur_d):
        if not depth1 and not depth2:
            return (depth1, depth2)
        if node.val == val1:
            depth1 = cur_d
        if node.val == val2:
            depth2 = cur_d
        
        for c in node.children:
            dfs_rec(c, val1, val2, depth1, depth2, cur_d+1)
            
    dfs_rec(root, val1, val2, depth1, depth2, 0)
    return abs(depth1 - depth2)

from random import randint

class MediaPlayer:
    
    def __init__(self, songs):
        self.songs = songs
        self.n = len(songs)
        self.indices = list(range(n))
    
    def add_song(self, song):
        self.songs.append(song)
        self.n += 1
        self.indices.append(self.n)
        if n > 1:
            # put newest at second to last to maintain current at index -1
            self.indices[-2], self.indices[-1] = self.indices[-1], self.indices[-2]
    
    def get_song(self, idx):
        # read the song from directory and return it
        return self.song[idx]
    
    def play(self, song):
        # play the song
        print("playing song")
    
    def start(self):
        idx = randint(0,n)
        # put current songs idx at the end
        self.indices[idx], self.indices[-1] = self.indices[-1], self.indices[idx]
        return idx
    
    def next(self):
        idx = randint(0, n-1)
        self.indices[idx], self.indices[-1] = self.indices[-1], self.indices[idx]
        return idx

def divide(num, denom, remainder = False):
    c = 0
    
    while num >= denom:
        num -= denom
        c += 1
    
    if remainder:
        return (c, num)
    else:
        return c

print(divide(26, 5))
print(divide(3, 1))
print(divide(310432, 323, True))



