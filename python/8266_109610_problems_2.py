class SparseMatrix:
    
    def __init__(self, nCols, nRows, elements):
        """ Initialize with a size, and some elements. """
        self.num_cols = num_cols
        self.num_cows = num_rows
        self.rows = {}
        for e in elements:
            self.set(e.col, e.row, e.val)
        
    
    def set(self, col, row, val):
        """ update a value inplace, indexes start at zero """
        if col < 0 or col > (self.num_cols - 1):
            raise Exception('col out of bounds')
        elif row < 0 or row > (self.num_rows - 1):
            raise Exception('row out of bounds')
        elif row in self.rows:
            self.rows[row][col] = val
        else:
            self.rows[row] = {col: val}
    
    def sum(self, col, row):
        """ Sum the sub-matrix from (0,0) to (row, col)"""
        """ this can be faster with better data-structures """
        
        for r, cols in self.rows.items():
            s = 0
            if r <= row:
                for c, v in cols.items():
                    if c <= col:
                        s += v
        return s

def convert(number):
    
    firstTwenty = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine","ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
    tens = ["", "twenty", "thirty", "fourty", "fifty", "sixty", "seventy", "eighty", "ninty"]
    thousands = ["", "thousand", "million", "billion", "trillion"]
    
    def teen_string(num):
        if num < 20:
            return firstTwenty[num]
        else:
            return tens[num/20] + " " + firstTwenty[num%20]
    
    def hundreds_string(num):
        output = ""
        if (num / 100) > 0:
            output += firstTwenty[num / 100] + " and "
        output += teen_string(num % 100)
        return output
    
    if number < 1000:
        output = hundreds_string(number)
    else:
        power = 1
        output = hundreds_string(number)
        while number > 0:
            pNum = number % 1000
            output = hundreds_string(pNum) + " " + thousands[power]  + ", " + output
            power += 1
            number = number / 1000
    
    return output

import collections

Node = collections.namedtuple('Node', ['val', 'down', 'right'])

def flatten(head):
    tail = head
    
    def flattenRec(root, output = None):
        tail.right = root
        tail = root
        while tail.down:
            tail.right = tail.down
            tail = tail.down
        if root.right:
            return flattenRec(root.right)
        else:
            return head
        
    return flattenRec(head)

import collections

TreeNode = collections.namedtuple('TreeNode', ['val', 'left', 'right'])

tree_root = TreeNode(6, 
            TreeNode(3,
                 TreeNode(5,
                      TreeNode(9, None, None),
                      TreeNode(2, 
                           None, 
                           TreeNode(7, None, None)
                          )
                     ),
                 TreeNode(1,None,None)
                ),
            TreeNode(4,
                 None,
                 TreeNode(0,
                     TreeNode(8, None, None), 
                     None
                     )
                )
           )

"""
Solution:


Also create an auxilary hashmap of column_number -> array[int], 
    where the ints will be the TreeNode values, in order from top to bottom.
    
We then do a pre-order, depth first traversal, where we keep track of the column number by starting at 0 at the root, 
    then modifying the column number -= 1 when we go left, and += 1 when we go right.
    For each node, we check if the column number is in the map, if it is, 
        we append the current nodes value to the list at that column number, if it is not, 
        we add a new column number key, and a new list with that new node value

After the traversal, we get the keys from the map, and sort them in ascending order.  
    We walk the ordered keys and print the node values at that key, in the order they were inserted.

run time will be O(n) most of the time, unless the tree is degenerate and has a lot of columns.
    In which case, the run-time could be as bad as O(nlogn) because of the sorting of the column numbers
"""
        
        
def print_tree_columns(root):
    columns = {}
    def preorder_dfs_traversal(node, col_num):
        if node is not None:
            if col_num in columns:
                columns[col_num].append(node.val)
            else:
                columns[col_num] = [node.val]
            preorder_dfs_traversal(node.left, col_num - 1)
            preorder_dfs_traversal(node.right, col_num + 1)
    preorder_dfs_traversal(root, 0)
    
    output = ""
    for col_num in sorted(columns.keys()):
        for val in columns[col_num]:
            output += str(val) + " "
    print(output)



