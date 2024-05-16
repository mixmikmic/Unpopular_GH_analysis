m = [
    ["O", "O", "O", "O", "O", "O", "O", "W", "O", "G"],
    ["O", "O", "O", "O", "O", "O", "O", "W", "O", "O"],
    ["O", "O", "O", "O", "W", "O", "O", "W", "O", "O"],
    ["O", "O", "O", "O", "W", "O", "O", "W", "W", "W"],
    ["O", "O", "O", "O", "W", "O", "G", "O", "O", "O"],
    ["W", "W", "W", "O", "W", "O", "O", "O", "O", "O"],
    ["O", "O", "O", "O", "W", "O", "O", "O", "O", "O"],
    ["O", "O", "O", "O", "W", "O", "O", "O", "O", "O"],
    ["O", "O", "O", "O", "W", "O", "O", "O", "O", "O"],
    ["G", "O", "O", "O", "W", "O", "O", "O", "O", "O"],
]

def make_guard_distance_map(m):
    n_rows = len(m)
    n_cols = len(m[0])
    
    def valid_next_space(row, col, dist):
        if (-1 < row < n_rows) and (-1 < col < n_cols):
            if not m[row][col] == "G" and not m[row][col] == "W":
                if m[row][col] == "O":
                    return True
                if m[row][col] > dist:
                    return True
        return False
    
    def find_guards():
        guards = []
        for row in range(n_rows):
            for col in range(n_cols):
                if m[row][col] == "G":
                    guards.append((row, col, 0))
        return guards
    
    
    def bfs_step(this_step):
        n_this = len(this_step)
        next_step = []
        for i in range(n_this):
            row, col, dist = this_step.pop()
            # up
            if valid_next_space(row + 1, col, dist + 1):
                m[row + 1][col] = dist + 1
                next_step.append((row + 1, col, dist + 1))
            # down
            if valid_next_space(row - 1, col, dist + 1):
                m[row - 1][col] = dist + 1
                next_step.append((row - 1, col, dist + 1))
            # left
            if valid_next_space(row, col - 1, dist + 1):
                m[row][col - 1] = dist + 1
                next_step.append((row, col - 1, dist + 1))
            # right
            if valid_next_space(row, col + 1, dist + 1):
                m[row][col + 1] = dist + 1
                next_step.append((row, col + 1, dist + 1))

        if next_step:
            bfs_step(next_step)
    
    guards = find_guards()
    
    for g in guards:
        bfs_step([g])

make_guard_distance_map(m)
expected = [
    [10 ,   9,   8,  7,   6,  5,  4,'W',   1, 'G'], 
    [9  ,   8,   7,  6,   5,  4,  3,'W',   2,   1], 
    [10 ,   9,   8,  7, 'W',  3,  2,'W',   3,   2], 
    [11 ,  10,   9,  8, 'W',  2,  1,'W', 'W', 'W'], 
    [11 ,  10,   9,  8, 'W',  1,'G',  1,   2,   3], 
    ['W', 'W', 'W',  7, 'W',  2,  1,  2,   3,   4], 
    [  3,   4,   5,  6, 'W',  3,  2,  3,   4,   5], 
    [  2,   3,   4,  5, 'W',  4,  3,  4,   5,   6], 
    [  1,   2,   3,  4, 'W',  5,  4,  5,   6,   7], 
    ['G',   1,   2,  3, 'W',  6,  5,  6,   7,   8]
]
print(m == expected)





