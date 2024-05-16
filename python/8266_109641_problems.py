



def shortestDistance(self, grid):
    if not grid or not grid[0]: 
        return -1
    
    rows, cols = len(grid), len(grid[0])
    objectives = sum(val for line in grid for val in line if val == 1)
    
    hit = [[0] * cols for i in range(rows)]
    distSum = [[0] * cols for i in range(rows)]

    def BFS(start_x, start_y):
        visited = [[False] * cols for k in range(rows)]
        visited[start_x][start_y] = True
        
        count1 = 1
        queue = collections.deque([(start_x, start_y, 0)])
        
        while queue:
            x, y, dist = queue.popleft()
            for i, j in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= i < rows and 0 <= j < cols and not visited[i][j]:
                    visited[i][j] = True
                    if not grid[i][j]:
                        queue.append((i, j, dist + 1))
                        hit[i][j] += 1
                        distSum[i][j] += dist + 1
                    elif grid[i][j] == 1:
                        count1 += 1
        return count1 == objectives  

    for x in range(rows):
        for y in range(cols):
            if grid[x][y] == 1:
                if not BFS(x, y): return -1
    
    return min([distSum[i][j] for i in range(rows) for j in range(cols) if not grid[i][j] and hit[i][j] == objectives] or [-1])



