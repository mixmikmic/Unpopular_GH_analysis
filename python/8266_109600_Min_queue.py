class MinQueue:
    
    def __init__(self):
        self.queue = [None]
        self.size = 0

    def _getParentIdx(self, idx):
        return idx / 2
    
    def lookParent(self, idx):
        return _lookIdx(_getParentIdx(idx))
    
    def _getLeftChildIdx(self, idx):
        return idx * 2
    
    def _lookLeftChild(self, idx):
        return _lookIdx(self._getLeftChildIdx(idx))
    
    def _getRightChildIdx(self, idx):
        return idx * 2 + 1
    
    def _lookRightChild(self, idx):
        return _lookIdx(self._getRightChildIdx(idx))
    
    def _lookIdx(self, idx):
        return self.queue[idx]
    
    def _swap(a,b):
        self.queue[a], self.queue[b] = self.queue[b], self.queue[a]
    
    def _nodeExists(self, idx):
        return idx > 0 and idx <= self.size()
    
    def _shiftUp(self, idx):
        if (_nodeExists(_getParentIdx(idx))) and (lookParent(idx) < _lookIdx(idx)):
            _swap(_getParentIdx(idx), idx)
            _shiftUp(_getParentIdx(idx))
        else:
            return
    
    def _shiftDown(self, idx):
        if not _nodeExists(_getLeftChildIdx(idx)):
            return
        elif not _nodeExists(_getRightChildIdx(idx)):
            if (_lookIdx(idx) > _lookLeftChild(idx)):
                _swap(_getLeftChildIdx(idx), idx)
        
        elif (_lookIdx(idx) > _lookLeftChild(idx)) or (_lookIdx(idx) > _lookRightChild(idx)):
            if (_lookRightChild(idx) > _lookLeftChild(idx)):
                _swap(_getRightChildIdx(idx), idx)
                _shiftDown(self, _getRightChildIdx(idx))
            else:
                _swap(_getLeftChildIdx(idx), idx)
                _shiftDown(self, _getLeftChildIdx(idx))
        else:
            return
    
    def size(self):
        return size
    
    def pop(self):
        minVal = self.queue[1]
        self.queue[1] = self.queue[-1]
        del self.queue[-1]
        self.size -= 1
        _shiftDown(1)
        return minVal
    
    def push(self, item):
        self.queue.append(item)
        self.size += 1
        _shiftUp(self.size)

    
    def look(self):
        return self.queue[0]



