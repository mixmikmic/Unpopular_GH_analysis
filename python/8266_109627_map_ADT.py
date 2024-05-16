from collections import MutableMapping

class MapBase(MutableMapping):
    class Item:
        __slots__ = '_key', '_value'
        
        def __init__(self,k,v):
            self._key = k
            self._value = v
            
            



