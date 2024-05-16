class LRUCacheNaive:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tm = 0
        self.cache = {}
        self.lru = {}

    def get(self, key):
        if key in self.cache:
            self.lru[key] = self.tm
            self.tm += 1
            return self.cache[key]
        return -1

    def set(self, key, value):
        if len(self.cache) >= self.capacity:
            # find the LRU entry
            # this is an O(n) search for the minimum lookup value
            old_key = min(self.lru.keys(), key=lambda k:self.lru[k])
            self.cache.pop(old_key)
            self.lru.pop(old_key)
        self.cache[key] = value
        self.lru[key] = self.tm
        self.tm += 1

import collections

class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        # ordered dict
        self.cache = collections.OrderedDict()

    def get(self, key):
        # we re-insert the item after each lookup, thus the ordered dict is ordering items by their last access
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return -1

    def set(self, key, value):
        try:
            # if in the cache, remove it
            self.cache.pop(key)
        except KeyError:
            # if we're over capacity, remove the oldest item
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        # re-insert it
        self.cache[key] = value

class Node:
    def __init__(self, key, value, prev = None, next = None):
        self.key = key
        self.value = value
        self.next = next
        self.prev = prev

    def pop_self(self):
        self.prev.next = self.next
        self.next.prev = self.prev
        return self.key


class DoubleLinkedList:

    def __init__(self):
        # explicit head and tail nodes
        self.tail = Node("tail", None, None, None)
        self.head = Node("head", None, prev = None, next = self.tail)
        self.tail.prev = self.head

    def insert(self, new_node):
        new_node.prev = self.tail.prev
        new_node.next = self.tail
        self.tail.prev.next = new_node # old last node points to new
        self.tail.prev = new_node # old tail points to newest node

    def pop_oldest(self):
        return self.head.next.pop_self()


class LRUCache:
    
    def __init__(self, capacity = 10):
        self.capacity = capacity
        self.size = 0
        self.cache = {}
        self.order = DoubleLinkedList()

    
    def get(self, key):
        if key in self.cache():
            # if in cache, remove from queue and re-instert at end
            node = self.cache[key]
            node.pop_self()
            self.order.insert(node)
            return node.value
        else:
            return None
    
    
    def set(self, key, value):
        if (self.size + 1) > self.capacity:
            oldest_key = self.order.pop_oldest() # remove oldest form order
            self.cache.pop(oldest_key) # remove oldest from dict
        else:
            self.size += 1
        new_node = Node(key, value, None, None)
        self.cache[key] = new_node # insert new into dict
        self.order.insert(new_node) # insert new into end of ordering
    
    
    def remove(self, key):
        if key in self.cache:
            node = self.cache.pop(key)
            node.pop_self()
            self.size -= 1
            return node.value
        else:
            return None



