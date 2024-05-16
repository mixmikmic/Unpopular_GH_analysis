# Design a deck of cards, a standard deck, to be used for various games

from random import randint

class Card:
    
    def __init__(self, value, suit):
        """
        value is the cards number or type as a string, ace, 2, 3, ..., jack, queen, king
        """
        # check valid values and suits
        self.suit = suit
        self.value = value

    def get_order_value(self, value, ace_high):
        """
        a method to get a numeric value for the cards face
        """
        if value in "23456789" or value == "10":
            order_val = int(value)
        elif value == "jack":
            order_val = 11
        elif value == "queen":
            order_val = 12
        elif value == "king":
            order_val = 13
        elif value == "joker":
            order_val = -1
        elif ace_high:
            order_val = 14
        else:
            order_val = 1
            
        return order_val

class Deck_of_cards:
    
    values = [str(i) for i in range(2,11)] + ["jack", "queen", "king", "ace"]
    suits = ["hearts", "diamonds", "spades", "clubs"]
    jokers = [Card("joker", None), Card("joker", None)]
    standard_deck_no_jokers = [] # standard deck of cards
    
    def __init__(self, num_decks = 1, jokers = False):
        self.num_decks = num_decks
        self.jokers = jokers

    def initialize(self):
        if self.jokers:
            one_deck = standard_deck_no_jokers + jokers
        else:
            one_deck = standard_deck_no_jokers
        
        self.stack = one_deck*self.num_decks # the cards remaining in the deck, in a stack
        self.shuffle()
    
        
    def remaining(self):
        return len(self.stack)
    
    
    def collect_cards(self):
        """
        re-initialize the deck with all cards
        """
        self.stack = one_deck*num_decks
        
    
    def shuffle(self):
        """
        randomly re-order the stack
        """
        n = len(self.stack)
        for i in range(n-1):
            j = randint(i,n-1)
            self.stack[i], self.stack[j] = self.stack[j], self.stack[i]
    
    
    def peek(self):
        return self.stack[0]
    
    
    def draw(self, num = 1):
        return [self.stack.pop(0) for i in range(num)]
    
    
class CardGame:
    
    def __init__(self):
        self.deck = Deck_of_cards()
        
    # game specific behavior

class obj:
    
    def __init__(self,value,size):
        self.value = value
        self.size = size # in MB



class memcache:
    
    def __init__(self, size):
        self.size = size # in MB
        self.cache = {}
    
    def get(self, key):
        return self.cache[key]
    
    def set(self, key, value):
        self.cache[key] = value


class BigCache:
    
    def __init__(self, mem):
        blocks = mem.size
        self.avaiable_blocks = list(range(0, blocks))
        self.mem = mem
        self.cache = {} # map from the object key, to the block inices
    
    def get(self, key):
        if key in self.cache:
            partitions = [self.mem.get(idx) for idx in self.cache[key]]
            return _reconstruct_object(partitions)
        else:
            raise Exception('Key not found')
        
        
    
    def set(self, key, value):
        if len(self.avaiable_blocks) >= _get_necessary_blocks(value):
            self.cache[key] = []
            partitioned = _partition(value)
            for i in partitioned:
                idx = self.avaiable_blocks.pop()
                self.cache[key].append(idx)
                self.mem[idx] = i
        else:
            raise Exception('cache is full')
        
    
    def delete(self, key):
        for idx in self.cache[key]:
            # set blocks associated with this key to available
            # we might want to explicitly remove them, but our memcache api doesn't allow this
            self.avaiable_blocks.append(idx)
    
    def _get_necessary_blocks(self,value):
        blocks = value.size//1
        if (value.size % 1) > 0:
            blocks += 1
        return blocks
    
    def _partition(self, value):
        remainder = value.size % 1
        return [obj(value.value,1) for i in range(value.size//1)] + [(obj(v,remainder))]
    
    def _reconstruct_object(self, partitions):
        size = len(partitions) + partitions[-1].size
        return obj(partitions[0].value, size)



