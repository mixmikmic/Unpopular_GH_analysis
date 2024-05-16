get_ipython().magic('matplotlib inline')
#%load_ext autoreload
#%autoreload 2
get_ipython().magic('reload_ext autoreload')
import numpy as np
import matplotlib.pyplot as plt
import math, sys, os
from numpy.random import randn

PROJECT_HOME = os.environ.get('PROJECT_HOME', None)
sys.path.insert(0, PROJECT_HOME + "/util")
from loaders import get_english_dictionary

class Trie_dict:
    
    def __init__(self):
        self._end = '_end_'
        self._root = dict()
    
    def insert(self, word):
        current_dict = self._root
        for letter in word:
            current_dict = current_dict.setdefault(letter, {})
        current_dict[self._end] = self._end
    
    def insert_batch(self, words):
        for word in words:
            self.insert(word)
    
    def view(self):
        print(self._root)
    
    def view_root_keys(self):
        print(self._root.keys())
    
    def contains(self, word):
        current_dict = self._root
        for letter in word:
            if letter in current_dict:
                current_dict = current_dict[letter]
            else:
                return False
        # the _end flag indicates this is the end of a word
        # if it's not there, the word continues
        if self._end in current_dict:
            return True
        else:
            return False
    

    def suggest(self, partial, limit = 5):
        """
        Since this trie doesn't store frequency of words as it trains, we're just 
        going to return the alphabetically first 'limit', shortest, terms.
        """
        suggestions = []

        def suggest_dfs(partial_dict, partial ):
                if len(suggestions) < limit:
                    for ch in sorted(partial_dict.keys()): 
                        # sorting by alpha, this happens to give us _end_ first
                        # could be pre-sorting by frequency for better 
                        #   speed and smarted recommendations
                        if len(suggestions) >= limit:
                            break
                        elif ch == self._end:
                            suggestions.append(partial)
                        else:
                            # recurse
                            suggest_dfs(partial_dict[ch], partial + ch)

        partial_dict = self._find_patial(partial)
        if not partial_dict == None:
            suggest_dfs(partial_dict, partial)
        
        return suggestions

    def _find_patial(self, partial):
        top_dict = self._root
        for char in partial:
            if char in top_dict:
                top_dict = top_dict[char]
            else:
                # there are no words starting with this sequence
                return None
        return top_dict

        

# A note on the dictionary.set_default(key, default_val) method.  
# This method is equivilant to a method that looks like this:
def set_default(dictionary, key, default_val = {}):
    if key in dictionary:
        return dictionary[key]
    else:
        dictionary[key] = default_val
        return dictionary[key]

trie = Trie_dict()
trie.insert_batch(get_english_dictionary())

print("Suggestions")
print("")
print("'reac': ")
print(trie.suggest("reac"))
print( "")
print( "'poo': ")
print( trie.suggest("poo"))
print( "")
print( "'whal': ")
print( trie.suggest("whal"))
print( "")
print( "'dan': ")
print( trie.suggest("dan"))
print( "")

class Trie_Statistical:
    
    def __init__(self):
        self._end = '_end_'
        self._root = dict()
        self._total_words = 0
        self._search_limit = 100
    
    def insert(self, word):
        current_dict = self._root
        for letter in word:
            current_dict = current_dict.setdefault(letter, {})
        # keep counts at { last_letter : {'_end_' : count} }
        if self._end in current_dict:
            current_dict[self._end] += 1
        else:
            current_dict[self._end] = 1
        self._total_words += 1
        
    
    def insert_batch(self, words):
        for word in words:
            self.insert(word)
    
    def view(self):
        print(self._root)
    
    def view_root_keys(self):
        print(self._root.keys())
    
    def _normalize_suggestion_probs(self, suggestions):
        total = 0
        for w, c in suggestions:
            total += c
        for i, t in enumerate(suggestions):
            suggestions[i] = (t[0], t[1] / total)
    
    def contains(self, word):
        current_dict = self._root
        for letter in word:
            if letter in current_dict:
                current_dict = current_dict[letter]
            else:
                return False
        # the _end flag indicates this is the end of a word
        # if it's not there, the word continues
        if self._end in current_dict:
            return True
        else:
            return False
    
    def suggest(self, partial, limit = 5):
        """
        """
        suggestions = []

        def suggest_dfs(partial_dict, partial ):
                if len(suggestions) < self._search_limit:
                    for ch in sorted(partial_dict.keys()): 
                        # sorting by alpha, this happens to give us _end_ first
                        # could be pre-sorting by frequency for better 
                        #   speed and smarter recommendations
                        if len(suggestions) >= self._search_limit:
                            break
                        elif ch == self._end:
                            suggestions.append((partial, partial_dict[self._end]))
                        else:
                            # recurse
                            suggest_dfs(partial_dict[ch], partial + ch)

        partial_dict = self._find_patial(partial)
        if not partial_dict == None:
            suggest_dfs(partial_dict, partial)
        
        self._normalize_suggestion_probs(suggestions)
        sorted_suggestions = sorted(suggestions, key=lambda pair: pair[1])
        if limit > 0:
            return sorted_suggestions[:limit]
        else:
            return sorted_suggestions


    def _find_patial(self, partial):
        top_dict = self._root
        for char in partial:
            if char in top_dict:
                top_dict = top_dict[char]
            else:
                # there are no words starting with this sequence
                return None
        return top_dict

trie = Trie_Statistical()
# we're reading a dictionary, so we will have 1 example of every word.
trie.insert_batch(get_english_dictionary())

print("Suggestions")
print("")
print("'reac': ")
print(trie.suggest("reac"))
print( "")
print( "'poo': ")
print( trie.suggest("poo"))
print( "")
print( "'whal': ")
print( trie.suggest("whal"))
print( "")
print( "'dan': ")
print( trie.suggest("dan"))
print( "")



