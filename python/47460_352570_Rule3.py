import re
import pandas as pd
import pprint as pp

def load_csv(filename):
    try:
        f = open(filename)
    except:
        pp.pprint('Bad filename ' + filename)
        return None
    words = f.read().split(',')
    return words

def regex_for_word(word):
    return word.replace('*', '[a-zA-Z]+')

# Save the regexes to find unnecessary words as a global variable
unnecessary_regexes = load_csv('unnecessary_words.csv')

def remove_quotes_from_text(text):
    # Check for all types of quotes
    quote_regex = r'"(.*?)"|“(.*?)”'
    text = re.sub(quote_regex, '', text)
    return text

def find_phrases_in_text(text, phrases):
    phrase_list = []
    for phrase in phrases:
        phrase_count = len(re.findall(regex_for_word(phrase), text, flags=re.IGNORECASE))
        if phrase_count is not 0:
            phrase_list.append((phrase, phrase_count))
    return phrase_list

def unnecessary_phrase_count_in_text(text):
    text = remove_quotes_from_text(text)
    text_phrases = find_phrases_in_text(text, unnecessary_regexes)
    frame = pd.DataFrame(text_phrases)
    frame.columns = ['PHRASE', 'COUNT']
    return frame

# This article can be found here:
# http://www.newyorker.com/magazine/2008/10/20/late-bloomers-malcolm-gladwell
def test_on_gladwell():
    with open('gladwell_latebloomers.txt', 'r') as f:
        rule3_count = unnecessary_phrase_count_in_text(f.read())
        print(rule3_count)

def rule3_ranges_in_text(text):
    phrase_location_list = []
    for phrase in unnecessary_regexes:
        phrase_matches = re.finditer(regex_for_word(phrase), text, flags=re.IGNORECASE)
        for phrase_match in phrase_matches:            
            phrase_location_list.append(phrase_match.span())
    return [(start, end - start) for (start, end) in phrase_location_list]

