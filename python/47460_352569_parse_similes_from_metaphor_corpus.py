# import xml.etree.ElementTree as etree
from lxml import etree, objectify

root = etree.parse('VUAMC.xml')

## Cleanup xml schema/namespaces from tags ##    
for elem in root.getiterator():
    if not hasattr(elem.tag, 'find'): continue  # (1)
    i = elem.tag.find('}')
    if i >= 0:
        elem.tag = elem.tag[i+1:]
objectify.deannotate(root, cleanup_namespaces=True)

import pandas as pd

def extract_similes(root):
    rows = []
    for sent in root.findall('.//s'): # scan all sentences
        text = ''
        mflag = ''
        mrw = ''
        for word in sent.findall('.//w'): # for each word in sentence
            aseg = word.find('.//seg')
            if aseg is not None:
                if not aseg.text or not aseg.text.strip():
                    continue
                ft = aseg.text.strip()#.encode('UTF-8')
                if aseg.get('function') == 'mFlag': # flag for similes
                    mflag += ' ' + ft
                    text += ' ' + ft
                elif aseg.get('function') == 'mrw' and not (not mflag): # start collecting keywords only after mflag
                    mrw += ' ' + ft
                    text += ' ' + ft
            elif not (not word.text):
                text += ' ' + word.text.strip()#.encode('UTF-8')

        text = text.strip()
        mrw = mrw.strip()
        mflag = mflag.strip()
        if not (not mflag): # we are only interested in similes; for metaphors: if not mflag 
            rows.append([mflag, mrw, text])
    df = pd.DataFrame(rows)
    df.columns = ['mflag', 'mrw', 'sentence']
    return df
   

df = extract_similes(root)
df.to_csv('similes.csv')



