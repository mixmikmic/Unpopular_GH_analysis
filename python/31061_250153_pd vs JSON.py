import pandas as pd
import json

df = pd.DataFrame([['a', 'b'], ['c', 'd']],
                   index=['row 1', 'row 2'],
                   columns=['col 1', 'col 2'])

df

a = df.to_json(orient='split')

print json.dumps(json.loads(a), indent=4, sort_keys=True) 

pd.read_json(a, orient='split')

b = df.to_json(orient='index')  ## b is a string, need to be parsed

di = json.loads(b) # parsed into a dictionary

j = json.dumps(di, indent=4, sort_keys=True)  # stringify to a string with indent and order

import pprint ## pprint is not able to parsed, it takes advantage of the existing data structure
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(b)

pp.pprint(di)

print j

for row in di:
    print di[row]

pd.read_json(b, orient='index')

pd.read_json(j, orient='index')

c = df.to_json(orient='records')

print json.dumps(json.loads(c), indent=4, sort_keys=True) 

pd.read_json(c, orient='records')



