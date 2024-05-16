import os
from sympy import *
import numpy as np
import pandas as pd
init_printing()

# initialize some variables, count the types of elements
num_rlc = 0 # number of passive elements
num_v = 0    # number of independent voltage sources
num_i = 0    # number of independent current sources
num_opamps = 0   # number of op amps
num_vcvs = 0     # number of controlled sources of various types
num_vccs = 0
num_cccs = 0
num_ccvs = 0
num_cpld_ind = 0 # number of coupled inductors

fn = 'example420'
fd1 = open(fn+'.net','r')
content = fd1.readlines()
content = [x.strip() for x in content]  #remove leading and trailing white space
# remove empty lines
while '' in content:
    content.pop(content.index(''))

# remove comment lines, these start with a asterisk *
content = [n for n in content if not n.startswith('*')]
# remove spice directives. these start with a period, .
content = [n for n in content if not n.startswith('.')]
# converts 1st letter to upper case
#content = [x.upper() for x in content] <- this converts all to upper case
content = [x.capitalize() for x in content]
# removes extra spaces between entries
content = [' '.join(x.split()) for x in content]

branch_cnt = len(content)
# check number of entries on each line
for i in range(branch_cnt):
    x = content[i][0]
    tk_cnt = len(content[i].split())

    if (x == 'R') or (x == 'L') or (x == 'C'):
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_rlc += 1
    elif x == 'V':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_v += 1
    elif x == 'I':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_i += 1
    elif x == 'O':
        if tk_cnt != 4:
            print("branch {:d} not formatted correctly, {:s}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_opamps += 1
    elif x == 'E':
        if (tk_cnt != 6):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 6".format(tk_cnt))
        num_vcvs += 1
    elif x == 'G':
        if (tk_cnt != 6):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 6".format(tk_cnt))
        num_vccs += 1
    elif x == 'F':
        if (tk_cnt != 5):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 5".format(tk_cnt))
        num_cccs += 1
    elif x == 'H':
        if (tk_cnt != 5):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 5".format(tk_cnt))
        num_ccvs += 1
    elif x == 'K':
        if (tk_cnt != 4):
            print("branch {:d} not formatted correctly, {}".format(i,content[i]))
            print("had {:d} items and should only be 4".format(tk_cnt))
        num_cpld_ind += 1
    else:
        print("unknown element type in branch {:d}, {}".format(i,content[i]))

# build the pandas data frame
count = []        # data frame index
element = []      # type of element
p_node = []       # positive node
n_node = []       # neg node, for a current source, the arrow terminal
cp_node = []      # controlling positive node of branch
cn_node = []      # controlling negitive node of branch
v_out = []        # op amp output node
value = []        # value of element or voltage
v_name = []       # voltage source through which the controlling current flows
l_name1 = []      # name of coupled inductor 1
l_name2 = []      # name of coupled inductor 2

df = pd.DataFrame(index=count, columns=['element','p node','n node','cp node','cn node',
    'v out','value','v name','l_name1','l_name2'])

# loads voltage or current sources into branch structure
def indep_source(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'value'] = float(tk[3])

# loads passive elements into branch structure
def rlc_element(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'value'] = float(tk[3])

'''
loads multi-terminal sub-networks
into branch structure
Types:
E - VCVS
G - VCCS
F - CCCS
H - CCVS
not implemented yet:
K - Coupled inductors
O - Op Amps
'''
def opamp_sub_network(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'v out'] = int(tk[3])

def vccs_sub_network(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'cp node'] = int(tk[3])
    df.loc[br_nu,'cn node'] = int(tk[4])
    df.loc[br_nu,'value'] = float(tk[5])

def vcvs_sub_network(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'cp node'] = int(tk[3])
    df.loc[br_nu,'cn node'] = int(tk[4])
    df.loc[br_nu,'value'] = float(tk[5])

def cccs_sub_network(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'v name'] = tk[3]
    df.loc[br_nu,'value'] = float(tk[4])

def ccvs_sub_network(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'p node'] = int(tk[1])
    df.loc[br_nu,'n node'] = int(tk[2])
    df.loc[br_nu,'v name'] = tk[3]
    df.loc[br_nu,'value'] = float(tk[4])

def cpld_ind_network(br_nu):
    tk = content[br_nu].split()
    df.loc[br_nu,'element'] = tk[0]
    df.loc[br_nu,'l name1'] = tk[1]
    df.loc[br_nu,'l name2'] = tk[2]
    df.loc[br_nu,'value'] = float(tk[3])

# function to scan df and get largest node number
def count_nodes():
    # need to check that nodes are consecutive
    # fill array with node numbers
    p = np.zeros(branch_cnt+1)
    for i in range(branch_cnt-1):
        p[df['p node'][i]] = df['p node'][i]
        p[df['n node'][i]] = df['n node'][i]

    # find the largest node number
    if df['n node'].max() > df['p node'].max():
        largest = df['n node'].max()
    else:
        largest =  df['p node'].max()

        largest = int(largest)
    # check for unfilled elements, skip node 0
    for i in range(1,largest):
        if p[i] == 0:
            print("nodes not in continuous order");

    return largest

# load branches into data frame
for i in range(branch_cnt):
    x = content[i][0]

    if (x == 'R') or (x == 'L') or (x == 'C'):
        rlc_element(i)
    elif (x == 'V') or (x == 'I'):
        indep_source(i)
    elif x == 'O':
        opamp_sub_network(i)
    elif x == 'E':
        vcvs_sub_network(i)
    elif x == 'G':
        vccs_sub_network(i)
    elif x == 'F':
        cccs_sub_network(i)
    elif x == 'H':
        ccvs_sub_network(i)
    elif x == 'K':
        cpld_ind_sub_network(i)
    else:
        print("unknown element type in branch {:d}, {}".format(i,content[i]))

# count number of nodes
num_nodes = count_nodes()

# print a report
print('Net list report')
print('number of branches: {:d}'.format(branch_cnt))
print('number of nodes: {:d}'.format(num_nodes))
print('number of passive components: {:d}'.format(num_rlc))
print('number of independent voltage sources: {:d}'.format(num_v))
print('number of independent current sources: {:d}'.format(num_i))
print('number of op amps: {:d}'.format(num_opamps))

# not implemented yet
print('\nNot implemented yet')
print('number of E - VCVS: {:d}'.format(num_vcvs))
print('number of G - VCCS: {:d}'.format(num_vccs))
print('number of F - CCCS: {:d}'.format(num_cccs))
print('number of F - CCCS: {:d}'.format(num_ccvs))
print('number of K - Coupled inductors: {:d}'.format(num_cpld_ind))

# store the data frame as a pickle file
df.to_pickle(fn+'.pkl')

# initialize some symbolic matrix with zeros
# A is formed by [[G, C] [B, D]]
# Z = [I,E]
# X = [V, J]
V = zeros(num_nodes,1)
I = zeros(num_nodes,1)
G = zeros(num_nodes,num_nodes)
s = Symbol('s')  # the Laplace variable

if (num_v+num_opamps) != 0:
    B = zeros(num_nodes,num_v+num_opamps)
    C = zeros(num_v+num_opamps,num_nodes)
    D = zeros(num_v+num_opamps,num_v+num_opamps)
    E = zeros(num_v+num_opamps,1)
    J = zeros(num_v+num_opamps,1)

# G matrix
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the passive elements, save conductance to temp value
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'R':
        g = 1/sympify(df.loc[i,'element'])
    if x == 'L':
        g = 1/(s/sympify(df.loc[i,'element']))
    if x == 'C':
        g = sympify(df.loc[i,'element'])*s

    if (x == 'R') or (x == 'L') or (x == 'C'):
        # If neither side of the element is connected to ground
        # then subtract it from appropriate location in matrix.
        if (n1 != 0) and (n2 != 0):
            G[n1-1,n2-1] += -g
            G[n2-1,n1-1] += -g

        # If node 1 is connected to ground, add element to diagonal of matrix
        if n1 != 0:
            G[n1-1,n1-1] += g

        # same for for node 2
        if n2 != 0:
            G[n2-1,n2-1] += g

G  # display the G matrix

# generate the I matrix, current sources have N2 = arrow end
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the passive elements, save conductance to temp value
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'I':
        g = sympify(df.loc[i,'element'])
        # sum the current into each node
        if n1 != 0:
            I[n1-1] -= g
        if n2 != 0:
            I[n2-1] += g

I  # display the I matrix

# generate the V matrix
for i in range(num_nodes):
    V[i] = sympify('v{:d}'.format(i+1))

V  # display the V matrix

# generate the B Matrix
# loop through all the branches and process independent voltage sources
sn = 0   # count source number
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the independent voltage sources
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        if num_v+num_opamps > 1:
            if n1 != 0:
                B[n1-1,sn] = 1
            if n2 != 0:
                B[n2-1,sn] = -1
            sn += 1   #increment source count
        else:
            if n1 != 0:
                B[n1-1] = 1
            if n2 != 0:
                B[n2-1] = -1

# loop through all the branches and process op amps
oan = 0   # running count of op amp number
for i in range(branch_cnt):
    n_vout = df.loc[i,'v out'] # node connected to op amp output
    # look for branches with op amps and process
    x = df.loc[i,'element'][0]   # get 1st letter of element name
    if x == 'O':
        B[n_vout-1,oan+num_v] = 1
        oan += 1   # increment op amp count

B   # display the B matrix

# The J matrix is an mx1 matrix, with one entry for the current through each voltage source.
sn = 0   # count source number
oan = 0   #count op amp number
for i in range(branch_cnt):
    # process all the passive elements
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        J[sn] = sympify('I_{:s}'.format(df.loc[i,'element']))
        sn += 1
    if x == 'O':  # this needs to be checked <---- needs debugging
        J[oan+num_v] = sympify('I_{:s}'.format(df.loc[i,'element']))
        oan += 1

J  # diplay the J matrix

# generate the C matrix
sn = 0   # count source number
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    # process all the independent voltage sources
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        if num_v+num_opamps > 1:
            if n1 != 0:
                C[sn,n1-1] = 1
            if n2 != 0:
                C[sn,n2-1] = -1
            sn += 1   #increment source count
        else:
            if n1 != 0:
                C[n1-1] = 1
            if n2 != 0:
                C[n2-1] = -1

# loop through all the branches and process op amps
oan = 0   # running count of op amp number
for i in range(branch_cnt):
    n1 = df.loc[i,'p node']
    n2 = df.loc[i,'n node']
    n_vout = df.loc[i,'v out'] # node connected to op amp output
    # look for branches with op amps and process
    x = df.loc[i,'element'][0]   # get 1st letter of element name
    if x == 'O':
        if n1 != 0:
            C[oan+num_v,n1-1] = 1
        if n2 != 0:
            C[oan+num_v,n2-1] = -1
        oan += 1  # increment op amp number

C   # display the C matrix

# display the The D matrix
D

# generate the E matrix
sn = 0   # count source number
for i in range(branch_cnt):
    # process all the passive elements
    x = df.loc[i,'element'][0]   #get 1st letter of element name
    if x == 'V':
        E[sn] = sympify(df.loc[i,'element'])
        sn += 1

E   # display the E matrix

Z = I[:] + E[:]
Z  # display the Z matrix

X = V[:] + J[:]
X  # display the X matrix

n = num_nodes
m = num_v+num_opamps
A = zeros(m+n,m+n)
for i in range(n):
    for j in range(n):
        A[i,j] = G[i,j]

if num_v+num_opamps > 1:
    for i in range(n):
        for j in range(m):
            A[i,n+j] = B[i,j]
            A[n+j,i] = C[j,i]
else:
    for i in range(n):
        A[i,n] = B[i]
        A[n,i] = C[i]

A  # display the A matrix

# generate the circuit equations
n = num_nodes
m = num_v+num_opamps
eq_temp = 0  # temporary equation used to build up the equation
equ = zeros(m+n,1)  #initialize the array to hold the equations
for i in range(n+m):
    for j in range(n+m):
        eq_temp += A[i,j]*X[j]
    equ[i] = Eq(eq_temp,Z[i])
    eq_temp = 0

equ   # display the equations

str(equ)

str(equ.free_symbols)

df





