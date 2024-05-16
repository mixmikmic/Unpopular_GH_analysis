import h5py, numpy as np
f = h5py.File("mytestfile.hdf5", "w") # create hdf5 file
dset = f.create_dataset("mydataset", (100,), dtype='i') # create dataset
dset[...] = np.arange(100)  # assign value
dset.name  # u'/mydataset'
f.name   # u'/', root group

grp = f.create_group("subgroup")   # create group
dset2 = grp.create_dataset("another_dataset", (50,), dtype='f') # create dataset from grp
dset2.name  # u'/subgroup/another_dataset'
dset3 = f.create_dataset('subgroup2/dataset_three', (10,), dtype='i') # create dataset and group

# print all keys
for name in f:
  print name
f.keys() # another to access all keys  

# print all keys and subgroups
def printname(name):
  print name
f.visit(printname)

# add attribute to dataset
dset.attrs['temperature'] = 99.5
dset.attrs['temperature']
for name in dset.attrs:
  print name

# add attribute to group  
grp.attrs['hello'] = 9
for i in grp.attrs:
  print i

