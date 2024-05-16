import pandas as pd
import os
import shutil

batch = False
kallisto_loc = '../input/kallisto_all/'
genmap = pd.read_csv('../input/library_genotype_mapping.txt', comment='#')
genmap.genotype = genmap.genotype.apply(str)
genmap.genotype = genmap.genotype.apply(str.lower) # make sure everything is always in lowercase
# Make all the folders required for sleuth processing
sleuth_loc = '../sleuth/'

genmap.head()

# Make all possible combinations of WT, X
combs = []
for gene in genmap.genotype.unique():
    if gene != 'wt':
        combs += [['WT', gene]]


if not os.path.exists(sleuth_loc):
    os.makedirs(sleuth_loc)

# sort the groups by batches, then do the comparisons by batch
grouped = genmap.groupby('batch')

# do the comparison by batches
for name, group in grouped:
    if batch == True:
        WTnames = genmap[genmap.genotype=='wt'].project_name.values
    else:
        WTnames = group[group.genotype=='wt'].project_name.values
    print(name, )

    # For each combination, make a folder
    for comb in combs:
        current = sleuth_loc + comb[0]+'_'+comb[1]
        MTnames = group[group.genotype == comb[1]].project_name.values
        if len(MTnames) == 0:
            continue
    
        if not os.path.exists(current):
            os.makedirs(current)
    
        # copy the right files into the new directory
        # inside a folder called results
        def copy_cat(src_folder, dst_folder, names):
            """
            A function that copies a set of directories from one place to another.
            """
            for name in names:
#               print('The following file was created:', dst_folder+name)
                shutil.copytree(src_folder + name, dst_folder + name)
        
        # copy WT files into the new directory
        copy_cat(kallisto_loc, current+'/results/', WTnames)
    
        # copy the MT files into the new directory
        copy_cat(kallisto_loc, current+'/results/', MTnames)

def matrix_design(name, factor, df, a, b, directory, batch=False):
    """
    A function that makes the matrix design file for sleuth.
    
    This function can only make single factor design matrices. 
    
    This function requires a folder 'results' to exist within
    'directory', and the 'results' folder in turn must contain
    files that are named exactly the same as in the dataframe.
    
    name - a string
    factor - list of factors to list in columns
    df - a dataframe containing the list of project names and the value for each factor
    i.e. sample1, wt, pathogen_exposed.
    a, b - conditions to slice the df with, i.e: a=WT, b=MT1
    directory - the directory address to place file in folder is in.

    """
    
    with open(directory + name, 'w') as f:
        f.write('# Sleuth design matrix for {0}-{1}\n'.format(a, b))
        f.write('experiment {0}'.format(factor))
        if batch:
            f.write(' batch')
        f.write('\n')
        
        # walk through the results directory and get each folder name
        # write in the factor value by looking in the dataframe
        names = next(os.walk(directory+'/results/'))[1]
        for name in names:
            fval = df[df.project_name == name][factor].values[0]
            
            if batch:
                batchvar = df[df.project_name == name].batch.values[0]
            
            # add a if fval is WT or z otherwise
            # this is to ensure sleuth does
            # the regression as WT --> MT
            # but since sleuth only works alphabetically
            # simply stating WT --> MT doesn't work
            if fval == 'wt':
                fval = 'a' + fval
            else:
                fval = 'z' + fval
            
            if batch:
                line = name + ' ' + fval + ' ' + batchvar + '\n'
            else:
                line = name + ' ' + fval + '\n'
            f.write(line)
        

# Now make the matrix for each combination
# I made this separately from the above if loop
# because R is stupid and wants the files in this
# folder to be in the same order as they are 
# listed in the matrix design file....
for comb in combs:
    current = sleuth_loc + comb[0]+'_'+comb[1] + '/'
    
    # write a file called rna_seq_info for each combination
    matrix_design('rna_seq_info.txt', 'genotype', genmap,
                  comb[0], comb[1], current, batch=False)







