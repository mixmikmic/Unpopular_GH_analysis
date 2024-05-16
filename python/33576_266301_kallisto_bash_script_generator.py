import os
import pandas as pd
# params
directory = '../input/rawseq'
length = 180
sigma = 60
btstrp = 200
thrds = 6

# sequences:
seqs = next(os.walk(directory))[1]

# params
directory = '../input/rawseq'
length = 180
sigma = 60
btstrp = 200
thrds = 6

# sequences:
seqs = next(os.walk(directory))[1]

def explicit_kallisto(directory, files, res_dir):
    """
    TODO: Make a function that allows you to systematically 
    set up each parameter for each sequencing run individually.
    """
    
    if type(directory) is not str:
        raise ValueError('directory must be a str')
    if type(files) is not list:
        raise ValueError('files must be a list')
    
    print('This sequence file contains a Kallisto_Info file            and cannot be processed at the moment.')
    return '# {0} could not be processed'.format(res_dir), ''
    
def implicit_kallisto(directory, files, res_dir):
    """
    A function to write a Kallisto command with standard parameter
    setup
    """
    if type(directory) is not str:
        raise ValueError('directory must be a str')
    if type(files) is not list:
        raise ValueError('files must be a list')

    # parts of each kallisto statement
    
    # information
    info = '# kallisto command for {0}'.format(directory)
    # transcript file location:
    k_head = 'kallisto quant -i input/transcripts.idx -o '
    
    # output file location
    k_output = 'input/kallisto_all/' + res_dir + '/kallisto '
    # parameter info:
    k_params = '--single -s {0} -l {1} -b {2} -t {3} --bias --fusion'.format(sigma, length, btstrp, thrds)
    
    # what files to use:
    k_files = ''    
    # go through each file and add it to the command
    # unless it's a SampleSheet.csv file, in which
    # case you should ignore it. 
    for y in files:
        if y != 'SampleSheet.csv':
            if directory[:3] == '../':
                d = directory[3:]
            else:
                d = directory[:]
            k_files += ' '+ d + '/' + y
    # all together now:
    kallisto = k_head + k_output + k_params + k_files +';'
    return info, kallisto

def walk_seq_directories(directory):
    """
    Given a directory, walk through it,
    find all the rna-seq repository folders
    and generate kallisto commands
    """
    kallisto = ''
    #directory contains all the projects, walk through it:
    for x in os.walk(directory):
        # first directory is always parent
        # if it's not the parent, move forward:
        if x[0] != directory:
            # cut the head off and get the project name:
            res_dir = x[0][len(directory)+1:]
            
            # if this project has attributes explicitly written in
            # use those parameter specs:
            if 'Kallisto_Info.csv' in x[2]:
                info, command = explicit_kallisto(x[0], x[2], res_dir)
                continue
            
            # otherwise, best guesses:
            info, command = implicit_kallisto(x[0], x[2], res_dir)
            kallisto += info + '\n' + command + '\n'
            
            if not os.path.exists('../input/kallisto_all/' + res_dir):
                os.makedirs('../input/kallisto_all/' + res_dir)
    return kallisto

with open('../kallisto_commands.sh', 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('# make transcript index\n')
    f.write('kallisto index -i input/transcripts.idx input/c_elegans_WBcel235.rel79.cdna.all.fa;\n')
    kallisto = walk_seq_directories(directory)
    f.write(kallisto)





