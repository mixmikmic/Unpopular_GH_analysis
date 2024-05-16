import os
import pandas as pd

# params
directory = '../sleuth/'
batch = False
# sequences:
analysis = next(os.walk(directory))[1]

def sleuth_analysis(directory, genovar, batch=False):
    """
    A function to write the differential_expression_analyzer batch command.
    """
    if not batch:
        heart = 'Rscript diff_exp_analyzer.R -d {0} --genovar {1}'.format(directory, genovar)
    else:
        heart = 'Rscript diff_exp_analyzer.R -d {0} --genovar {1} --batch'.format(directory, genovar)
    return heart

def walk_sleuth_directories(directory, batch=False):
    """
    Given a directory, walk through it,
    find all the rna-seq repository folders
    and generate kallisto commands
    """
    sleuth = ''
    #directory contains all the projects, walk through it:
    current, dirs, files = next(os.walk(directory))
    for d in dirs:
        # genovar always begins with a z:
        genovar = 'z' + d[-1:]
        message = '# Sleuth analysis command for {0}\n'.format(d)
        command = sleuth_analysis(d, genovar, batch) +'\n'
        sleuth += message
        sleuth += command
    return sleuth

with open(directory + 'sleuth_commands.sh', 'w') as f:
    f.write('#!/bin/bash\n')
    f.write('# Bash commands for diff. expression analysis using Sleuth.\n')
    sleuth_command = walk_sleuth_directories(directory, batch)
    f.write(sleuth_command)
#     print(sleuth_command)





