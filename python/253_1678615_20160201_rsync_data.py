get_ipython().run_line_magic('run', 'imports_mac.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')

get_ipython().system('rsync -avL /nfs/team112_internal/production/release_build/Pf3K/pilot_5_0  malsrv2:/nfs/team112_internal/production/release_build/Pf3K/')

get_ipython().system('rsync -avL {DATA_DIR} malsrv2:{os.path.dirname(DATA_DIR)}')



