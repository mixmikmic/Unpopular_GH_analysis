get_ipython().run_line_magic('run', 'imports_mac.ipynb')
get_ipython().run_line_magic('run', '_shared_setup.ipynb')

get_ipython().system('rsync -avL {DATA_DIR} malsrv2:{os.path.dirname(DATA_DIR)}')

