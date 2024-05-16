from IPython.display import Javascript,display
from corticalmapping.ipython_lizard.html_widgets import raw_code_toggle
raw_code_toggle()
display(Javascript("""var nb = IPython.notebook;
                      //var is_code_cell = (nb.get_selected_cell().cell_type == 'code')
                      //var curr_idx = (nb.get_selected_index() == 3);
                      nb.select(3);
                      nb.execute_cell();
                      """))

from IPython.display import Javascript
from corticalmapping.ipython_lizard.ipython_filedialog import IPythonTkinterFileDialog
initial_dir = r"C:"

tkinter_file_dialog = IPythonTkinterFileDialog(initial_dir)
tkinter_file_dialog.execute_below = True
tkinter_file_dialog.show()

import os
from PyQt4 import QtGui,QtCore
import matplotlib.pyplot as plt
import matplotlib as mpl
from warnings import warn
#mpl.rcParams['figure.figsize'] = 10, 10
from corticalmapping import ipython_lizard
from corticalmapping.ipython_lizard.wrapped_retinotopic_mapping import WrappedRetinotopicMapping
from corticalmapping.ipython_lizard.patchplot_ipywidgets import PatchPlotWidgets
from corticalmapping.ipython_lizard.html_widgets import getSignMapWidget,getRawPatchMapWidget,getRawPatchesWidget,                                                         splitPatchesWidget,mergePatchesWidget,getEccentricityMapWidget,                                                         saveFinalResultWidget,submitAndRunBelowButton

get_ipython().magic('matplotlib inline')
#%load_ext autoreload
#%autoreload 2

pkl_path = tkinter_file_dialog.file_path

TEST_PKL_IDX = 0
TEST_PATH = ipython_lizard.TEST_PKLS[TEST_PKL_IDX] #there are like 6  different test pkls in this iterable
current_dir = os.getcwd()
adj_pkl_paths = [os.path.join(current_dir,f) for f in os.listdir(current_dir) if f.endswith("pkl")]
if adj_pkl_paths:
    adj_pkl_path = adj_pkl_paths[0]
else:
    adj_pkl_path = None
pkls = zip(["MANUAL","ADJACENT","TEST"],[pkl_path,adj_pkl_path,TEST_PATH])
for p_type,pkl in pkls:
    try:
        trial = WrappedRetinotopicMapping.load_from_pkl(pkl)
        print "Successfully loaded from: {0}, {1}".format(p_type,pkl)
        __pkl_path = pkl
        break
    except Exception as e:
        #warn(str(e))
        warn("Failed to load from: {0}, {1}".format(p_type,pkl))

_TEST_PKL_USED = (p_type == "TEST")

phaseMapFilterSigma = 1.0
signMapFilterSigma = 9.0

getSignMapWidget(trial,
                 phaseMapFilterSigmaDefault=phaseMapFilterSigma,
                 signMapFilterSigmaDefault=signMapFilterSigma,
)
submitAndRunBelowButton()

signMapThr = 0.35
openIter = 3
closeIter = 3


getRawPatchMapWidget(trial,
                     signMapThrDefault=signMapThr,
                     openIterDefault=openIter,
                     closeIterDefault=closeIter,
)
submitAndRunBelowButton()

dilationIter = 15
borderWidth = 1
smallPatchThr = 100

getRawPatchesWidget(trial,
                   dilationIterDefault=dilationIter,
                   borderWidthDefault=borderWidth,
                   smallPatchThrDefault=smallPatchThr,
)
submitAndRunBelowButton()

trial.getDeterminantMap()

eccMapFilterSigma = 10.0

getEccentricityMapWidget(trial,eccMapFilterSigmaDefault=eccMapFilterSigma)
submitAndRunBelowButton()

visualSpacePixelSize = 0.5
visualSpaceCloseIter = 15
splitLocalMinCutStep = 5.0
splitOverlapThr = 1.2

splitPatchesWidget(trial,
                   visualSpacePixelSizeDefault=visualSpacePixelSize,
                   visualSpaceCloseIterDefault=visualSpaceCloseIter,
                   splitLocalMinCutStepDefault=splitLocalMinCutStep,
                   splitOverlapThrDefault=splitOverlapThr
)
submitAndRunBelowButton()

mergeOverlapThr = 0.1

mergePatchesWidget(trial,mergeOverlapThrDefault=mergeOverlapThr)
submitAndRunBelowButton()

patchplot_widgets = PatchPlotWidgets(trial,{},[],figsize=(5,5))
patchplot_widgets.plot_reference_img()

rename_patches_dict = dict(trial.finalPatches)

DESIRED_PATCH_NAMES = ['A','AL','AM','LI','LLA','LM','M','MMA','MMP','P','PM','POR','RL','RLL','RS','S1','V1']

for patch in rename_patches_dict.keys(): #replace 'patch01' with 01, etc
    rename_patches_dict[patch.replace("patch","")] = rename_patches_dict.pop(patch)
patchplot_widgets = PatchPlotWidgets(trial,rename_patches_dict,DESIRED_PATCH_NAMES,figsize=(12,6))

patchplot_widgets.show()
submitAndRunBelowButton()

rename_patches_dict = patchplot_widgets.patches_dict
finalPatchBorder_figure = trial.plotFinalPatchBorders(rename_patches_dict,borderWidth=4)
#trial.params["finalPatchesMarked"] = rename_patches_dict
#trial.finalPatchesMarked = rename_patches_dict

pkl_save_path = None
#saveTrialDictPkl(trial,pkl_save_path)
saveFinalResultWidget(trial,finalPatchBorder_figure,__pkl_path,pkl_save_path,avoid_overwrite=(not _TEST_PKL_USED))

