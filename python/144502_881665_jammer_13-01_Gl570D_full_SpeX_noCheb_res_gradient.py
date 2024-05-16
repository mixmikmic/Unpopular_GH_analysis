import pandas as pd
from matplotlib.ticker import MaxNLocator

label = [r"$T_{\mathrm{eff}}$", r"$\log{g}$",r"$v_z$", r"$v\sin{i}$", r"$\log{\Omega}$", 
         r"sigAmp", r"logAmp", r"$l$"] 

ws = np.load("../sf/Gl570D_resg/output/marley_grid/run01/temp_emcee_chain.npy")

burned = ws[:, :1399,:]
xs, ys, zs = burned.shape
fc = burned.reshape(xs*ys, zs)
nx, ny = fc.shape

burned.shape

fig, axes = plt.subplots(8, 1, sharex=True, figsize=(8, 9))
for i in range(0, 8, 1):
    axes[i].plot(burned[:, :, i].T, color="k", alpha=0.2)
    axes[i].yaxis.set_major_locator(MaxNLocator(5))
    axes[i].set_ylabel(label[i])

axes[7].set_xlabel("step number")
fig.tight_layout(h_pad=0.0)

