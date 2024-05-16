get_ipython().magic('matplotlib inline')
get_ipython().magic('load_ext signature')

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import helpers

mpl.style.use('ggplot')
logo = plt.imread('img/ramiro.org-branding.png')
ts = pd.Series(np.random.randn(1000), index=pd.date_range('1/1/2010', periods=1000)).cumsum()

title = 'Random Time Series Plot with Watermark'

ax = ts.plot(figsize=(14, 8))
ax.set_title(title, fontsize=20)
ax.figure.figimage(logo, 40, 40, alpha=.15, zorder=1)

plt.savefig('img/{}.png'.format(helpers.slug(title)), bbox_inches='tight')

title = 'Random Time Series Plot with Branding Image'

gs = gridspec.GridSpec(2, 1, height_ratios=[24,1])

ax1 = plt.subplot(gs[0])
ax1.set_title(title, size=20)
ax1.figure.set_figwidth(14)
ax1.figure.set_figheight(8)
ax1.plot(ts)

ax2 = plt.subplot(gs[1])
img = ax2.imshow(logo)
ax2.axis('off')

plt.savefig('img/{}.png'.format(helpers.slug(title)), bbox_inches='tight')

get_ipython().magic('signature')

