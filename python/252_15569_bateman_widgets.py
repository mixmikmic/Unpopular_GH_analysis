from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import clear_output, display, HTML
from scipy.integrate import odeint
from pylab import *

global lam

def dcdt(c,t):
    dfdt = np.zeros(4)
    dfdt[0] = c[0]* -lam[0]               - c[0]*lam[3]
    dfdt[1] = c[1]* -lam[1] + c[0]*lam[0] - c[1]*lam[3] 
    dfdt[2] = c[2]* -lam[2] + c[1]*lam[1] - c[2]*lam[3]
    dfdt[3] =                 c[2]*lam[2] - c[3]*lam[3]
    return dfdt

def on_button_clicked(b):
    print("Calculate button clicked.")
    global lam
    lam = array([float(L0.value),float(L1.value),                 float(L2.value),float(L3.value)])
    C0 = array([.68, .23, .06, 0.])
    t = linspace(0.0,100.,50)
    C = odeint(dcdt,C0,t)

    fig = plt.figure(figsize=(6,5))
    plot(t+float(DS.value),C[:,0],label='DDE')
    plot(t+float(DS.value),C[:,1],label='DDMU')
    plot(t+float(DS.value),C[:,2],label='DDNS')
    plot(t+float(DS.value),C[:,3],label='?')
    plt.legend()
    plt.ylabel('Inventory')
    
DS = widgets.TextWidget(description = r'Start year',value='1992')
L0 = widgets.TextWidget(description = r'DDE  -> DDMU',value='0.052')
L1 = widgets.TextWidget(description = r'DDMU -> DDNS',value='0.07')
L2 = widgets.TextWidget(description = r'DDNS ->  ?  ',value='0.161')
L3 = widgets.TextWidget(description = r'DDX  -> lost',value='0.00')
B  = widgets.ButtonWidget(description = r'Calculate!')

display(DS,L0,L1,L2,L3,B)
B.on_click(on_button_clicked)



