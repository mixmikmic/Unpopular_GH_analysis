import numpy as np

def q_overtop_EOT():
    """Principal wave overtopping formula (4.1)"""
    g=9.86
    a=1.
    b=1.
    Rc=1.
    Hm0 = 1.
    qND=np.sqrt(g*Hm0**3)   # dimensionless  discharge
    RcHm0 = Rc/Hm0          # relative freeboard
    q = qND*a*np.exp(-b*RcHm0)
    return q

def runup_EOT(Hm0=1.,Tm0=8.,tanalpha=.2,yb=1.,yf=1.,yB=1.):
    g = 9.86
    c1 = 1.65
    c2 = 4.
    c3 = 1.5
    L0 = g*Tm0**2./(2*np.pi)
    em0 = tanalpha/np.sqrt(Hm0/L0)
    Ru2Hm0 = np.max(c1*yb*yf*yB*em0,yb*yf*yB*(c2-c3/np.sqrt(em0)))
    print Ru2Hm0
    return Ru2


print q_overtop_EOT()
print runup_EOT()



