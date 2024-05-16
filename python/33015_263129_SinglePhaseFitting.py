get_ipython().magic('matplotlib inline')

TDB_STR = """
$=============================================================================
$
$  Al-Mg-Zn System, almgzn.tdb
$  Parameterized by R. Otis from an Al-Cu-Mg-Zn database, 04-13-2015
$  Last updated: 01-07-1997 by H. Liang
$                                             
$  Based on Al-Mg-Zn[96Liang]
$
$=============================================================================
$
 ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!
 ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8322E+01!
 ELEMENT MG   HCP_A3                    2.4305E+01  4.9980E+03  3.2671E+01!
 ELEMENT ZN   HCP_A3                    6.5380E+01  5.6567E+03  4.1631E+01!

 
 FUNCTION UN_ASS     2.98140E+02 0.0; 3.00000E+02  N !
 FUNCTION GHSERAL    2.98130E+02  -7976.15+137.071542*T-24.3671976*T*LN(T)
     -.001884662*T**2-8.77664E-07*T**3+74092*T**(-1);  7.00000E+02  Y
      -11276.24+223.02695*T-38.5844296*T*LN(T)+.018531982*T**2
     -5.764227E-06*T**3+74092*T**(-1);  9.33600E+02  Y
      -11277.683+188.661987*T-31.748192*T*LN(T)-1.234264E+28*T**(-9);
     2.90000E+03  N !
 FUNCTION GHSERMG    2.98130E+02  -8367.34+143.677875*T-26.1849782*T*LN(T)
     +4.858E-04*T**2-1.393669E-06*T**3+78950*T**(-1);  9.23000E+02  Y
      -14130.185+204.718543*T-34.3088*T*LN(T)+1.038192E+28*T**(-9);
     3.00000E+03  N !
 FUNCTION GHSERZN    2.98140E+02  -7285.787+118.470069*T-23.701314*T*LN(T)
   -.001712034*T**2-1.264963E-06*T**3;  6.92680E+02  Y
   -11070.559+172.34566*T-31.38*T*LN(T)+4.70514E+26*T**(-9);
     1.70000E+03  N !
$
 FUNCTION GALLIQ     2.98130E+02  +11005.553-11.840873*T+7.9401E-20*T**7
     +GHSERAL#;  9.33600E+02  Y
      +10481.974-11.252014*T+1.234264E+28*T**(-9)+GHSERAL#;  
     2.90000E+03  N !
 FUNCTION GMGLIQ     2.98130E+02  +8202.24-8.83693*T-8.01759E-20*T**7
     +GHSERMG#;  9.23000E+02  Y
      +8690.32-9.39216*T-1.03819E+28*T**(-9)+GHSERMG#;  3.00000E+03  N !
 FUNCTION GZNLIQ     2.70000E+02  -128.517+108.176926*T-23.701314*T*LN(T)
     -.001712034*T**2-1.264963E-06*T**3-3.58652E-19*T**7;  6.92730E+02  Y
      -3620.474+161.608677*T-31.38*T*LN(T);  2.90000E+03  N !
$
 FUNCTION GBCCAL  2.98150E+02  +10083-4.813*T+GHSERAL#; 6.00000E+03  N !
 FUNCTION GBCCZN  2.98150E+02  +2886.96-2.5104*T+GHSERZN#;
		  6.00000E+03  N !
 FUNCTION GFCCZN  2.98150E+02  +2969.82-1.56968*T+GHSERZN#;
		  6.00000E+03  N !
 FUNCTION GFCCMG  2.98150E+02  2600-0.9*T+GHSERMG#;
		  6.00000E+03 N !

 TYPE_DEFINITION % SEQ *!
 DEFINE_SYSTEM_DEFAULT SPECIE 2 !
 DEFAULT_COMMAND DEF_SYS_ELEMENT VA !
 
 PHASE LIQUID:L %  1  1.0  !
    CONSTITUENT LIQUID:L :AL,MG,ZN :  !
 
   PARAMETER G(LIQUID,AL;0)  2.98130E+02  +GALLIQ#;
			     3.00000E+03  N REF: 0 !
   PARAMETER G(LIQUID,MG;0)  2.98130E+02  +GMGLIQ#;  
			     3.00000E+03  N REF: 0 !
   PARAMETER G(LIQUID,ZN;0)  2.98130E+02  +GZNLIQ#;  
			     3.00000E+03  N REF: 0 !
   PARAMETER G(LIQUID,AL,MG;0)  2.98150E+02  LIQALMG0A+LIQALMG0B*T;   
				6.00000E+03  N REF: 0 !
   PARAMETER G(LIQUID,AL,ZN;0)  2.98150E+02  LIQALZN0A+LIQALZN0B*T;
				6.00000E+03   N REF: 3 !
   PARAMETER G(LIQUID,AL,ZN;1)  2.98150E+02  LIQALZN1A+LIQALZN1B*T;
				6.00000E+03   N REF: 3 !                      
   PARAMETER G(LIQUID,MG,ZN;0) 298.15  LIQMGZN0A+LIQMGZN0B*T+LIQMGZN0C*T*LN(T); 
			       6000.0   N REF: 4 ! 
   PARAMETER G(LIQUID,MG,ZN;1) 298.15  LIQMGZN1A+LIQMGZN1B*T; 
			       6000.0 N REF: 4 ! 
   PARAMETER G(LIQUID,MG,ZN;2) 298.15  LIQMGZN2A;   
			       6000.0   N REF:   4 ! 
$
   PARAMETER G(LIQUID,AL,MG,ZN;0) 298.15 LIQALMGZN0A;  
				  6000.0  N REF:  0 !
   PARAMETER G(LIQUID,AL,MG,ZN;1) 298.15 LIQALMGZN1A; 
				  6000.0 N REF:  0 !
   PARAMETER G(LIQUID,AL,MG,ZN;2) 298.15 LIQALMGZN2A;  
				  6000.0  N REF:  0 !

$----------------------------------------------------------------------
 TYPE_DEFINITION ' GES A_P_D FCC_A1 MAGNETIC  -3.0    2.80000E-01 !

 PHASE FCC_A1  %'  2 1   1 !
    CONSTITUENT FCC_A1  :AL%,MG,ZN : VA% :  !
 
   PARAMETER G(FCC_A1,AL:VA;0)  2.98130E+02  +GHSERAL#;  
				2.90000E+03  N REF:  0 !
   PARAMETER G(FCC_A1,MG:VA;0)  2.98130E+02  +2600-.9*T+GHSERMG#;
				3.00000E+03  N REF: 0 !
   PARAMETER G(FCC_A1,ZN:VA;0)  2.98150E+02  +2969.82-1.56968*T+GHSERZN#;
				6.00000E+03   N REF: 1 !
   PARAMETER G(FCC_A1,AL,MG:VA;0)  2.98150E+02  FCCALMG0A+FCCALMG0B*T;
				   6.00000E+03   N REF: 0 !
   PARAMETER G(FCC_A1,AL,MG:VA;1)  2.98150E+02  0+0*T;
				   6.00000E+03   N REF: 0 !
   PARAMETER G(FCC_A1,AL,ZN:VA;0)  2.98150E+02  FCCALZN0A+FCCALZN0B*T;
				   6.00000E+03   N REF: 3 !
   PARAMETER G(FCC_A1,AL,ZN:VA;1)  2.98150E+02  FCCALZN1A+FCCALZN1B*T;
				   6.00000E+03   N REF: 3 !
   PARAMETER G(FCC_A1,AL,ZN:VA;2)  2.98150E+02  FCCALZN2A+FCCALZN2B*T;
				   6.00000E+03   N REF: 3 !
   PARAMETER G(FCC_A1,MG,ZN:VA;0)  2.98150E+02  FCCMGZN0A; 
				   6000.0        N REF: 4 !
   PARAMETER G(FCC_A1,AL,MG,ZN:VA;0) 298.15 FCCALMGZN0A;  
				     6000.0  N REF:  0 !
   
$----------------------------------------------------------------------

 TYPE_DEFINITION ( GES A_P_D HCP_A3 MAGNETIC  -3.0    2.80000E-01 !
 
 PHASE HCP_A3  %(  2 1   .5 !
    CONSTITUENT HCP_A3  :AL,MG%,ZN : VA% :  !
 
   PARAMETER G(HCP_A3,AL:VA;0)  2.98130E+02  +5481-1.8*T+GHSERAL#;
				2.90000E+03  N REF: 0 !
   PARAMETER G(HCP_A3,MG:VA;0)  2.98130E+02  +GHSERMG#;  
				3.00000E+03  N REF:  0 !
   PARAMETER G(HCP_A3,ZN:VA;0)  2.98150E+02  +GHSERZN#;
				6.00000E+03   N REF: 1 !
   PARAMETER G(HCP_A3,AL,MG:VA;0)  2.98150E+02  HCPALMG0A+HCPALMG0B*T;
				   6.00000E+03   N REF: 0 !
   PARAMETER G(HCP_A3,AL,MG:VA;1)  2.98150E+02  HCPALMG1A;   
				   6.00000E+03   N  REF: 0 !
   PARAMETER G(HCP_A3,AL,ZN:VA;0)  2.98150E+02  HCPALZN0A;
				   6.00000E+03   N REF: 3 !
   PARAMETER G(HCP_A3,MG,ZN:VA;0) 298.15  HCPMGZN0A+HCPMGZN0B*T; 
				  6000.0 N REF: 4 !
   PARAMETER G(HCP_A3,MG,ZN:VA;1) 298.15  HCPMGZN1A+HCPMGZN1B*T; 
				  6000.0 N REF: 4 !

$-------------------------------------------------------------------

  PHASE SIGMA  %  2 .66667    .33333 !
    CONSTITUENT SIGMA  :AL,ZN : MG :  !
 
   PARAMETER G(SIGMA,AL:MG;0)  2.98150E+02  +20133.73+6.3946*T
			       +.66667*GALLIQ#+.33333*GMGLIQ#;  
			       3.00000E+03  N REF: 0 !
   PARAMETER G(SIGMA,ZN:MG;0)  2.98150E+02  -19389.65+13.644*T
			       +.66667*GZNLIQ#+.33333*GMGLIQ#; 
			       3.00000E+03  N REF: 0 ! 
   PARAMETER G(SIGMA,AL,ZN:MG;0) 2.98150E+02  SIGALZN0A;  
				 3.00000E+03  N  REF: 0 !
   PARAMETER G(SIGMA,AL,ZN:MG;1) 2.98150E+02  SIGALZN1A;  
				 3.00000E+03  N  REF: 0 !

$----------------------------------------------------------------

 PHASE T  %  2 .605   .395 !
    CONSTITUENT T  :AL,ZN : MG :  !
 
   PARAMETER G(T,AL:MG;0)  2.98150E+02  -10910.836+8.71*T
			   +.605*GALLIQ#+.395*GMGLIQ#;  
			   3.00000E+03  N REF: 0 !
   PARAMETER G(T,ZN:MG;0)  2.98150E+02  -15733.501+12.6746*T
			   +.605*GZNLIQ#+.395*GMGLIQ#;  
			   3.00000E+03  N REF: 0 !
   PARAMETER G(T,AL,ZN:MG;0)  2.98150E+02  TALZN0A+TALZN0B*T;  
			      3.00000E+03  N  REF: 0 !
   PARAMETER G(T,AL,ZN:MG;1)  2.98150E+02  TALZN1A;  
			      3.00000E+03  N REF:   0 !
$----------------------------------------------------------------- 

 PHASE EPS  %  2 1   1 !
    CONSTITUENT EPS  :AL,MG,ZN : VA :  !

   PARAMETER G(EPS,AL:VA;0)  2.98150E+02   5481-1.8*T+GHSERAL#;
			     6.00000E+03 N REF: 0 !
   PARAMETER G(EPS,MG:VA;0)  2.98150E+02  +10+GFCCMG#;     
			     6.00000E+03 N REF: 0 !
   PARAMETER G(EPS,ZN:VA;0)  2.98150E+02  +GFCCZN#;
			     6.00000E+03   N REF: 0 !
   PARAMETER G(EPS,AL,ZN:VA;0)  2.98150E+02   EPSALZN0A;
				6.00000E+03   N REF: 0 !
                
$-----------------------------------------------------------------              

 TYPE_DEFINITION & GES A_P_D BCC_A2 MAGNETIC  -1.0    4.00000E-01 !
 
  PHASE BCC_A2  %  2 1   3 !
    CONSTITUENT BCC_A2  :AL,ZN : VA% :  !

   PARAMETER G(BCC_A2,AL:VA;0)  2.98150E+02  +10083-4.813*T+GHSERAL#;
				6.00000E+03   N REF: 1 !
   PARAMETER G(BCC_A2,ZN:VA;0)  2.98150E+02  +2886.96-2.5104*T+GHSERZN#;
				6.00000E+03   N REF: 1 !
   PARAMETER G(BCC_A2,AL,ZN:VA;0)  2.98150E+02   BCCALZN0A;
				   6.00000E+03  N REF: 0 !

$------------------------------------------------------------------
 
PHASE GAMMA  %  2 1   1 !
    CONSTITUENT GAMMA  :AL,ZN : VA :  !

   PARAMETER G(GAMMA,AL:VA;0)  2.98150E+02  +GHSERAL#+10.0;   
			       6.00000E+03  N  REF: 0 !
   PARAMETER G(GAMMA,ZN:VA;0)  2.98150E+02  +GHSERZN#+10.0;   
			       6.00000E+03  N  REF: 0 !
   PARAMETER G(GAMMA,AL,ZN:VA;0)  2.98150E+02  GAMALZN0A;
				  6.00000E+03  N REF: 0 !
$---------------------------------------------------------------------

 PHASE PHI  %  3  2  5  2 !
    CONSTITUENT PHI  :AL : MG : ZN :  !

  PARAMETER G(PHI,AL:MG:ZN;0)    298.15  -169985.46+136.8*T
	+2*GALLIQ#+5*GMGLIQ#+2*GZNLIQ#;  6000.0 N REF: 0 !


$====================================================================== 
$              Binary Intermetallic Phases
$======================================================================
 
$---------------------------------------------------------------------

 PHASE ALMG_BETA  %  2 .615   .385 !
    CONSTITUENT ALMG_BETA  :AL : MG :  !
 
   PARAMETER G(ALMG_BETA,AL:MG;0)  2.98150E+02  -1451.1-1.907*T
	      +.615*GHSERAL#+.385*GHSERMG#;   6.00000E+03   N REF: 0 !
 
$---------------------------------------------------------------------
 
 PHASE ALMG_EPSILON  %  2 .56   .44 !
    CONSTITUENT ALMG_EPSILON  :AL : MG :  !
   PARAMETER G(ALMG_EPSILON,AL:MG;0)  2.98150E+02  ALMG_EPSILONALMG0A+ALMG_EPSILONALMG0B*T
	       +.56*GHSERAL#+.44*GHSERMG#;   6.00000E+03   N REF: 0 !

$---------------------------------------------------------------------- 
 
 PHASE ALMG_GAMMA  %  3 .4483   .1379   .4138 !
    CONSTITUENT ALMG_GAMMA  :MG : AL,MG : AL,MG :  !

   PARAMETER G(ALMG_GAMMA,MG:AL:AL;0)  2.98150E+02  ALMG_GAMMAMGALAL0A+ALMG_GAMMAMGALAL0B*T
	   +.5517*GHSERAL#+.4483*GHSERMG#;  6.00000E+03   N REF: 0 !
   PARAMETER G(ALMG_GAMMA,MG:MG:AL;0)  2.98150E+02  ALMG_GAMMAMGMGAL0A+ALMG_GAMMAMGMGAL0B*T
	   +.4138*GHSERAL#+.5862*GHSERMG#;   6.00000E+03   N REF: 0 !
   PARAMETER G(ALMG_GAMMA,MG:AL:MG;0)  2.98150E+02  ALMG_GAMMAMGALMG0A+ALMG_GAMMAMGALMG0B*T
	   +.1379*GHSERAL#+.8621*GHSERMG#;   6.00000E+03   N REF: 0 !
   PARAMETER G(ALMG_GAMMA,MG:MG:MG;0)  2.98150E+02  ALMG_GAMMAMGMGMG0A+GHSERMG#;
	   6.00000E+03   N REF: 0 !

$---------------------------------------------------------------------- 
$ 
 PHASE ALMG_ZETA  %  2 .525   .475 !
    CONSTITUENT ALMG_ZETA  :AL : MG :  !
 
   PARAMETER G(ALMG_ZETA,AL:MG;0)  2.98150E+02  -837.8-3.163*T
	    +.525*GHSERAL#+.475*GHSERMG#;   6.00000E+03   N REF: 0 !
 
$----------------------------------------------------------------------
$
 PHASE MG2ZN11  %  2 .153846   .846154 !
    CONSTITUENT MG2ZN11  :MG : ZN :  !
 
  PARAMETER G(MG2ZN11,MG:ZN;0)  298.15  -5823.05+1.94323*T
  +0.153846154*GHSERMG#+0.846153846*GHSERZN#;   6000.0   N REF: 4 !
$
$-----------------------------------------------------------------------
$
 PHASE MG2ZN3  %  2 .4   .6 !
    CONSTITUENT MG2ZN3  :MG : ZN :  !
 
  PARAMETER G(MG2ZN3,MG:ZN;0)  298.15  -11014.5+3.67151*T+0.4*GHSERMG#
			       +0.6*GHSERZN#;   6000.0   N REF: 4 !
$
$-----------------------------------------------------------------------
$
 PHASE MG7ZN3  %  2 .71831   .28169 !
    CONSTITUENT MG7ZN3  :MG : ZN :  !
 
  PARAMETER G(MG7ZN3,MG:ZN;0)  298.15  -4814.11+T+0.71831*GHSERMG#
			       +0.28169*GHSERZN#;   6000.0   N REF: 4 !
$
$-----------------------------------------------------------------------
$
 PHASE MGZN  %  2 .48   .52 !
    CONSTITUENT MGZN  :MG : ZN :  !
 
  PARAMETER G(MGZN,MG:ZN;0)  298.15  -9590.44+3.19681*T+0.48*GHSERMG#
			     +0.52*GHSERZN#;   6000.0   N REF: 4 !
$

$===========================================================================
"""

from pycalphad import Database, Model, binplot
import pycalphad.variables as v
import matplotlib
matplotlib.style.use('ggplot')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sympy import Symbol
import numpy as np

dbf = Database(TDB_STR)

import pandas as pd
from pycalphad.residuals import fit_model

liang96_almg_parameters = {
    'FCCALMG0A':   1000,
    'FCCALMG0B':   10,
}
import lmfit
phase_eq = pd.read_csv('almg_fcc_enthalpy.csv')
#phases = ['LIQUID', 'FCC_A1', 'HCP_A3', 'ALMG_EPSILON', 'ALMG_BETA', 'ALMG_GAMMA']
phases = ['FCC_A1']
comps = ['AL', 'MG', 'VA']
start_models = {name: Model(dbf, comps, name, parameters=liang96_almg_parameters) for name in phases}
get_ipython().magic('time result, mi = fit_model(liang96_almg_parameters, phase_eq, dbf, comps, phases, maxfev=50)')
fit_models = {name: Model(dbf, comps, name, parameters=result) for name in phases}

print(lmfit.report_fit(mi.params))
print(lmfit.conf_interval(mi))

