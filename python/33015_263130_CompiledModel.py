from pycalphad.core.compiled_model import CompiledModel
from pycalphad import calculate, Database, Model
from pycalphad.core.phase_rec import PhaseRecord
import pycalphad.variables as v
import numpy as np

dbf = Database('alcocr-sandbox.tdb')

cmpmdl = CompiledModel(dbf, ['AL', 'CO', 'CR', 'VA'], 'BCC_A2')
out = np.zeros(7)
get_ipython().magic('time cmpmdl.eval_energy_gradient(out, np.array([101325, 600, 0.3, 0.3, 0.4-1e-12, 1e-12, 1]), np.array([]))')
out

mdl = Model(dbf, ['AL', 'CO', 'CR', 'VA'], 'BCC_A2')
#mdl.models['idmix'] = 0
#mdl.models['ref'] = 0
#print(mdl.GM.diff(v.Y('BCC_A2', 0, 'VA')))
[mdl.GM.diff(x).subs({v.T: 600., v.P:101325., v.Y('BCC_A2', 0, 'AL'): 0.3, v.Y('BCC_A2', 0, 'CO'): 0.3,
                    v.Y('BCC_A2', 0, 'CR'): 0.4-1e-12, v.Y('BCC_A2', 0, 'VA'): 1e-12, v.Y('BCC_A2', 1, 'VA'): 1.})
 for x in [v.P, v.T, v.Y('BCC_A2', 0, 'AL'),v.Y('BCC_A2', 0, 'CO'),v.Y('BCC_A2', 0, 'CR'),v.Y('BCC_A2', 0, 'VA'), v.Y('BCC_A2', 1, 'VA')]]

get_ipython().magic("time res = calculate(dbf, ['AL', 'CO', 'CR', 'VA'], sorted(dbf.phases.keys()), T=2000, P=101325, output='GM', pdens=200, model=CompiledModel)")
res.GM[..., :20]

get_ipython().magic("time res2 = calculate(dbf, ['AL', 'CO', 'CR', 'VA'], sorted(dbf.phases.keys()), T=2000, P=101325, output='GM', pdens=200, model=Model)")
res2.GM[..., :20]



