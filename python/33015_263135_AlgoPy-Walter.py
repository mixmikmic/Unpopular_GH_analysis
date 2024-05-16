import numpy as np
import algopy
import autograd.numpy as anp


def func(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, mod=anp):
    "Toy function (calibrated to take roughly the same time as real one)"
    return 8.3145 * x0 * (x1 * mod.log(x1) + x2 * mod.log(x2)) + 2000*x1*x2 + 100*x1*x2*(x1-x2) + 200*x2*x3*(x1-x2)**2 + 2000*x3*x4 + 100*x5*x6*(x1-x2) + 200*x7*x8*(x8-x9)**2 + 2400*x1*x2 + 120*x1*x2*(x1-x2) + 200*x1*x2*(x1-x2)**2 + 2000*x1*x2 + 100*x1*x2*(x1-x2) + 200*x1*x2*(x1-x2)**2 + 2000*x1*x2 + 160*x1*x2*(x1-x2) + 1000*x1*x2*(x1-x2)**2 + 2000*x1*x2 + 100*x1*x2*(x1-x2) + 500*x1*x2*(x1-x2)**2

# We repeat the same two values over and over to simulate lots of different input
C = 20000
N = 10
inp_arr = np.tile([[300,0.5,0.5, 300,0.5,0.5, 300,0.5,0.5, 1], [600, 0.4, 0.6, 600, 0.4, 0.6, 600, 0.4, 0.6, 1]], (int(C/2),1))
print(inp_arr.shape)

get_ipython().run_cell_magic('timeit', '', '\n# record computational graph\ncg = algopy.CGraph()\nfx = algopy.Function(np.ones(N))\nfy = func(fx[0], fx[1], fx[2], fx[3], fx[4], fx[5], fx[6], fx[7], fx[8], fx[9], mod=algopy)\ncg.independentFunctionList = [fx]\ncg.dependentFunctionList = [fy]')

get_ipython().run_cell_magic('timeit', '', '\n# compute obj and grad in reverse mode\nux = algopy.UTPM(np.zeros((1, C, N)))\nux.data[0, ...] = inp_arr\ncg.pushforward([ux])\nuy_bar = algopy.UTPM(np.ones((1,C)))\ncg.pullback([uy_bar])\n\nobj_cgraph = cg.dependentFunctionList[0].x.data[0]\ngrad_cgraph = cg.independentFunctionList[0].xbar.data[0,:,:]')

get_ipython().run_cell_magic('timeit', '', '\n# compute obj, grad, hess in combined forward+reverse mode\nux = algopy.UTPM(np.zeros((2, N*C, N)))\nfor i in range(N):\n    ux.data[0, i::N, :] = inp_arr\n    ux.data[1, i::N, i::N] = 1\n\ncg.pushforward([ux])\nuy_bar = algopy.UTPM(np.zeros((2,N*C)))\nuy_bar.data[0,:] = 1\ncg.pullback([uy_bar])\n\nobj_cgraph = cg.dependentFunctionList[0].x.data[0, ::N]\ngrad_cgraph = cg.independentFunctionList[0].xbar.data[0,::N]\nhess_cgraph = cg.independentFunctionList[0].xbar.data[1,:,:].reshape((C, N, N))')

def extract_hessian(N, y):
    "Vectorized version of extract_hessian"
    H = np.zeros((y.data.shape[1], N,N), dtype=y.data.dtype)
    for n in range(N):
        for m in range(n):
            a =  sum(range(n+1))
            b =  sum(range(m+1))
            k =  sum(range(n+2)) - m - 1
            #print 'k,a,b=', k,a,b
            if n!=m:
                tmp = (y.data[2, :, k] - y.data[2, :, a] - y.data[2, :, b])
                H[:, m,n]= H[:, n,m]= tmp
        a =  sum(range(n+1))
        H[:, n,n] = 2*y.data[2, :, a]
    return H

# generate directions
M = (N*(N+1))/2
L = (N*(N-1))/2
S = np.zeros((N,M))

s = 0
i = 0
for n in range(1,N+1):
    S[-n:, s:s+n] = np.eye(n)
    S[-n, s:s+n] = np.ones(n)
    s += n
    i += 1
#print(S)
S = S[::-1].T
#print(S)
x = algopy.UTPM(np.zeros((3, inp_arr.shape[0]) + S.shape))
x.data[0, :, :, :] = inp_arr[..., None, :]
x.data[1, :, :] = S

get_ipython().run_cell_magic('timeit', '', "y = func(*[x[..., i] for i in range(N)], mod=algopy)\nobj_algopy = y.data[0, :, 0]\ngrad_algopy = y.data[1, :, np.cumsum(np.arange(N), dtype=np.int)].T\nhess_algopy = extract_hessian(N, y)\n#print('OBJ SHAPE', obj_algopy.shape)\n#print('GRAD SHAPE', grad_algopy.shape)\n#print('HESS SHAPE', hess_algopy.shape)")

import autograd.numpy as anp
from autograd import elementwise_grad, jacobian
from itertools import chain

def elementwise_hess(fun, argnum=0):
    """
    From https://github.com/HIPS/autograd/issues/60
    """
    def sum_latter_dims(x):
        return anp.sum(x.reshape(x.shape[0], -1), 1)

    def sum_grad_output(*args, **kwargs):
        return sum_latter_dims(elementwise_grad(fun)(*args, **kwargs))
    return jacobian(sum_grad_output, argnum)


def build_functions():
    obj = func

    def argwrapper(args):
        return obj(*args)

    def grad_func(*args):
        inp = anp.array(anp.broadcast_arrays(*args))
        result = anp.atleast_2d(elementwise_grad(argwrapper)(inp))
        # Put 'gradient' axis at end
        axes = list(range(len(result.shape)))
        result = result.transpose(*chain(axes[1:], [axes[0]]))
        return result

    def hess_func(*args):
        result = anp.atleast_3d(elementwise_hess(argwrapper)(anp.array(anp.broadcast_arrays(*args))))
        # Put 'hessian' axes at end
        axes = list(range(len(result.shape)))
        result = result.transpose(*chain(axes[2:], axes[0:2]))
        return result

    return obj, grad_func, hess_func

autograd_obj, autograd_grad, autograd_hess = build_functions()

get_ipython().run_cell_magic('timeit', '', 'ag_o = autograd_obj(*[inp_arr[..., i] for i in range(inp_arr.shape[-1])])\nag_g = autograd_grad(*[inp_arr[..., i] for i in range(inp_arr.shape[-1])])\nag_h = autograd_hess(*[inp_arr[..., i] for i in range(inp_arr.shape[-1])])')

# Have to comment out '%%timeit' lines to actually run this
import numpy.testing

numpy.testing.assert_allclose(obj_algopy, ag_o)
numpy.testing.assert_allclose(grad_algopy, ag_g)
numpy.testing.assert_allclose(hess_algopy, ag_h)

numpy.testing.assert_allclose(obj_cgraph, ag_o)
numpy.testing.assert_allclose(grad_cgraph, ag_g)
numpy.testing.assert_allclose(hess_cgraph, ag_h)

print('equivalent')





