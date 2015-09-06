'''
Author: Giggle Leo
Date : 8 September 2014
Description : physics library
'''
from numpy import *
from scipy.linalg import *
import gmpy2

############################ DEFINITIONS ##############################
# pauli spin
sx = array([[0, 1],[ 1, 0]])
sy = array([[0, -1j],[1j, 0]])
sz = array([[1, 0],[0, -1]])

############################ FUNCTIONS ##############################
def eigh_pauliv_npy(a0,a1,a2,a3):
    '''
    eigen values for pauli vectors - numpy version.

    a0/a1/a2/a3:
        pauli components.
    '''
    e0=sqrt(a1**2+a2**2+a3**2)
    evals=array([a0-e0,a0+e0])
    if abs(a1+a2*1j)<1e-50:
        evecs=array([[0,1],[1,0j]])
    else:
        evecs=array([[(a3-e0)/(a1+a2*1j),1],[(a3+e0)/(a1+a2*1j),1]])
    for i in xrange(2):
        evecs[i]=evecs[i]/norm(evecs[i])
    return evals,evecs.T

def ode_ronge_kutta(func,y0,tlist,**kwargs):
    '''
    Integrate use Ronge Kutta method.

    func:
        the function of (x,y).
    y0:
        the starting y.
    tlist:
        a list of t.
    '''
    y0=y0;t0=tlist[0]
    tf=ode(func)
    tf.set_integrator('dopri5',**kwargs)
    tf.set_initial_value(y0,t0)
    yl=[y0]
    for t in tlist[1:]:
        nt=t
        tf.integrate(nt,step=nt-tf.t)
        yl.append(tf.y)
    return array(yl)

def s2vec(s):
    '''
    Transform a spin to a 4 dimensional vector, corresponding to s0,sx,sy,sz component.

    s: 
        the spin.
    '''
    res=array([trace(s),trace(dot(sx,s)),trace(dot(sy,s)),trace(dot(sz,s))])/2
    return res

def vec2s(n,unit=False):
    '''
    Transform a vector of length 3 or 4 to a pauli matrix.

    n: 
        a 1-D array of length 3 or 4 to specify the `direction` of spin.
    '''
    if unit:
        n=n/norm(n)
    if len(n)<=3:
        res=zeros([2,2],dtype='complex128')
        for i in xrange(len(n)):
            res+=s[i+1]*n[i]
        return res
    elif len(n)==4:
        return identity(2)*n[0]+sx*n[1]+sy*n[2]+sz*n[3]
    else:
        raise Exception('length of vector %s too large.'%len(n))

def H2G(h,w,tp='r',geta=1e-2,sigma=None):
    '''
    Get G(k) from H(k).

    h: 
        an array of hamiltonian.
    w:
        the energy(frequency).
    tp:
        the type of Green's function.
        'r': retarded Green's function.(default)
        'a': advanced Green's function.
        'matsu': finite temperature Green's function.
    geta:
        smearing factor. default is 1e-2.
    '''
    if tp=='r':
        z=w+1j*geta
    elif tp=='a':
        z=w-1j*geta
    elif tp=='matsu':
        z=1j*w
    if sigma!=None:
        h=h+sigma
    if ndim(h)>0:
        return inv(z*identity(len(h))-h)
    else:
        return 1./(z-h)

def qr2(A):
    '''
    analytically, get the QR decomposition of a 2 dimensional hermion matrix.

    A:
        the matrix.
    '''
    ndim=A.shape[1]
    Q=zeros([A.shape[0],0])
    csqrt=vectorize(gmpy2.sqrt)
    for i in xrange(ndim):
        ui=A[:,i:i+1]
        if i>0:
            ui=ui-dot(Q,dot(mpconj(Q.T),ui))
        ui=ui/csqrt(dot(mpconj(ui.T),ui))
        Q=concatenate([Q,ui],axis=-1)
    R=dot(mpconj(Q.T),A)
    return Q,R

def mpconj(A):
    '''
    get the conjugate of matrix A
    
    A:
        the input matrix.
    '''
    N1,N2=A.shape
    B=ndarray(A.shape,dtype='O')
    for i in xrange(N1):
        for j in xrange(N2):
            data=A[i,j]
            B[i,j]=gmpy2.mpc(data.real,-data.imag)
    return B


