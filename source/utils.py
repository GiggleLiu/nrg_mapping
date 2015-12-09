'''
Physics utility library
'''

from numpy import *
from scipy.linalg import eigh,norm
from matplotlib import cm
from matplotlib.pyplot import *
import gmpy2

__all__=['sx','sy','sz','Gmat','eigh_pauliv_npy','plot_pauli_components','mpconj','qr2','H2G','s2vec','vec2s']

############################ DEFINITIONS ##############################
# pauli spin
sx = array([[0, 1],[ 1, 0]])
sy = array([[0, -1j],[1j, 0]])
sz = array([[1, 0],[0, -1]])
Gmat=array([kron(sz,sx),kron(identity(2),sy),kron(sx,sx),kron(sy,sx),kron(identity(2),sz)])

############################ FUNCTIONS ##############################
def eigh_pauliv_npy(a0,a1,a2,a3):
    '''
    eigen values for pauli vectors - numpy version.

    a0/a1/a2/a3:
        pauli components.
    *return*:
        (evals,evecs)
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

def s2vec(s):
    '''
    Transform a 2 x 2 matrix to a 4 dimensional vector, corresponding to s0,sx,sy,sz component.

    s: 
        the matrix.
    '''
    res=array([trace(s),trace(dot(sx,s)),trace(dot(sy,s)),trace(dot(sz,s))])/2
    return res

def vec2s(n):
    '''
    Transform a vector of length 3 or 4 to a pauli matrix.

    n: 
        a 1-D array of length 3 or 4 to specify the `direction` of spin.
    *return*:
        2 x 2 matrix.
    '''
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
    Get Green's function g from Hamiltonian h.

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
    sigma:
        additional self energy.
    *return*:
        a Green's function.
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
    analytically, get the QR decomposition of a matrix.

    A:
        the matrix.
    *return*:
        (Q,R), where QR=A
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
    Get the conjugate of matrix A(to avoid a bug of gmpy2.mpc.conj).
    
    Parameters
    -------------------
    A:
        The input matrix.

    Return
    --------------------
    2D array of dtype `mpc`, the conjugate matrix of A
    '''
    N1,N2=A.shape
    B=ndarray(A.shape,dtype='O')
    for i in xrange(N1):
        for j in xrange(N2):
            data=A[i,j]
            B[i,j]=gmpy2.mpc(data.real,-data.imag)
    return B

def plot_pauli_components(x,y,method='plot',ax=None,label=r'\sigma',**kwargs):
    '''
    Plot data by pauli components.

    Parameters
    -------------------
    x,y:
        Datas.
    ax:
        The axis to plot, will use gca() to get one if None.
    label:
        The legend of plots.
    method:
        `plot` or `scatter`
    kwargs:
        The key word arguments for plot/scatter.
    
    Return
    -------------------
    A list of plot instances.
    '''
    if ax is None: ax=gca()
    assert(ndim(x)==1 and ndim(y)==3 and y.shape[2]==2 and y.shape[1]==2)
    assert(method=='plot' or method=='scatter')
    subscripts=['0','x','y','z']

    yv=array([s2vec(yi) for yi in y]).real
    colormap=cm.rainbow(linspace(0,0.8,4))
    plts=[]
    for i in xrange(4):
        if method=='plot':
            plts+=plot(x,yv[:,i],lw=3,color=colormap[i],**kwargs)
        else:
            plts.append(scatter(x,yv[:,i],s=30,edgecolors=colormap[i],facecolors='none',**kwargs))
    legend(plts,[r'$%s_%s$'%(label,sub) for sub in subscripts])
    return plts
