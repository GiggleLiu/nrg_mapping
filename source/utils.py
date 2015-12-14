'''
Physics utility library
'''

from numpy import *
from scipy.linalg import eigh,norm,inv
from scipy.interpolate import interp1d
from matplotlib import cm
from matplotlib.pyplot import *
import gmpy2,pdb

__all__=['sx','sy','sz','Gmat','eigh_pauliv_npy','plot_pauli_components','mpconj','mpqr','H2G','s2vec','vec2s','sqrth2']

############################ DEFINITIONS ##############################
# pauli spin
sx = array([[0, 1],[ 1, 0]])
sy = array([[0, -1j],[1j, 0]])
sz = array([[1, 0],[0, -1]])
Gmat=array([kron(sz,sx),kron(identity(2),sy),kron(sx,sx),kron(sy,sx),kron(identity(2),sz)])

############################ FUNCTIONS ##############################

#Get the conjugate of matrix A(to avoid a bug of gmpy2.mpc.conj).
mpconj=vectorize(lambda x:gmpy2.mpc(x.real,-x.imag))

def eigh_pauliv_npy(a0,a1,a2,a3):
    '''
    eigen values for pauli vectors - numpy version.

    Parameters:
        :a0/a1/a2/a3: Pauli components.

    Return:
        Tuple of (eval,evecs), the eigenvalue decomposition of A.
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

    Parameters:
        :s: The matrix.
    Return:
        len-4 vector indicating the pauli components.
    '''
    res=array([trace(s),trace(dot(sx,s)),trace(dot(sy,s)),trace(dot(sz,s))])/2
    return res

def vec2s(n):
    '''
    Transform a vector of length 3 or 4 to a pauli matrix.

    Parameters:
        :n: 1D array of length 3 or 4 to specify the `direction` of spin.
    Return:
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

    Parameters:
        :h: An array of hamiltonian.
        :w: The energy(frequency).
        :tp: The type of Green's function.

            * 'r': retarded Green's function.(default)
            * 'a': advanced Green's function.
            * 'matsu': finite temperature Green's function.
        :geta: Smearing factor. default is 1e-2.
        :sigma: Additional self energy.

    Return:
        A(Array of) Green's function.
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
        return inv(z*identity(h.shape[-1])-h)
    else:
        return 1./(z-h)

def mpqr(A):
    '''
    Analytically, get the QR decomposition of a matrix.

    Parameters:
        :A: The matrix.

    Return:
        (Q,R), where QR=A, Q is orthogonal by columns, and R is upper triangular.
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

def plot_pauli_components(x,y,method='plot',ax=None,label=r'\sigma',**kwargs):
    '''
    Plot data by pauli components.

    Parameters
        :x,y: Datas.
        :ax: The axis to plot, will use gca() to get one if None.
        :label: The legend of plots.
        :method: `plot` or `scatter`
        :kwargs: The key word arguments for plot/scatter.
        
    Return:
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
            plts+=plot(x,yv[:,i],color=colormap[i],**kwargs)
        else:
            plts.append(scatter(x,yv[:,i],edgecolors=colormap[i],facecolors='none',**kwargs))
    legend(plts,[r'$%s_%s$'%(label,sub) for sub in subscripts])
    return plts

def sqrth2(A):
    '''
    analytically, get square root of a hermion matrix.

    Parameters:
        :A: The matrix.
    Return:
        2D array, The matrix squareroot of A.
    '''
    if len(A)!=2:
        raise Exception('Error','Matrix Dimension Error!')
    #first, diagonalize it
    evals,evecs=eigh2(A)
    gmsqrt=gmpy2.sqrt
    evals=array([gmsqrt(ev) for ev in evals])
    #recombine it
    v1=evecs[:,0:1]
    v1h=array([[v1[i,0].real-v1[i,0].imag*1j for i in xrange(2)]])
    v2=evecs[:,1:2]
    v2h=array([[v2[i,0].real-v2[i,0].imag*1j for i in xrange(2)]])
    res=evals[0]*dot(v1,v1h)+evals[1]*dot(v2,v2h)
    return res

def invh2(A):
    '''
    analytically, get inversion of a 2 dimensional hermion matrix.

    Parameters:
        :A: the matrix.
    Return:
        2D array, The inversion of A.
    '''
    if len(A)!=2:
        raise Exception('Error','Matrix Dimension Error!')
    a0=(A[0,0]+A[1,1])/2
    a1=(A[0,1]+A[1,0])/2
    a2=(A[0,1]-A[1,0])*1j/2
    a3=(A[0,0]-A[1,1])/2
    N=a1**2+a2**2+a3**2-a0**2
    return array([[-a0+a3,a1-a2*1j],[a1+a2*1j,-a0-a3]])/N

def eigh2(A):
    '''
    analytically, get eigenvalues and eigenvactors of a 2 dimensional hermion matrix.

    Parameters:
        :A: The matrix.

    Return:
        Tuple of (eval,evecs), the eigenvalue decomposition of A.
    '''
    if len(A)!=2:
        raise Exception('Error','Matrix Dimension Error!')
    a0=(A[0,0]+A[1,1])/2.
    a1=(A[0,1]+A[1,0])/2.
    a2=(A[0,1]-A[1,0])*1j/2.
    a3=(A[0,0]-A[1,1])/2.
    if A.dtype==object:
        return eigh_pauliv_mpc(a0,a1,a2,a3)
    else:
        return eigh_pauliv_npy(a0,a1,a2,a3)

def eigh_pauliv_mpc(a0,a1,a2,a3):
    '''
    eigen values for pauli vectors - gmpy version.

    Parameters:
        a0/a1/a2/a3: Pauli vectors.
    Return:
        Tuple of (eval,evecs), the eigenvalue decomposition of A.
    '''
    gmsqrt=gmpy2.sqrt
    gmnorm=gmpy2.norm
    e0=gmsqrt(a1**2+a2**2+a3**2)
    evals=array([a0-e0,a0+e0])
    if gmnorm(a1+a2*1j)<1e-50:
        evecs=array([[gmpy2.mpc(0),gmpy2.mpc(1)],[gmpy2.mpc(1),gmpy2.mpc(0)]])
    else:
        evecs=array([[(a3-e0)/(a1+a2*1j),gmpy2.mpc(1)],[(a3+e0)/(a1+a2*1j),gmpy2.mpc(1)]])
    for i in xrange(2):
        #evecs[i]=evecs[i]/gmsqrt(sum(gmnorm(evecs[i])))
        evecs[i]=evecs[i]/gmsqrt(gmnorm(evecs[i,0])+gmnorm(evecs[i,1]))
    return evals,evecs.T

def exinterp(xlist,ylist):
    '''
    Linearly interplolate or extrapolate a curve, optimizing scipy.iterpolate.iterp1d to allow extrapolation.

    Parameters:
        :xlist/ylist: The input function y=f(x).
    Return:
        A inter/exter-polate function.
    '''
    interpolator=interp1d(xlist,ylist,axis=0)
    xs = interpolator.x
    ys = interpolator.y
    def pointwise(x):
        rk=ndim(x)
        if rk==0:
            if x < xs[0]:
                return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
            elif x > xs[-1]:
                return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
            else:
                return interpolator(x)
        else:
            smask=x<xs[0]
            lmask=x>xs[-1]
            mmask=~(smask|lmask)
            lxs=x[lmask]
            sxs=x[smask]
            mxs=x[mmask]
            syl=ys[0]+(sxs-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
            lyl=ys[-1]+(lxs-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
            myl=interpolator(mxs)
            return concatenate([syl,myl,lyl],axis=0)
    return pointwise


