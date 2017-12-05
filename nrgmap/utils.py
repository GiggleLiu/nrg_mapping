'''
Physics utility library
'''

from numpy import *
from scipy.linalg import eigh,norm,inv
from scipy.interpolate import interp1d
from matplotlib import cm
from matplotlib.pyplot import *
import pdb

__all__=['sx','sy','sz','Gmat','plot_pauli_components','H2G','s2vec','vec2s','get_wlist']

############################ DEFINITIONS ##############################
# pauli spin
sx = array([[0, 1],[ 1, 0]])
sy = array([[0, -1j],[1j, 0]])
sz = array([[1, 0],[0, -1]])
#Clifford algebra
Gmat=array([kron(sz,sx),kron(identity(2),sy),kron(sx,sx),kron(sy,sx),kron(identity(2),sz)])

############################ FUNCTIONS ##############################

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
        for i in range(len(n)):
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
    for i in range(4):
        if method=='plot':
            plts+=plot(x,yv[:,i],color=colormap[i],**kwargs)
        else:
            plts.append(scatter(x,yv[:,i],edgecolors=colormap[i],facecolors='none',**kwargs))
    legend(plts,[r'$%s_%s$'%(label,sub) for sub in subscripts])
    return plts

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

def get_wlist(w0,Nw,mesh_type,D=1,Gap=0):
    '''
    A log mesh can make low energy component of rho(w) more accurate.

    Parameters:
        :w0: float, The starting w for wlist for `log` and `sclog` type wlist, it must be smaller than lowest energy scale!
        :Nw: integer/len-2 tuple, The number of samples in each branch.
        :mesh_type: string, The type of wlist.

            * `linear` -> linear mesh.
            * `log` -> log mesh.
            * `sclog` -> log mesh suited for superconductors.
        :D: Interger/len-2 tuple, the band interval.
        :Gap: Interger/len-2 tuple, the gap interval.

    Return:
        1D array, the frequency space.
    '''
    assert(mesh_type=='linear' or mesh_type=='log' or mesh_type=='sclog')
    if ndim(Gap)==0: Gap=[-Gap,Gap]
    if ndim(D)==0: D=[-D,D]
    if ndim(Nw)==0: Nw=[Nw/2,Nw-Nw/2]

    if mesh_type=='linear':
        wlist=[linspace(-Gap[0],-D[0],Nw[0]),linspace(Gap[1],D[1],Nw[1])]
        return concatenate([-wlist[0][::-1],wlist[1]])
    elif mesh_type=='log':
        wlist=[logspace(log(w0)/log(10),log(-D[0]+Gap[0])/log(10),Nw[0]-1)-Gap[0],logspace(log(w0)/log(10),log(D[1]-Gap[1])/log(10),Nw[1]-1)+Gap[1]]
        #add zeros
        return concatenate([-wlist[0][::-1],array([Gap[0]-1e-30,Gap[1]+1e-30]),wlist[1]])
    elif mesh_type=='sclog':
        wlist=[logspace(log(w0),log(-D[0]+Gap[0]),Nw[0]-1,base=e)-Gap[0],logspace(log(w0),log(D[1]-Gap[1]),Nw[1]-1,base=e)+Gap[1]]
        #add zeros
        return concatenate([-wlist[0][::-1],array([Gap[0]-1e-30,Gap[1]+1e-30]),wlist[1]])
    if (wlist[0][1]-wlist[0][0])==0 or (wlist[1][1]-wlist[1][0])==0:
        raise Exception('Precision Error, Reduce your scaling factor or scaling level!')
