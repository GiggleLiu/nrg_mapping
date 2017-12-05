'''
Map a discretized model into a Chain by the method of (block-)lanczos tridiagonalization.
checking method is also provided.
'''

from numpy import *
from scipy.sparse import block_diag
from scipy.linalg import eigvalsh,norm,qr
from scipy.integrate import trapz
from matplotlib.pyplot import *
from matplotlib import cm
import pdb

from .utils import H2G,s2vec
from .tridiagonalize import *
from .chain import Chain
from .lib.futils import hybri_chain

__all__=['map2chain','check_spec','show_scaling']

def map2chain(model,nsite=None,normalize_method='qr'):
    '''
    Map discretized model to a chain model using lanczos method.

    Parameters:
        :model: <DiscModel>, the discretized model(<DiscModel> instance).
        :nsite: int, the number of site, default value is the half size of the bath to avoid symmetry cutoff.
        :normalize_method: str, the normalization strategy for block lanczos.

            * 'qr': QR decomposition, will get upper triangular hoppings.
            * 'sqrtm': Matrix squareroot, will get more symmetric hoppings.
            * 'mpqr': High precision version of 'qr'.

    Return:
        A <Chain> instance.
    '''
    Elist,Tlist=model.Elist,model.Tlist
    if nsite is None: nsite=model.N_pos
    is_scalar=model.is_scalar

    #first, orthogonalize Tlist to get the first site.
    cl=[]
    for i in range(model.nz):
        ti=Tlist[:,i]  #unitary vector
        ti=ti.reshape([-1,model.nband])
        qq,t0=qr(ti,mode='economic')

        print('Mapping to chain through lanczos tridiagonalization for z-number %s ...'%model.z[i])
        #the m should not be greater than scale length-N, otherwise artificial `zero modes` will take place, but how does this happen?
        eml=Elist[:,i]
        #multi-band lanczos,
        H=block_diag(eml)
        if is_scalar and normalize_method=='mpqr':
            data,offset=tridiagonalize_mp(H,q=qq[:,0],m=nsite)
            chain=Chain(asarray(data[1])[:,newaxis,newaxis],append(t0[0],data[2])[:,newaxis,newaxis])
        else:
            if normalize_method=='sqrtm':
                data,offset=tridiagonalize_sqrtm(H,q=qq.reshape([-1,model.nband]),m=nsite)
            if normalize_method=='mpqr':
                data,offset=tridiagonalize_mpqr(H,q=qq.reshape([-1,model.nband]),m=nsite)
            else:
                data,offset=tridiagonalize_qr(H,q=qq.reshape([-1,model.nband]),m=nsite)
            chain=Chain(asarray(data[1]),concatenate([t0[newaxis],data[0]],axis=0))
        cl.append(chain)
    return cl

def check_spec(chains,rhofunc,wlist,mode='eval',smearing=1.):
    '''
    Check mapping quality for wilson chain.

    Parameters:
        :chains: list, a list of chain after mapping.
        :rhofunc: function, Hybridization function.
        :wlist: 1D array, the frequency space.
        :mode: str, Choose the checking method,
            * 'eval' -> check eigenvalues.
            * 'pauli' -> check pauli components, it is only valid for 2 band system.

        :smearing: The smearing factor.
    '''
    nband=chains[0].nband
    nz=len(chains)
    print('Recovering Spectrum ...')
    dlv=hybri_chain(tlist=array([chain.tlist for chain in chains]),elist=array([chain.elist for chain in chains]),wlist=wlist,smearing=smearing*1./nz)
    if nband==1:
        dlv=dlv[:,0,0]
        nplt=1
    elif nband==2 and mode=='pauli':
        dlv=array([s2vec(d) for d in dlv])
        nplt=4
    else:
        dlv=array([eigvalsh(d) for d in dlv])
        nplt=nband
    print('Integrate = %s'%trapz(dlv,wlist,axis=0))
    dlv0=array([rhofunc(w) for w in wlist])
    dlv=dlv.real
    if nband==1:
        dlv=dlv[:,newaxis]
    colormap=cm.rainbow(linspace(0,0.8,nplt))
    if mode=='pauli':
        dlv0=array([s2vec(d) for d in dlv0]).real
    elif nband==1:
        dlv0=dlv0[:,newaxis]
    else:
        dlv0=array([eigvalsh(d) for d in dlv0])
    plts=[]
    for i in range(nplt):
        for mask in [wlist>0,wlist<0]:
            plts+=plot(wlist[mask],dlv0[mask,i],lw=3,color=colormap[i])
    for i in range(nplt):
        for mask in [wlist>0,wlist<0]:
            plts.append(scatter(wlist[mask],dlv[mask,i],s=30,edgecolors=colormap[i],facecolors='none'))
    if mode=='pauli':
        legend(plts[::2],[r"$\rho_0$",r"$\rho_x$",r"$\rho_y$",r"$\rho_z$",r"$\rho''_0$",r"$\rho''_x$",r"$\rho''_y$",r"$\rho''_z$"],ncol=2)
    else:
        legend(plts[::2],[r"$\rho_%s$"%i for i in range(nband)]+[r"$\rho''_%s$"%i for i in range(nband)],ncol=2)
    xlabel(r'$\omega$',fontsize=16)
    xticks([-1,0,1],['-D',0,'D'],fontsize=16)

def show_scaling(chain,logy=True):
    '''
    Check the scaling of chain.
    '''
    tlist=concatenate([chain.t0[newaxis,...],chain.tlist])
    ydata=([norm(t) for t in tlist])
    if logy: ydata=log(ydata)
    plot(arange(len(tlist)),ydata)
    xlabel('site')
    ylabel('log(|t|)' if logy else '|t|')
