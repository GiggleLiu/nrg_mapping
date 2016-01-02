'''
Map a discretized model into a Chain by the method of (block-)lanczos tridiagonalization.
checking method is also provided.
'''

from numpy import *
from scipy.sparse import block_diag
from scipy.linalg import eigvalsh,norm
from matplotlib.pyplot import *
from matplotlib import cm
import gmpy2,pdb

from utils import H2G,s2vec,mpqr
from tridiagonalize import tridiagonalize_qr,tridiagonalize
from chain import Chain
from nrg_setting import DATA_FOLDER

__all__=['map2chain','check_spec']

#MPI setting
try:
    from mpi4py import MPI
    COMM=MPI.COMM_WORLD
    SIZE=COMM.Get_size()
    RANK=COMM.Get_rank()
except:
    COMM=None
    SIZE=1
    RANK=0

def map2chain(model,prec=5000):
    '''
    Map discretized model to a chain model using lanczos method.

    Parameters:
        :model: The discretized model(<DiscModel> instance).
        :prec: The precision(by bit instead of digit),
        :the higher, the slower, 2000 to 6000 is recommended.

    Return:
        A <Chain> instance.
    '''
    Elist,Tlist=model.Elist,model.Tlist
    nsite=model.N_pos
    is_scalar=model.is_scalar

    #first, orthogonalize Tlist to get the first site.
    ntask=(model.nz-1)/SIZE+1
    cl=[]
    for i in xrange(model.nz):
        if i/ntask==RANK:
            ti=Tlist[:,i]  #unitary vector
            ti=ti.reshape([-1,model.nband])
            qq,t0=mpqr(ti)

            print 'Mapping to chain through lanczos tridiagonalization for z-number %s ...'%model.z[i]
            #the m should not be greater than scale length-N, otherwise artificial `zero modes` will take place, but how does this happen?
            eml=Elist[:,i]
            #multi-band lanczos,
            H=block_diag(eml)
            if is_scalar:
                data,offset=tridiagonalize(H,q=qq[:,0],m=nsite,prec=prec)  #we need to perform N+1 recursion to get N sub-diagonal terms.
                chain=Chain(t0[0,0],data[1],data[2])
            else:
                data,offset=tridiagonalize_qr(H,q=qq.reshape([-1,model.nband]),m=nsite,prec=prec)  #we need to perform N+1 recursion to get N sub-diagonal terms.
                chain=Chain(t0,data[1],data[2])
            cl.append(chain)
    if SIZE>1:
        cl=COMM.gather(cl,root=0)
        if RANK==0:
            cl=concatenate(cl,axis=0)
        cl=COMM.bcast(cl,root=0)
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
    print 'Recovering Spectrum ...'
    dlv=[]
    for iz in xrange(nz):
        print 'Recovering spectrum for %s-th z number.'%iz
        chain=chains[iz]
        el=chain.elist
        tl=concatenate([chain.t0[newaxis,...],chain.tlist],axis=0)
        dl=[]
        for w in wlist:
            sigma=0
            for e,t in zip(el[::-1],tl[::-1]):
                geta=abs(w)+1e-10
                g0=H2G(w=w,h=e+sigma,geta=smearing*geta/nz)
                tH=transpose(conj(t))
                sigma=dot(tH,dot(g0,t))
            dl.append(1j*(sigma-sigma.T.conj())/2./pi)
        dlv.append(dl)
    dlv=mean(dlv,axis=0)
    if chain.is_scalar:
        dlv=dlv
        nplt=1
    elif nband==2 and mode=='pauli':
        dlv=array([s2vec(d) for d in dlv])
        nplt=4
    else:
        dlv=array([eigvalsh(d) for d in dlv])
        nplt=chain.nband
    dlv0=array([rhofunc(w) for w in wlist])
    dlv=dlv.real
    if chain.is_scalar:
        dlv=dlv[:,newaxis]
    colormap=cm.rainbow(linspace(0,0.8,nplt))
    if mode=='pauli':
        dlv0=array([s2vec(d) for d in dlv0]).real
    elif chain.is_scalar:
        dlv0=dlv0[:,newaxis]
    else:
        dlv0=array([eigvalsh(d) for d in dlv0])
    plts=[]
    for i in xrange(nplt):
        for mask in [wlist>0,wlist<0]:
            plts+=plot(wlist[mask],dlv0[mask,i],lw=3,color=colormap[i])
    for i in xrange(nplt):
        for mask in [wlist>0,wlist<0]:
            plts.append(scatter(wlist[mask],dlv[mask,i],s=30,edgecolors=colormap[i],facecolors='none'))
    if mode=='pauli':
        legend(plts[::2],[r"$\rho_0$",r"$\rho_x$",r"$\rho_y$",r"$\rho_z$",r"$\rho''_0$",r"$\rho''_x$",r"$\rho''_y$",r"$\rho''_z$"],ncol=2)
    else:
        legend(plts[::2],[r"$\rho_%s$"%i for i in xrange(chain.nband)]+[r"$\rho''_%s$"%i for i in xrange(chain.nband)],ncol=2)
    xlabel(r'$\omega$',fontsize=16)
    xticks([-1,0,1],['-D',0,'D'],fontsize=16)
