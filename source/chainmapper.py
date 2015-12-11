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

    model:
        the discretized model(DiscModel instance).

    *return*:
        a Chain object
    '''
    Elist,Tlist=model.Elist,model.Tlist
    nsite=model.N_pos
    is_scalar=model.is_scalar

    #first, orthogonalize Tlist to get the first site.
    ntask=(model.nz-1)/SIZE+1
    t0li=[];eli=[];tli=[]
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
            data,offset=tridiagonalize_qr(H,q=qq.reshape([-1,model.nband]),m=nsite,prec=prec)  #we need to perform N+1 recursion to get N sub-diagonal terms.
            t0li.append(t0)
            eli.append(data[1])
            tli.append(data[2])
    if SIZE>1:
        t0l=COMM.gather(t0li,root=0)
        el=COMM.gather(eli,root=0)
        tl=COMM.gather(tli,root=0)
        if RANK==0:
            t0l=concatenate(t0l,axis=0)
            el=concatenate(el,axis=0)
            tl=concatenate(tl,axis=0)
        t0l=COMM.bcast(t0l,root=0)
        el=COMM.bcast(el,root=0)
        tl=COMM.bcast(tl,root=0)
    else:
        t0l=t0li;tl=tli;el=eli
    t0=complex128(t0l)
    el=swapaxes(complex128(el),0,1)
    tl=swapaxes(complex128(tl),0,1)
    if is_scalar:
        t0,el,tl=t0[...,0,0],el[...,0,0],tl[...,0,0]
    return Chain(t0,el,tl)

def check_spec(chain,rhofunc,wlist,mode='eval',smearing=1.,save_token=''):
    '''
    check mapping quality.

    chain:
        The chain after mapping.
    rhofunc:
        Hybridization function.
    wlist:
        The frequency space.
    mode:
        `eval` -> check eigenvalues
        `pauli` -> check pauli components
    smearing:
        The smearing factor.
    save_token:
        The token to save figure.
    '''
    tlist=chain.tlist
    elist=chain.elist
    nband=chain.nband
    t0=chain.t0
    nz=chain.elist.shape[1]
    is_scalar=ndim(elist)==2
    ion()
    print 'Recovering Spectrum ...'
    dlv=[]
    for iz in xrange(nz):
        print 'Recovering spectrum for %s-th z number.'%iz
        el=elist[:,iz]
        tl=concatenate([t0[iz][newaxis,...],tlist[:,iz]],axis=0)
        dl=[]
        for w in wlist:
            sigma=0
            for e,t in zip(el[::-1],tl[::-1]):
                geta=abs(w)+0.01
                g0=H2G(w=w,h=e+sigma,geta=smearing/nz*geta)
                tH=transpose(conj(t))
                sigma=dot(tH,dot(g0,t))
            dl.append(1j*(sigma-sigma.T.conj())/2./pi)
        dlv.append(dl)
    dlv=mean(dlv,axis=0)
    if is_scalar:
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
    if is_scalar:
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
            plts.append(scatter(wlist[mask][::3],dlv[mask,i][::3],s=30,edgecolors=colormap[i],facecolors='none'))
    if mode=='pauli':
        legend(plts[::2],[r"$\rho_0$",r"$\rho_x$",r"$\rho_y$",r"$\rho_z$",r"$\rho''_0$",r"$\rho''_x$",r"$\rho''_y$",r"$\rho''_z$"],ncol=2)
    else:
        legend(plts[::2],[r"$\rho_%s$"%i for i in xrange(chain.nband)]+[r"$\rho''_%s$"%i for i in xrange(chain.nband)],ncol=2)
    xlabel(r'$\omega$',fontsize=16)
    xticks([-1,0,1],['-D',0,'D'],fontsize=16)

    #saving data
    filename='%s/checkspec_%s'%(DATA_FOLDER,save_token)
    savefig(filename+'.png')

