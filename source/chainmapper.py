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

from utils import H2G,s2vec,qr2
from tridiagonalize import tridiagonalize_qr,tridiagonalize
from chain import Chain

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

class ChainMapper(object):
    '''
    A Chain Model Mapper for NRG.

    prec:
        the precision in mapping process.
    '''
    def __init__(self,prec=3000):
        self.prec=prec
        gmpy2.get_context().precision=prec

    def map(self,model):
        '''
        Map discretized model to a chain model using lanczos method.

        model:
            the discretized model(DiscModel instance).

        *return*:
            a Chain object
        '''
        prec=self.prec
        Elist,Tlist=model.Elist,model.Tlist

        #first, orthogonalize Tlist to get the first site.
        ntask=(model.nz-1)/SIZE+1
        t0li=[];eli=[];tli=[]
        for i in xrange(model.nz):
            if i/ntask==RANK:
                ti=Tlist[:,i]  #unitary vector
                ti=ti.reshape([-1,model.nband])
                qq,t0=qr2(ti)

                print 'Mapping to chain through lanczos tridiagonalization for z-number %s ...'%model.z[i]
                #the m should not be greater than scale length-N, otherwise artificial `zero modes` will take place, but how does this happen?
                eml=Elist[:,i]
                #multi-band lanczos,
                #dense matrix is used here for simplicity, we should use sparse matrix here for efficiency consideration,
                H=block_diag(eml).toarray()
                H=vectorize(gmpy2.mpc)(H)
                data,offset=tridiagonalize_qr(H,q=qq.reshape([-1,model.nband]),m=model.N,prec=prec)  #we need to perform N+1 recursion to get N sub-diagonal terms.
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
        return Chain(complex128(t0l),complex128(swapaxes(el,0,1)),complex128(swapaxes(tl,0,1)))

    def check_spec(self,chain,dischandler,rhofunc,mode='eval',Nw=500,smearing=1.5):
        '''
        check mapping quality.

        chain:
            the chain after mapping.
        dischandler:
            discretization handler.
        rhofunc:
            hybridization function.
        mode:
            `eval` -> check eigenvalues
            `pauli` -> check pauli components
        Nw:
            number of samples in w-space.
        '''
        Lambda=dischandler.Lambda
        tlist=chain.tlist
        elist=chain.elist
        t0=chain.t0
        nz=chain.elist.shape[1]
        is_scalar=chain.is_scalar
        Gap=dischandler.Gap
        D=dischandler.D
        filename='data/checkspec%s_%s_%s'%(nz,dischandler.token,Gap[1])
        ion()
        print 'Recovering Spectrum ...'
        dlv=0;dle=0
        for iz in xrange(nz):
            print 'Running for %s-th z number.'%iz
            wlist1=(D[0]-Gap[0])*logspace(-10,0,Nw)+Gap[0]
            wlist2=(D[1]-Gap[1])*logspace(-10,0,Nw)+Gap[1]
            wlist=append(wlist1[::-1],wlist2)
            el=elist[:,iz]
            tl=concatenate([t0[iz][newaxis,...],tlist[:,iz]],axis=0)
            dl=[]
            for w in wlist:
                sigma=0
                for e,t in zip(el[::-1],tl[::-1]):
                    g0=H2G(w=w,h=e+sigma,geta=smearing/nz*(max(w-Gap[1] if w>0 else Gap[0]-w,1e-2)))
                    tH=transpose(conj(t))
                    sigma=dot(tH,dot(g0,t))
                dl.append(1j*(sigma-sigma.T.conj())/2./pi)
            if is_scalar:
                dlv=dlv+array(dl)
                nplt=1
            elif dischandler.nband==2 and mode=='pauli':
                dlv=dlv+array([s2vec(d) for d in dl])
                nplt=4
            else:
                dlv=dlv+array([eigvalsh(d) for d in dl])
                nplt=chain.nband
        dlv0=array([rhofunc(w) for w in wlist])
        dlv=dlv.real/nz
        if is_scalar:
            dlv=dlv[:,newaxis]
        colormap=cm.rainbow(linspace(0,0.8,nplt))
        if mode=='pauli':
            dlv0=array([s2vec(d) for d in dlv0]).real
        elif dischandler.is_scalar:
            dlv0=dlv0[:,newaxis]
        else:
            dlv0=array([eigvalsh(d) for d in dlv0])
        print dlv0.shape,dlv.shape
        savetxt(filename+'.dat',concatenate([wlist[:,newaxis],dlv0,dlv],axis=1))
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
        print 'Check Spectrum Finished, Press `c` to Save Figure.'
        pdb.set_trace()
        savefig(filename+'.png')

