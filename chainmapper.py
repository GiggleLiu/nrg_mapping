#!/usr/bin/python
'''
Map a discretized model into a Chain by the method of (block-)lanczos tridiagonalization.
checking method is also provided.
'''
import gmpy2,pdb
from numpy import *
from utils import qr2,H2G,s2vec
from tridiagonalize import tridiagonalize_qr,tridiagonalize
from scipy.sparse import block_diag
from matplotlib.pyplot import *
from matplotlib import cm

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
        t0l=[];el=[];tl=[]
        for i in xrange(model.nz):
            ti=Tlist[:,i]  #unitary vector
            if model.nband==1:
                t0=norm(ti)
                qq=ti/t0
            else:
                ti=ti.reshape([-1,model.nband])
                qq,t0=qr2(ti)

            print 'Mapping to chain through lanczos tridiagonalization for z-number %s ...'%model.z[i]
            #the m should not be greater than scale length-N, otherwise artificial `zero modes` will take place, but how does this happen?
            eml=Elist[:,i]
            if not model.nband>1:
                #single band lanczos.
                H=diag(eml,0)
                data,offset=tridiagonalize(H,q=qq,m=model.N,prec=prec)  #we need to perform N+1 recursion to get N sub-diagonal terms.
            else:
                #double band lanczos.
                H=block_diag(eml).todense()
                H=vectorize(gmpy2.mpc)(H)
                data,offset=tridiagonalize_qr(H,q=qq.reshape([-1,2]),m=model.N,prec=prec)  #we need to perform N+1 recursion to get N sub-diagonal terms.
            t0l.append(t0)
            el.append(data[1])
            tl.append(data[2])
        return Chain(complex128(t0l),complex128(swapaxes(el,0,1)),complex128(swapaxes(tl,0,1)))

    def check_spec(self,chain,dischandler,rhofunc):
        '''
        check mapping quality.

        chain:
            the chain after mapping.
        dischandler:
            discretization handler.
        rhofunc:
            hybridization function.
        '''
        Lambda=dischandler.Lambda
        tlist=chain.tlist
        elist=chain.elist
        t0=chain.t0
        nz=chain.elist.shape[1]
        drank=dischandler.nband
        Gap=dischandler.Gap
        D=dischandler.D
        if drank==0:
            nband=1
        elif drank==2:
            nband=rhofunc(0).shape[-1]
            if nband!=2:
                raise Exception('Error','%s Bands system is not supported!'%nband)
        else:
            raise Exception('Error','rank of rhofunc should be 0 or 2 but get %s.'%drank)
        filename='data/checkspec%s_%s_%s'%(nz,dischandler.token,Gap[1])
        ion()
        print 'Recovering Spectrum ...'
        dlv=0;dle=0
        for iz in xrange(nz):
            print 'Running for %s-th z number.'%iz
            wlist1=(D[0]-Gap[0])*logspace(-10,0,500)+Gap[0]
            wlist2=(D[1]-Gap[1])*logspace(-10,0,500)+Gap[1]
            wlist=append(wlist1[::-1],wlist2)
            #wlist=linspace(scale.D[0],scale.D[1],1000)
            el=elist[:,iz]
            tl=concatenate([t0[iz].T.conj()[newaxis,...],tlist[:,iz]],axis=0)
            dl=[]
            for w in wlist:
                sigma=0
                for e,t in zip(el[::-1],tl[::-1]):
                    g0=H2G(w=w,h=e+sigma,geta=1.5/nz*(max(w-Gap[1] if w>0 else Gap[0]-w,1e-3)))
                    tH=transpose(conj(t))
                    sigma=dot(tH,dot(g0,t))
                dl.append(1j*(sigma-sigma.T.conj())/2./pi)
            if nband==1:
                dlv=dlv+array(dl)
            else:
                dlv=dlv+array([s2vec(d) for d in dl])
        dlv0=array([rhofunc(w) for w in wlist])
        dlv=dlv.real/nz
        if nband==2:
            colormap=cm.rainbow(linspace(0,0.8,4))
            colormap2=cm.rainbow(linspace(0.2,1.0,4))
            dlv0=array([s2vec(d) for d in dlv0]).real
            savetxt(filename+'.dat',concatenate([wlist[:,newaxis],dlv0,dlv],axis=1))
            plts=[]
            for i in xrange(4):
                for mask in [wlist>0,wlist<0]:
                    plts+=plot(wlist[mask],dlv0[mask,i],lw=3,color=colormap[i])
            for i in xrange(4):
                for mask in [wlist>0,wlist<0]:
                    plts.append(scatter(wlist[mask][::3],dlv[mask,i][::3],s=30,edgecolors=colormap[i],facecolors='none'))
            legend(plts[::2],["$\\rho_0$","$\\rho_x$","$\\rho_y$","$\\rho_z$","$\\rho''_0$","$\\rho''_x$","$\\rho''_y$","$\\rho''_z$"],ncol=2)
        else:
            savetxt(filename+'.dat',concatenate([wlist[:,newaxis],dlv0[:,newaxis],dlv[:,newaxis]],axis=1))
            plot(wlist,dlv0,lw=3)
            plot(wlist,dlv,'--',lw=3)
            legend(["$\\rho$","$\\rho''$"])
        xlabel('$\\omega$',fontsize=16)
        xticks([-1,0,1],['-D',0,'D'],fontsize=16)
        print 'Check Spectrum Finished, Press `c` to Save Figure.'
        pdb.set_trace()
        savefig(filename+'.png')

def save_chain(token,chain):
    '''
    save a Chain instance to files.

    token:
        a string as a prefix to store datas of a chain.
    chain:
        a chain instance.
    '''
    token='data/'+token
    tlist=concatenate([chain.t0[newaxis,...],chain.tlist],axis=0)
    savetxt(token+'.info.dat',array(chain.elist.shape))
    savetxt(token+'.el.dat',chain.elist.ravel().view('float64'))
    savetxt(token+'.tl.dat',tlist.ravel().view('float64'))

def load_chain(token):
    '''
    load a Chain instance from files.

    token:
        a string as a prefix to store datas of a chain.
    *return*:
        a Chain instance.
    '''
    token='data/'+token
    shape=loadtxt(token+'.info.dat')
    elist=loadtxt(token+'.el.dat').view('complex128').reshape(shape)
    tlist=loadtxt(token+'.tl.dat').view('complex128').reshape(shape)
    return Chain(tlist[0],elist,tlist[1:])


class Chain(object):
    '''
    NRG chain class.
    
    t0:
        the coupling term of the first site and the impurity.
    elist/tlist:
        a list of on-site energies and coupling terms.
    '''
    def __init__(self,t0,elist,tlist):
        self.t0=complex128(t0)
        self.elist=complex128(elist)
        self.tlist=complex128(tlist)
