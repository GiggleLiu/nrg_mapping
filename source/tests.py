'''
Tests for nrg mapping procedures.
'''

from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from matplotlib.pyplot import *
from scipy import sparse as sps
from scipy.linalg import qr,eigvalsh,norm
import time,pdb,gmpy2

from hybri_sc import *
from discretization import *
from utils import *
from ticklib import *
from chainmapper import *
from discmodel import *
from chain import *
from nrg_setting import PRECISION
from tridiagonalize import tridiagonalize_qr,tridiagonalize,tridiagonalize2,linalg_mp,linalg_np,construct_tridmat

def test_get_wlist():
    '''test for get_wlist'''
    D=[-1,2]
    Gap=[-0.1,0.4]
    w0=1e-8
    Nw=500
    for mesh_type in ['log','sclog','linear']:
        print 'Testing for %s wlist.'%mesh_type
        wlist=get_wlist(w0,Nw,mesh_type,D=D,Gap=Gap)
        assert_(all(diff(wlist)>0))
        assert_almost_equal(wlist[0],D[0])
        assert_almost_equal(wlist[-1],D[-1])
        assert_(all(wlist[wlist>0]>=Gap[1]) and all(wlist[wlist<0]<=Gap[0]))
        assert_(len(wlist)==2*Nw)

def test_hybri_sc():
    '''test hybridization function for superconductor.'''
    D0=2.
    Gap=0.3
    Gamma=0.5/pi
    skew=0.3
    D=sqrt(D0**2+Gap**2)
    wlist=get_wlist(w0=1e-8,Nw=2000,mesh_type='sclog',Gap=Gap,D=D0+Gap)
    dfunc1=get_hybri(Gap,Gamma,D0=D0,mu=0.,eta=1e-10)
    dfunc2=get_hybri_wideband(Gap,Gamma,D=D,mu=0.,eta=1e-10)
    dfunc3=get_hybri_skew(Gap,Gamma,D=D,eta=1e-10,skew=skew)
    dfuncs=[dfunc1,dfunc2,dfunc3]
    ion()
    fig=figure(figsize=(4,9))
    dls=[array([df(w) for w in wlist]) for df in dfuncs]
    titles=['normal','wideband','skew']
    for i in xrange(3):
        subplot('31%s'%(i+1))
        title(titles[i])
        plot_pauli_components(wlist,dls[i],lw=2)
        ylim(-0.1,0.2)
    assert_allclose(dls[0],dls[1],atol=1e-5)
    tight_layout()

class MapTest():
    '''test hybridization function for superconductor.'''
    def __init__(self,nband):
        self.nband=nband
        self.Gamma=0.5/pi
        self.Lambda=1.7
        if nband==2:
            D0=2.
            self.Gap=0.3
            D=sqrt(D0**2+self.Gap**2)
            self.D=[-D,D]
            self.wlist=get_wlist(w0=1e-8,Nw=10000,mesh_type='sclog',Gap=self.Gap,D=D)
            self.rhofunc=get_hybri_skew(self.Gap,self.Gamma,D=D,eta=1e-15,skew=0.3)
        elif nband==1:
            self.D=[-1,1.5]
            self.Gap=0
            self.rhofunc=lambda w:self.Gamma*abs(w)
            self.wlist=get_wlist(w0=1e-8,Nw=10000,mesh_type='log',Gap=self.Gap,D=self.D)
        elif nband==4:
            self.D=[-1.,0.5]             #the energy window.
            self.Gap=0
            self.wlist=get_wlist(w0=1e-8,Nw=10000,mesh_type='log',Gap=self.Gap,D=self.D)
            #self.rhofunc=lambda w:identity(4)+0.3*w*Gmat[0]+0.3*w**2*Gmat[2]+0.1*kron(sz,sz)     #the case without degeneracy.
            self.rhofunc=lambda w:identity(4)+0.3*w*Gmat[0]+0.3*w**2*Gmat[2] #the case with degeneracy
        self.rholist=array([self.rhofunc(w) for w in self.wlist])

        #create the model
        self.N=25
        nz=5
        self.z=linspace(0.5/nz,1-0.5/nz,nz)
        self.discmodel=quick_map(rhofunc=self.rhofunc,wlist=self.wlist,N=self.N,z=self.z,Nx=200000,tick_params={'tick_type':'adaptive','Gap':self.Gap,'Lambda':self.Lambda},autofix=1e-5)[1]
        assert_(self.discmodel.N_pos==self.N and self.discmodel.N_neg==self.N and self.discmodel.N==2*self.N)
        if nband==1:
            assert_(self.discmodel.Elist_pos.shape==(self.N,nz))
        else:
            assert_(self.discmodel.Elist_pos.shape==(self.N,nz,nband,nband))

        #map to a chain
        self.chains=map2chain(self.discmodel,prec=PRECISION)

    def test_saveload(self):
        '''save and load data.'''
        for iz,chain in zip(self.z,self.chains):
            token='test_%s'%iz
            self.discmodel.save(token)
            model=load_discmodel(token)
            assert_allclose(model.Elist,self.discmodel.Elist)
            assert_allclose(model.Tlist,self.discmodel.Tlist)
            assert_allclose(model.z,self.discmodel.z)

            chain.save(token)
            chain2=load_chain(token)
            assert_allclose(chain2.elist,chain.elist)
            assert_allclose(chain2.tlist,chain.tlist)
            assert_allclose(chain2.t0,chain.t0)

    def test_tick(self):
        '''test for ticks.'''
        tick_types=['log','sclog','adaptive','linear','adaptive_linear','ed']
        Lambda=self.Lambda
        N=self.N
        wlist=self.wlist
        pmask=wlist>0
        ion()
        rholist=self.rholist[pmask]
        if self.nband>1:
            rholist=sqrt((rholist*swapaxes(rholist,1,2)).sum(axis=(1,2)))
        colors=['r','g','b','k','y','c']
        for i,tick_type in enumerate(tick_types):
            offset_y=i
            ticker=get_ticker(tick_type,D=self.D[1],N=N,Lambda=Lambda,Gap=self.Gap,wlist=wlist[pmask],rholist=rholist)
            plt=scatter(ticker(arange(2,2+N+1)),offset_y*ones(N+1),edgecolor='none',color=colors[i],label=tick_type)
            #consistancy check
            assert_allclose(ticker(arange(1,N+2)),[ticker(i) for i in xrange(1,N+2)])
        legend(loc=3)

    @dec.slow
    def test_map(self):
        '''test for mapping'''
        plot_wlist=self.wlist[::50]
        if self.nband==2:
            check_disc(rhofunc=self.rhofunc,wlist=plot_wlist,discmodel=self.discmodel,smearing=1,mode='pauli')
            print '***The superconducting model needs some special gradients to cope the smearing factor here,\
                    \nwhich is not included for general purpose,\
                    \nso, don\'t be disappointed by the poor match here, they are artifacts.***'
            ylim(-0.1,0.2)
        elif self.nband==1 or self.nband==4:
            check_disc(rhofunc=self.rhofunc,wlist=plot_wlist,discmodel=self.discmodel,smearing=0.2 if self.nband==1 else 0.4)

    @dec.slow
    def test_chain(self):
        '''test for tridiagonalization.'''
        plot_wlist=self.wlist[::20]
        chains=self.chains
        assert_(chains[0].nsite==self.N)
        nband=self.nband
        if nband==2:
            smearing=1
        elif nband==4:
            smearing=0.4
        else:
            smearing=0.2
        check_spec(rhofunc=self.rhofunc,chains=chains,wlist=plot_wlist,smearing=smearing,mode='pauli' if self.nband==2 else 'eval')
        if self.nband==2:
            ylim(-0.1,0.2)

class TridTest(object):
    '''test tridiagonalization.'''
    def gen_randomH(self,n,p,matrix_type='array'):
        '''get matrix with n-nlock and blocksize p.'''
        if p is None:
            p=1
        N=n*p
        H0=random.random([N,N])+1j*random.random([N,N])
        H0=H0+H0.T.conj()
        if matrix_type=='array':
            return H0
        elif matrix_type=='sparse':
            return sps.csr_matrix(H0)
        else:
            return linalg_mp['tocomplex'](H0)

    def gen_randv0(self,n,p,matrix_type='array'):
        '''generate a random orthorgonal vetor.'''
        if p is not None:
            v0=random.random([n*p,p])
            v0=qr(v0,mode='economic')[0]  #The orthogonality of starting vector v0 will be re-inforced in the program.
        else:
            v0=random.random(n)
            v0=v0/norm(v0)
        if matrix_type=='array':
            return v0
        elif matrix_type=='sparse':
            return sps.csr_matrix(v0)
        else:
            return linalg_mp['tocomplex'](v0)

    def test1(self,n,p,prec,matrix_type,method):
        '''
        check the quality of tridiagonalization.
        '''
        if p==2:
            assert(method=='qr' or method=='sqrtm')
        else:
            assert(method=='qr')
        if prec is not None:
            gmpy2.get_context().precision=prec
        assert(matrix_type in ['array','sparse','mpc'])

        v0=self.gen_randv0(n,p,matrix_type='array')
        H0=self.gen_randomH(n,p,matrix_type=matrix_type)
        if p is not None:
            if method=='qr':
                data,offset=tridiagonalize_qr(H0,q=v0,prec=prec)
            else:
                data,offset=tridiagonalize2(H0,q=v0,prec=prec)
        else:
            data,offset=tridiagonalize(H0,q=v0,prec=prec)
        B=construct_tridmat(data,offset).toarray()
        if sps.issparse(H0):
            H0=H0.toarray()
        H0=complex128(H0)
        e1=eigvalsh(H0)
        e2=eigvalsh(B)
        assert_allclose(e1,e2)

    def test_all(self):
        '''test all the cases.'''
        nl=[1,15]
        pl=[None,1,2,3]
        matrix_types=['array','mpc','sparse']
        precs=[PRECISION]
        for n in nl:
            for p in pl:
                for matrix_type in matrix_types:
                    for prec in precs:
                        print 'Checking for parameter n=%s,p=%s,matrix_type=%s,prec=%s,method=%s'%(n,p,matrix_type,prec,'qr')
                        self.test1(n=n,p=p,prec=prec,matrix_type=matrix_type,method='qr')
                        if p==2:
                            print 'Running n=%s,p=%s,matrix_type=%s,prec=%s,method=%s'%(n,p,matrix_type,prec,'sqrtm')
                            self.test1(n=n,p=p,prec=prec,matrix_type=matrix_type,method='sqrtm')

class LinTest(object):
    def __init__(self):
        self.N=50
        A=random.random([self.N,self.N])+random.random([self.N,self.N])*1j
        self.A=A+A.conj().T
        self.A_high=linalg_mp['tocomplex'](self.A)

    def test_conj(self):
        '''test conj method.'''
        cA0=conj(self.A)
        cA1=linalg_mp['conj'](self.A)
        cA2=linalg_mp['conj'](self.A_high)
        assert_allclose(complex128(cA2),cA0)
        assert_allclose(complex128(cA1),cA0)

    def test_qr(self):
        '''test qr decomposition'''
        q1,r1=linalg_mp['qr'](self.A)
        q2,r2=linalg_mp['qr'](self.A_high)
        assert_allclose(complex128(q1.dot(r1)),self.A,atol=1e-7)
        assert_allclose(complex128(q2.dot(r2)),self.A,atol=1e-7)
        assert_allclose(complex128(q2).T.conj().dot(complex128(q2)),identity(self.N),atol=1e-7)
        assert_allclose(complex128(q1).T.conj().dot(complex128(q1)),identity(self.N),atol=1e-7)

    def test_dim2(self):
        '''test 2 body operations.'''
        funcs=['eigh_pauliv','eigh','sqrtm','inv']
        A=random.random([2,2])+1j*random.random([2,2])
        A=A+A.conj().T
        A_high=linalg_mp['tocomplex'](A)
        for func in funcs:
            print 'Testing %s'%func
            if func=='eigh_pauliv':
                res_np=linalg_np[func](*s2vec(A))
                res_mp=linalg_mp[func](*s2vec(A))
            else:
                res_np=linalg_np[func](A)
                res_mp=linalg_mp[func](A_high)
            if func[:4]=='eigh':
                assert_allclose(res_np[0],complex128(res_mp[0]))
                assert_allclose(abs(res_np[1]),abs(complex128(res_mp[1])))
            elif func=='inv':
                assert_allclose(res_np,complex128(res_mp))
            else:
                assert_allclose(complex128(res_mp.dot(res_mp)),A)

    def test_all(self):
        '''test all'''
        self.test_dim2()
        self.test_conj()
        self.test_qr()

def test_all():
    ion()
    test_get_wlist()
    test_hybri_sc()
    for i in [1,2,4]:
        t0=time.time()
        ti=MapTest(i)
        t1=time.time()
        print 'Elapse, %s'%(t1-t0)
        ti.test_tick()
        ti.test_saveload()
        ti.test_map()
        ti.test_chain()
    TridTest().test_all()
    LinTest().test_all()

#test_get_wlist()
#test_hybri_sc()
#MapTest(2).test_tick()
#MapTest(2).test_saveload()
#MapTest(2).test_map()
#MapTest(4).test_chain()
#TridTest().test_all()
#LinTest().test_all()
test_all()
