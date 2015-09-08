#!/usr/bin/python
'''
Discretize the continuous hybridization function into a discretized model(DiscModel).
checking method is also provided.
'''
import pdb,time
from numpy import *
from utils import ode_ronge_kutta,eigh_pauliv_npy,sx,sy,sz,s2vec,H2G
from matplotlib.pyplot import *
import matplotlib.cm as cm
from scipy.interpolate import interp1d
from scipy.integrate import quadrature,cumtrapz,trapz,simps,quad
from numpy.linalg import eigh,inv,eigvalsh

def get_scalefunc_log(Lambda,D,Gap,sgn):
    '''
    get logarithmic scale tick function.
    \epsilon(x)=\Lambda^(2-x)

    Lambda:
        scaling factor.
    D/Gap:
        the bandwidth/Gap range.
    sgn:
        return the scale function for the positive branch if sgn>0, else the negative one.
    *return*:
        a scale function \epsilon(x).
    '''
    def scalefunc(x):
        res=Lambda**(2-x)*(D-Gap)+Gap
        if ndim(x)==0:
            if x<=2:
                return D
            else:
                return res
        else:
            res[x<=2]=D
            return res
    return scalefunc

def get_scalefunc_sclog(Lambda,D,Gap,sgn):
    '''
    get logarithmic scale tick function suited for superconductor(logarithmic for normal part).

    Lambda:
        scaling factor.
    D/Gap:
        the bandwidth/Gap range.
    sgn:
        return the scale function for the positive branch if sgn>0, else the negative one.
    *return*:
        a scale function \epsilon(x).
    '''
    D2=D**2-Gap**2
    def scalefunc(x):
        res=sqrt(Lambda**(2-x)*D2+Gap**2)
        if ndim(x)==0:
            if x<=2:
                return D
            else:
                return res
        else:
            res[x<=2]=D
            return res
    return scalefunc

class DiscHandler(object):
    '''
    Handler for discretization of hybridization function.

    token:
        the token of this handler, which is a string for saving/loading data.
    Lambda:
        scaling factor.
    N:
        the maximum discretization index.
    D:
        the band range, [-1,1] for default.
    z:
        number of z.
    Gap:
        the gap range.
    '''
    def __init__(self,token,Lambda,N,D=[-1.,1.],Gap=[0.,0.]):
        self.token=token
        self.Lambda=Lambda
        self.N=N
        if ndim(D)==0:
            D=[-D,D]
        if ndim(Gap)==0:
            Gap=[-Gap,Gap]
        self.D=D
        self.Gap=Gap

        self.nband=None
        self.rhofunc=None
        self.wlist=None
        self.rho_list=None
        self.rhoeval_list=None
        self.rhoeval_int_list=None     #integration over rho_list

    @property
    def unique_token(self):
        '''
        return a unique token, for saving/loading datas.
        '''
        return '%s_%.4f_%s'%(self.token,self.Lambda,self.N)

    def set_rhofunc(self,rhofunc,Nw=50001):
        '''
        rhofunc:
            the hybridization function.
        Nw:
            the number of ws for rho(w).
        '''
        d0=rhofunc(0)
        self.rhofunc=rhofunc
        if ndim(d0)==2:
            self.nband=len(d0)
        else:
            self.nband=1
        #[positive,negative]
        self.wlist=[linspace(-self.Gap[0],-self.D[0],Nw),linspace(self.Gap[1],self.D[1],Nw)]
        self.rho_list=[array([rhofunc(w) for w in wl]) for wl in self.wlist]
        self.rhoeval_list=[array([eigvalsh(rho) for rho in rhol]) for rhol in self.rho_list]
        self.rhoeval_int_list=[concatenate([zeros([1,self.nband]),cumtrapz(rhoevall,self.wlist[bindex],axis=0)],axis=0) for bindex,rhoevall in enumerate(self.rhoeval_list)]

    def get_wxfunc_singleband(self,scalefunc,sgn,bandindex=0):
        '''
        Get weight function w(x), which is equal to \int^e(x)_e(x+1) d(w,bandindex) dw

        scalefunc:
            the function for scale ticks.
        sgn:
            specify the branch.
        bandindex:
            the bandindex.
        *return*:
            weight of hybridization function w(x) at interval \epsilon(x)~\epsilon(x+1)
        '''
        branchindex=(1+sgn)/2
        wlist=self.wlist[branchindex]
        RL=self.rhoeval_int_list[branchindex][:,bandindex]
        Rfunc=interp1d(wlist,RL)
        wxfunc=lambda x:Rfunc(scalefunc(x))-Rfunc(scalefunc(x+1))
        return wxfunc

    def get_efunc_singleband(self,scalefunc,sgn,bandindex=0,Nx=500000,rk=False):
        '''
        get representative Energy function e(x) for specific band and branch -the python version.

        scalefunc:
            a function of discretization points $\epsilon(x)$.
        sgn:
            the positive branch if sgn>0, else the negative one.
        bandindex:
            the band index.
        Nx:
            the number of samples in x-space for integration.
        rk:
            use Ronge-Kutta if True(in this version, it's better to set False).
        *return*:
            a function of representative energy e(x) for specific band and branch.
        '''
        xlist=linspace(1,self.N+2,Nx)
        wlist=self.wlist[(1+sgn)/2]
        rhoeval_int_list=self.rhoeval_int_list[(1+sgn)/2][:,bandindex]
        Rfunc=interp1d(wlist,rhoeval_int_list)
        iRfunc=interp1d(rhoeval_int_list,wlist)
        Rmax=iRfunc.x.max()
        Rmin=iRfunc.x.min()
        RRlist=append([0],cumtrapz(Rfunc(scalefunc(xlist)),xlist))
        RRfunc=interp1d(xlist,RRlist)
        def efunc(x):
            Rmean=RRfunc(x+1)-RRfunc(x)
            Rmean=min(Rmean,Rmax)
            Rmean=max(Rmean,Rmin)
            return iRfunc(Rmean)
        return efunc

    def get_Efunc(self,Efuncs):
        '''
        Get the Efunc for multi-band system.

        Efuncs:
            energy functions for individual bands.
        *return*:
            a function of representative energy E(x).
        '''
        def Efunc(x):
            E=diag([Efuncs[i](x) for i in xrange(self.nband)])
            return E
        return Efunc

    def get_Tfunc(self,wxfuncs,efuncs):
        '''
        Get the hopping term for multi-band system.

        wxfuncs:
            functions of weights for individual bands, t=sqrt(w) in 1-band system and this is for multi-band.
        efuncs:
            functions of representative energies e(x) for each band.
        *return*:
            a function of representative hopping terms T(x).
        '''
        if self.nband==1:
            raise Exception('Error','Using multi-band Tfunc for single band model.')
        def Tfunc(x):
            Ul=[]
            ei_old=2.
            for i in xrange(self.nband):
                ei=efuncs[i](x)
                #The degeneracy must be handled correctly!! 
                #This step is a must(not only for performance).
                if abs(ei_old-ei)>1e-13:
                    Ui=eigh(self.rhofunc(ei))[1]
                    ei_old=ei
                Ul.append(Ui[:,i:i+1])
            Ux=concatenate(Ul,axis=1)
            Td=concatenate([sqrt(wxfuncs[i](x))*Ux[:,i:i+1] for i in xrange(self.nband)],axis=1)
            T=Td.conj().T
            return T
        return Tfunc

    def check_mapping_eval(self,rhofunc,Efunc,Tfunc,scalefunc,sgn,Nx=1000,Nw=200,smearing=0.02):
        '''
        check the mapping quality by eigenvalues - the multiple-band Green's function version.

        rhofunc:
            the original hybridization function.
        Efunc/Tfunc:
            the representative energy/hopping term as a function of x.
        scalefunc:
            scale function.
        sgn:
            the branch.
        Nx/Nw:
            number of samples in x(index)- and w(frequency)- space.
        smearing:
            smearing constant.
        '''
        print 'Start checking the `%s` branch of discretized model!'%('positive' if sgn==1 else 'negative')
        Nx=Nx
        nband=self.nband
        D=sgn*self.D[(1+sgn)/2]
        Gap=sgn*self.Gap[(1+sgn)/2]
        ion()
        xlist=linspace(1,self.N+1,Nx)
        w0l=linspace(0,Gap-1e-4,20) if Gap>0 else [0]
        wlist=append(w0l,exp(linspace(log(self.Lambda**-(self.N+1)),log(D-Gap),Nw))+Gap)
        wl_sgn=sgn*wlist
        filename='data/checkmapping_%s_%s'%(self.unique_token,sgn)

        Tlist=[Tfunc(x) for x in xlist]
        Elist=[Efunc(x) for x in xlist]
        colormap=cm.rainbow(linspace(0,0.8,4))
        t0=time.time()
        GL=trapz([[dot(Tlist[i].T.conj(),dot(H2G(Elist[i],w=w,geta=smearing*max(1e-16,w-Gap)),Tlist[i])) for w in wlist] for i,x in enumerate(xlist)],xlist,axis=0)
        print time.time()-t0
        AL=1j*(GL-transpose(GL,axes=(0,2,1)).conj())/(pi*2.)
        AV=array([eigvalsh(A) for A in AL])
        odatas=array([eigvalsh(rhofunc(sgn*w)) for w in wlist])
        savetxt(filename+'.dat',concatenate([wl_sgn[:,newaxis],odatas,AV],axis=1))
        plts=[]
        for i in xrange(nband):
            plts+=plot(wl_sgn,odatas[:,i],lw=3,color=colormap[i])
        for i in xrange(nband):
            sct=scatter(wl_sgn,AV[:,i],s=30,edgecolors=colormap[i],facecolors='none')
            plts.append(sct)
        legend(plts,[r'$\rho_%s$'%i for i in xrange(nband)]+[r"$\rho'_%s$"%i for i in xrange(nband)],ncol=2)
        xlabel('$\\omega$',fontsize=16)
        print 'Done, Press `c` to save figure and continue.'
        pdb.set_trace()
        savefig(filename+'.png')


    def check_mapping_pauli(self,rhofunc,Efunc,Tfunc,scalefunc,sgn,Nx=1000,Nw=200,smearing=0.02):
        '''
        check the mapping quality by Pauli decomposition - only 2-band Green's function is allowed.

        rhofunc:
            the original hybridization function.
        Efunc/Tfunc:
            the representative energy/hopping term as a function of x.
        scalefunc:
            scale function.
        sgn:
            the branch.
        Nx/Nw:
            number of samples in x(index)- and w(frequency)- space.
        smearing:
            smearing constant.
        '''
        print 'Starting checking the `%s` branch of discretized model!'%('positive' if sgn==1 else 'negative')
        Nx=Nx
        if self.nband!=2:
            raise Exception('function @check_mapping_pauli is designed for 2 band bath, but got %s.'%self.nband)
        D=sgn*self.D[(1+sgn)/2]
        Gap=sgn*self.Gap[(1+sgn)/2]
        ion()
        xlist=linspace(1,self.N+1,Nx)
        w0l=linspace(0,Gap-1e-4,20) if Gap>0 else [0]
        wlist=append(w0l,exp(linspace(log(self.Lambda**-(self.N+1)),log(D-Gap),Nw))+Gap)
        wl_sgn=sgn*wlist
        filename='data/checkmapping_%s_%s'%(self.unique_token,sgn)

        Tlist=[Tfunc(x) for x in xlist]
        Elist=[Efunc(x) for x in xlist]
        colormap=cm.rainbow(linspace(0,0.8,4))
        GL=trapz([[dot(Tlist[i].T.conj(),dot(H2G(Elist[i],w=w,geta=smearing*max(1e-16,w-Gap)),Tlist[i])) for w in wlist] for i,x in enumerate(xlist)],xlist,axis=0)
        AL=1j*(GL-transpose(GL,axes=(0,2,1)).conj())/(pi*2.)
        AV=array([s2vec(A) for A in AL]).real
        odatas=array([s2vec(rhofunc(sgn*w)) for w in wlist]).real
        savetxt(filename+'.dat',concatenate([wl_sgn[:,newaxis],odatas,AV],axis=1))
        for i in xrange(4):
            plot(wl_sgn,odatas[:,i],lw=3,color=colormap[i])
        for i in xrange(4):
            scatter(wl_sgn,AV[:,i],s=30,edgecolors=colormap[i],facecolors='none')
        legend(['$\\rho_0$','$\\rho_x$','$\\rho_y$','$\\rho_z$',"$\\rho^\\prime_0$","$\\rho^\\prime_x$","$\\rho^\\prime_y$","$\\rho^\\prime_z$"],ncol=2)
        xlabel('$\\omega$',fontsize=16)
        print 'Done, Press `c` to save figure and continue.'
        pdb.set_trace()
        savefig(filename+'.png')

    def quick_map(self,tick_type='log',Nx=500000):
        '''
        Perform quick mapping(All in one suit!) for nband x nband hybridization matrix.

        tick_type:
            the type of tick, `log`->logarithmic tick, `sclog`->logarithmic ticks suited for superconductor.
        Nx:
            the number of samples for integration over rho(epsilon(x)).

        *return*:
            a super tuple of functions -> (scalefunc_positve,Efunc_positive,Tfunc_positive),(scalefunc_negative,Efunc_negative,Tfunc_negative)
        '''
        if tick_type=='log':
            print 'Using Logarithmic ticks.'
            scaletick_generator=get_scalefunc_log
        elif tick_type=='sclog':
            print 'Using Logarithmic ticks designed for superconductor.'
            scaletick_generator=get_scalefunc_sclog
        else:
            raise Exception('Error','Undefined tick type %s'%tick_type)
        datas=[]
        for sgn in [1,-1]:
            D=sgn*self.D[(1+sgn)/2]
            Gap=sgn*self.Gap[(1+sgn)/2]
            scalefunc=scaletick_generator(self.Lambda,D=D,Gap=Gap,sgn=sgn)
            wxfuncl=[];efuncl=[]
            for i in xrange(self.nband):
                print 'Multiple Band, For the `%s` branch/band %s ->'%('positive' if sgn>0 else 'negative',i)
                print 'Getting w(x) function for band %s.'%i
                wxfunc=self.get_wxfunc_singleband(scalefunc=scalefunc,sgn=sgn,bandindex=i)
                print 'Getting e(x) function for band %s.'%i
                Efunc=self.get_efunc_singleband(scalefunc,sgn=sgn,bandindex=i,Nx=Nx)
                wxfuncl.append(wxfunc);efuncl.append(Efunc)
            print 'Getting representative energies/hopping terms -> E(x)/T(x) ...'
            Efunc=self.get_Efunc(efuncl)
            Tfunc=self.get_Tfunc(wxfuncl,efuncl)
            datas.append((scalefunc,Efunc,Tfunc))
        print 'Discretization completed!'
        return datas

    def get_discrete_model(self,funcs,z=1.,append=False):
        '''
        get a discrete set of models for specific zs.

        funcs:
            a super tuple -> (scalefunc_positve,Efunc_positive,Tfunc_positive),(scalefunc_negative,Efunc_negative,Tfunc_negative)
        z:
            twisting parameters, scalar or 1D array.
        append:
            append model informations instead of generating one.
        *return*:
            a discrentized model - DiscModel instance.
        '''
        filename='data/scale-%s'%(self.unique_token)
        if ndim(z)==0:
            z=array([z])
        nz=len(z)

        if append:
            Elist=load(filename+'.Elist.npy')
            Tlist=load(filename+'.Tlist.npy')
            if Elist.shape[1]!=nz:
                raise Exception('Error','Fail to load scale data, z-number do not match!')
        else:
            (scalefunc,Efunc,Tfunc),(scalenfunc,Enfunc,Tnfunc)=funcs
            ticks=array([arange(1+iz,self.N+1+iz) for iz in z]).T
            Tlist=concatenate([[[Tfunc(x) for x in xi] for xi in ticks],[[Tnfunc(x) for x in xi] for xi in ticks]],axis=0)
            Elist=concatenate([[[Efunc(x) for x in xi] for xi in ticks],[[-Enfunc(x) for x in xi] for xi in ticks]],axis=0)
            save(filename+'.Elist.npy',Elist)
            save(filename+'.Tlist.npy',Tlist)
        return DiscModel(Elist,Tlist,z)

class DiscModel(object):
    '''
    discrete model.

    Elist/Tlist:
        a list of on-site energies and hopping terms. The shape is (2N,nz,nband,nband)
    z:
        the twisting parameters.
    '''
    def __init__(self,Elist,Tlist,z=1.):
        if ndim(z)==0:
            self.z=array([z])
        elif ndim(z)==1:
            self.z=array(z)
        else:
            raise Exception('z must be a list or a scalar!')
        if any(z)>1. or any(z<=0.):
            raise Exception('z must >0 and <=1 !')
        self.Tlist=Tlist
        self.Elist=Elist

    @property
    def nz(self):
        '''number of twisting parameters.'''
        return len(self.z)

    @property
    def N(self):
        '''number of particles for each branch(positive or negative).'''
        return self.Elist.shape[0]/2

    @property
    def nband(self):
        '''number of bands.'''
        return self.Elist.shape[-1]

