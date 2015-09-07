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

        self.nband=2
        self.wlist=None
        self.rholist=None
        self.rholist_closed=None

    def __get_rhointerpolator__(self,branch,which,gapless=False):
        '''
        get a function of rho(w).

        branch:
            the `P`ositive or `N`egative branch or `B`oth.
        which:
            `evals` -> the eigenvalues
            `pauli` -> pauli decomposition.
        gapless:
            Gap closed or not.
        '''
        if self.rholist is None:
            raise Exception('Please specify rhofunc(with DiscHandler.set_rhofunc) first!')
        if gapless:
            rholist=self.rholist_closed
        else:
            rholist=self.rholist
        #get interpolator
        zpoint=searchsorted(self.wlist,0)
        if branch=='B':
            wlist=self.wlist
            rholist=rholist
        elif branch=='P':
            wlist=self.wlist[zpoint-1:]
            rholist=rholist[zpoint-1:]
        elif branch=='N':
            wlist=-self.wlist[zpoint::-1]
            rholist=rholist[zpoint::-1]
        else:
            raise Exception('Error','branch unknown %s.'%branch)
        if self.nband==1:
            return interp1d(wlist,rholist)
        elif which=='pauli':
            rhol=array([s2vec(rho) for rho in rholist]).real
            return [interp1d(wlist,rhol[:,i]) for i in xrange(4)]
        elif which=='evals':
            rhol=array([eigvalsh(rho) for rho in rholist])
            return [interp1d(wlist,rhol[:,i]) for i in xrange(self.nband)]
        else:
            raise Exception('Error','undefined case %s'%which)

    @property
    def unique_token(self):
        '''
        return a unique token, for saving/loading datas.
        '''
        return '%s_%.4f_%s'%(self.token,self.Lambda,self.N)

    def set_rhofunc(self,rhofunc,NW=50001):
        '''
        rhofunc:
            the hybridization function.
        NW:
            the number of ws for rho(w).
        '''
        d0=rhofunc(0)
        if ndim(d0)==0 or len(d0)!=2:
            raise Exception('Sorry, this program is for 2x2 rho(w) function only.')
        self.wlist=linspace(self.D[0],self.D[1],NW)
        self.rholist=array([rhofunc(w) for w in self.wlist])
        rclist=[]
        for w in self.wlist:
            if w>=0 and w<=self.D[1]:
                Gap=self.Gap[1]
                D=self.D[1]
                w2=w*(D-Gap)/D+Gap
                rc=rhofunc(w2)
            elif w<0 and w>=self.D[0]:
                Gap=-self.Gap[0]
                D=-self.D[0]
                w2=w*(D-Gap)/D-Gap
                rc=rhofunc(w2)
            else:
                rc=0. if self.nband==1 else zeros([self.nband,self.nband])
            rclist.append(rc)
        self.rholist_closed=array(rclist)

    def get_scalefunc_log(self,sgn):
        '''
        get logarithmic scale tick function.
        \epsilon(x)=\Lambda^(2-x)

        sgn:
            return the scale function for the positive branch if sgn>0, else the negative one.
        *return*:
            a scale function \epsilon(x).
        '''
        if sgn>0:
            shift=self.Gap[1]
            D=self.D[1]
        else:
            shift=-self.Gap[0]
            D=-self.D[0]
        def scalefunc(x):
            res=self.Lambda**(2-x)*(D-shift)+shift
            if ndim(x)==0:
                if x<=2:
                    return D
                else:
                    return res
            else:
                res[x<=2]=D
                return res
        return scalefunc

    def get_scalefunc_sclog(self,sgn):
        '''
        get logarithmic scale tick function suited for superconductor(logarithmic for normal part).

        sgn:
            return the scale function for the positive branch if sgn>0, else the negative one.
        *return*:
            a scale function \epsilon(x).
        '''
        if sgn>0:
            shift=self.Gap[1]
            D=self.D[1]
        else:
            shift=-self.Gap[0]
            D=-self.D[0]
        D2=D**2-shift**2
        def scalefunc(x):
            res=sqrt(self.Lambda**(2-x)*D2+shift**2)
            if ndim(x)==0:
                if x<=2:
                    return D
                else:
                    return res
            else:
                res[x<=2]=D
                return res
        return scalefunc

    def get_wxfunc(self,scalefunc,sgn,bandindex=0,N=10000):
        '''
        Get weight function w(x), which is equal to \int^e(x)_e(x+1) d(w,bandindex) dw

        scalefunc:
            the function for scale ticks.
        sgn:
            specify the branch.
        bandindex:
            the bandindex.
        N:
            the number of samples.
        *return*:
            weight of hybridization function w(x) at interval \epsilon(x)~\epsilon(x+1)
        '''
        Gap=abs(self.Gap[(1+sgn)/2])
        D=abs(self.D[(1+sgn)/2])
        
        dfunc=self.__get_rhointerpolator__(which='evals',branch='P' if sgn>0 else 'N',gapless=False)[bandindex]
        wlist=linspace(Gap,D,N)
        int_dfunc=append([0],cumtrapz(dfunc(wlist),wlist))
        DF=interp1d(wlist,int_dfunc)
        wxfunc=lambda x:DF(scalefunc(x))-DF(scalefunc(x+1))
        testl=linspace(1,1.003,10)
        return wxfunc

    def get_efunc(self,wxfunc,sgn,bandindex=0,N=100000,rk=False):
        '''
        get representative Energy function e(x) for specific band and branch -the python version.

        wxfunc:
            a function of w(x), which is equal to t(x)^2.
        sgn:
            the positive branch if sgn>0, else the negative one.
        bandindex:
            the band index.
        N:
            the number of samples for integration.
        rk:
            use Ronge-Kutta if True(in this version, it's better to set False).
        *return*:
            a function of representative energy e(x) for specific band and branch.
        '''
        Nmid=5
        shift=False
        xlist=linspace(1,self.N+1,N)
        if sgn>0:
            D=self.D[1]
            Gap=self.Gap[1]
            wlist=linspace(Gap,D,N)
        else:
            D=sgn*self.D[0]
            Gap=sgn*self.Gap[0]
            wlist=linspace(Gap,D,N)
        revals_func=self.__get_rhointerpolator__(branch='P' if sgn==1 else 'N',which='evals')[bandindex]
        if rk:
            #ronge_kutta approach
            Elist=ode_ronge_kutta(func=lambda x,y:-wxfunc(x)/revals_func(y),y0=D,tlist=xlist,integrator='dop853',atol=1e-20)
        else:
            #direct integrate approach
            wxl=wxfunc(xlist)
            WL=append([0],cumtrapz(wxfunc(xlist),xlist))
            Wfunc=interp1d(xlist,WL)
            #integrate over rho(w) and get the inverse function.
            revals=revals_func(wlist)
            RL=append([0],cumtrapz(revals,wlist))
            iRfunc=interp1d(RL[-1]-RL,wlist)

            print 'Difference of W[-1] and RL[-1]:',RL[-1]-WL[-1]
            if WL[-1]>RL[-1]:
                raise Exception('precision error @get_efunc')
            Elist=iRfunc(Wfunc(xlist))
        Efunc=interp1d(xlist,Elist)
        Emin=Elist[-1]
        print 'Emin - %s'%Emin
        if Emin<Gap:
            print 'Error, E < Gap detected! Please improve the precision of Mapping!'
            ion()
            plot(xlist,Efunc(xlist))
            pdb.set_trace()
            raise Exception('Error','E < Gap detected! Please Improve the accuracy of Mapping!')
        return Efunc

    def get_Ufunc2(self,Efuncs,sgn):
        '''
        get the U(x) function.

        Efuncs:
            the energy functions.
        *return*:
            a function of matrix U(x).
        '''
        if self.nband!=len(Efuncs):
            raise Exception('Wrong number of Efuncs.')
        if self.rholist is None:
            raise Exception('Please specify rhofunc(with DiscHandler.set_rhofunc) first!')
        dv=array([s2vec(d).real for d in self.rholist]).real
        dvfunc=[interp1d(self.wlist,dv[:,i]) for i in xrange(4)]
        def Ufunc(x):
            el=array([sgn*Efunc(x) for Efunc in Efuncs])
            vl=array([dvfunc[i](el) for i in xrange(4)])
            Ul=[eigh_pauliv_npy(*vl[:,i])[1] for i in xrange(self.nband)]
            return concatenate([Ul[iband][:,iband:iband+1] for iband in xrange(self.nband)],axis=1)
        return Ufunc

    def get_Efunc2(self,Efuncs,Ufunc):
        '''
        Get the Efunc for multi-band system.

        Efuncs:
            energy functions for individual bands.
        Ufunc:
            U function.
        *return*:
            a function of representative energy E(x).
        '''
        def Efunc2(x):
            E=diag([Efuncs[i](x) for i in xrange(self.nband)])
            return E
        return Efunc2

    def get_Tfunc2(self,wxfuncs,Ufunc):
        '''
        Get the hopping term for multi-band system.

        wxfuncs:
            functions of weights for individual bands, t=sqrt(w) in 1-band system and this is for multi-band.
        Ufunc:
            U function.
        *return*:
            a function of representative hopping terms T(x).
        '''
        if self.nband==1:
            raise Exception('Error','Using multi-band Tfunc for single band model.')
        def Tfunc2(x):
            Ux=Ufunc(x)
            Td=concatenate([sqrt(wxfuncs[i](x))*Ux[:,i:i+1] for i in xrange(self.nband)],axis=1)
            T=Td.conj().T
            return T
        return Tfunc2

    def check_mapping2(self,rhofunc,Efunc,Tfunc,scalefunc,sgn,Nx=1000,Nw=200,smearing=0.02):
        '''
        check the mapping quality - the multi-band Green's function version.

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
        NN=Nx
        D=sgn*self.D[(1+sgn)/2]
        Gap=sgn*self.Gap[(1+sgn)/2]
        ion()
        xlist=linspace(1,self.N+1,NN)
        w0l=linspace(0,Gap-1e-4,20) if Gap>0 else [0]
        wlist=append(w0l,exp(linspace(log(self.Lambda**-(self.N+1)),log(D-Gap),Nw))+Gap)
        wl_sgn=sgn*wlist
        filename='data/checkmapping_%s_%s'%(self.unique_token,sgn)

        Tlist=[Tfunc(x) for x in xlist]
        Elist=[Efunc(x) for x in xlist]
        colormap=cm.rainbow(linspace(0,0.8,4))
        colormap2=cm.rainbow(linspace(0.2,1.,4))
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

    def quick_map2(self,tick_type='log',NN=1000000):
        '''
        Perform quick mapping(All in one suit!) for 2 x 2 hybridization matrix.

        tick_type:
            the type of tick, `log`->logarithmic tick, `sclog`->logarithmic ticks suited for superconductor.
        NN:
            the number of samples for integration over rho(w).

        *return*:
            a super tuple -> (scalefunc_positve,Efunc_positive,Tfunc_positive),(scalefunc_negative,Efunc_negative,Tfunc_negative)
        '''
        if tick_type=='log':
            print 'Using Logarithmic ticks.'
            scaletick_generator=self.get_scalefunc_log
        elif tick_type=='sclog':
            print 'Using Logarithmic ticks designed for superconductor.'
            scaletick_generator=self.get_scalefunc_sclog
        else:
            raise Exception('Error','Undefined tick type %s'%tick_type)
        datas=[]
        for sgn in [1,-1]:
            scalefunc=scaletick_generator(sgn=sgn)
            wxfuncl=[];Efuncl=[]
            for i in xrange(self.nband):
                print 'Multiple Band, For the %s branch/%s band ->'%(sgn,i)
                print 'Getting w(x) function.'
                wxfunc=self.get_wxfunc(scalefunc=scalefunc,sgn=sgn,bandindex=i,N=NN)
                print 'Getting E(x) function.'
                Efunc=self.get_efunc(wxfunc,sgn=sgn,bandindex=i,N=NN)
                wxfuncl.append(wxfunc);Efuncl.append(Efunc)
            print 'Getting U(x)/E(x)/T(x) ...'
            Ufunc=self.get_Ufunc2(Efuncl,sgn)
            Efunc=self.get_Efunc2(Efuncl,Ufunc)
            Tfunc=self.get_Tfunc2(wxfuncl,Ufunc)
            datas.append((scalefunc,Efunc,Tfunc))
        print 'Discretization completed!'
        return datas

    def get_discrete_model(self,funcs,z=1.,append=False):
        '''
        get a discrete set of models for specific zs.

        funcs:
            a super tuple -> (scalefunc_positve,Efunc_positive,Tfunc_positive),(scalefunc_negative,Efunc_negative,Tfunc_negative)
        z:
            z or a list of zs.
        append:
            append model informations instead of generating one.
        *return*:
            a discrentized model - DiscModel instance.
        '''
        NN=200000
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

