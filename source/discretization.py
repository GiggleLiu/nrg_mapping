'''
Discretize the continuous hybridization function into a discretized model(DiscModel).
checking method is also provided.
'''

from numpy import *
from matplotlib.pyplot import *
from matplotlib import cm
from scipy.interpolate import InterpolatedUnivariateSpline,interp1d
from scipy.integrate import quadrature,cumtrapz,trapz,simps,quad
from numpy.linalg import eigh,inv,eigvalsh,norm
import pdb,time,warnings

from utils import sx,sy,sz,s2vec,H2G,plot_pauli_components
from discmodel import DiscModel
from ticklib import get_ticker
#from utils import exinterp as interp1d

__all__=['DiscHandler','SingleBandDiscHandler','MultiBandDiscHandler',\
        'quick_map','check_disc']

class DiscHandler(object):
    '''
    Base class of handlers for discretization of hybridization function.
    '''
    def get_TEfunc(self,sgn,xmax,Nx=500000,**kwargs):
        '''
        Get representative hopping T(x) and onsite-energy function E(x) for specific branch.
        '''
        raise NotImplementedError()

class SingleBandDiscHandler(DiscHandler):
    '''
    Handler for discretization of hybridization function.

    Construct:
        SingleBandDiscHandler(wlist,rholist,tickers),
        with the wlist, rholist containing both positive and negative branches.

    Attributes:
        :nband: integer, the number of bands with respect to hybridization function, readonly.
        :wlists: len-2 tuple (wlist_neg,wlist_pos) defining the frequency space.
        :rholists: len-2 tuple (rholist_neg,rholist_pos) of the hybridization function defined on `wlists`.
        :int_rholists: len-2 tuple (rholist_pos_int,rholist_neg_int), integration of rholists over `wlist`.
        :tickers: len-2 tuple with elements function/<Ticker> of discretization points $\epsilon(x)$.
        :wlist,rholist: 1D array,ndarray, Whole frequency space and rho(w) defined on it(readonly).
        :D: len-2 list with integer elements, the band range(readonly).
    '''
    def __init__(self,wlist,rholist,tickers):
        assert(all(rholist>=0))
        rholist=array(rholist)

        pmask=wlist>0
        self.wlists=[-wlist[~pmask][::-1],wlist[pmask]]
        self.rholists=[rholist[~pmask][::-1],rholist[pmask]]
        self.int_rholists=[append([0],cumtrapz(rholist,wlist)) for rholist,wlist in zip(self.rholists,self.wlists)]
        self.tickers=tickers

    @property
    def D(self):
        '''The band width.'''
        return self.wlist[0],self.wlist[-1]

    @property
    def wlist(self):
        '''The whole frequency space.'''
        return append(-self.wlists[0][::-1],self.wlists[1])

    @property
    def rholist(self):
        '''The rho(w) defined on whole frequency space.'''
        return append(self.rholists[0][::-1],self.rholists[1])

    def get_TEfunc(self,sgn,xmax,Nx=500000):
        '''
        Get hopping function T(x) and on-site energy function E(x).

        Parameters:
            :sgn: 0/1 to specify the negative/positive branch.
            :xmax,Nx: The upper limit of x, and the number of samples in x-space for integration and interpolation.

        Return:
            A tuple of functions of representative hopping T(x) and on-site energy E(x) for specific band and branch.
        '''
        assert(sgn==0 or sgn==1)
        ticker=self.tickers[sgn]
        wlist=self.wlists[sgn]

        #get the hopping function
        RL=self.int_rholists[sgn]
        Rfunc=interp1d(wlist,RL)
        Tfunc=lambda x:sqrt(Rfunc(ticker(x))-Rfunc(ticker(x+1)))

        #get the energy function
        xlist=linspace(1,xmax,Nx)
        iRfunc=interp1d(RL,wlist)
        #RRlist=append([0],cumtrapz(Rfunc(ticker(xlist)),xlist))
        #RRfunc=interp1d(xlist,RRlist)
        #Efunc=lambda x:(2*sgn-1)*(iRfunc(RRfunc(x+1)-RRfunc(x)))

        #change the integration order to avoid big+small.
        RRlist=append([0],cumtrapz(Rfunc(ticker(xlist[::-1])),xlist[::-1]))
        RRfunc=interp1d(xlist,RRlist[::-1])
        Efunc=lambda x:(2*sgn-1)*(iRfunc(RRfunc(x+1)-RRfunc(x)))

        return Tfunc,Efunc

class MultiBandDiscHandler(DiscHandler):
    '''
    Handler for discretization of hybridization function.

    Construct: DiscHandler(rhofunc,wlist,tickers,autofix=1e-5)

    *autofix* is the the negative tolerence for eigen values of rho(w), above which it will be automatically fixed to 0.

    Attributes:
        :tickers: len-2 list with elements <Ticker>/function, The tick generator, a function or a callable instance.
        :wlist: 1D array, The frequency space.
        :rholist: ndarray, The hybridization function defined on `wlist`.
        :single_band_handlers: <SingleBandDiscHandler>, A list of <SingleBandDiscHandler> instances to handler mapping at each eigen-value channel.
        :nband: integer, The number of bands with respect to hybridization function 
    '''
    def __init__(self,rhofunc,wlist,tickers,autofix=1e-5):
        self.rhofunc=rhofunc
        pmask=wlist>0
        self.wlist=wlist
        self.tickers=tickers

        self.rholist=array([rhofunc(w) for w in wlist])
        rhoeval_list=array([eigvalsh(rho) for rho in self.rholist])

        if any(array(rhoeval_list)<0):
            for i in xrange(2):
                rl=rhoeval_list[i]
                print rhoeval_list[i].min()
                if rl.min()>autofix:
                    print 'Fixing minus eigenvalue of rho(w) caused by numerical error!'
                    rl[rl<1e-20]=1e-20
                else:
                    raise Exception('Negative eigenvalue of rho(w) found! check your hybridization function!')

        assert(ndim(self.rholist)>1)
        nband=self.nband
        self.single_band_handlers=[]
        for i in xrange(self.nband):
            self.single_band_handlers.append(SingleBandDiscHandler(wlist,rhoeval_list[:,i],tickers=self.tickers))

    @property
    def nband(self):
        '''The number of bands'''
        return self.rholist.shape[-1]

    def get_TEfunc(self,sgn,xmax,Nx):
        '''
        Get the representative hopping and on-site energy Tfunc/Efunc for multi-band system.

        Parameters:
            :sgn: integer, 0/1 to specify the negative/positive branch.
            :xmax,Nx: interger, The upper limit of x, and the number of samples in x-space for integration and interpolation.

        Return:
            A tuple of functions of representative hopping and energy (T(x),E(x)).
        '''
        nband=self.nband
        tfuncs,efuncs=[],[]
        for i in xrange(nband):
            tf,ef=self.single_band_handlers[i].get_TEfunc(sgn=sgn,xmax=xmax,Nx=Nx)
            tfuncs.append(tf)
            efuncs.append(ef)

        #get Efunc
        Efunc=lambda x:diag([efuncs[i](x) for i in xrange(nband)])

        #get Tfunc
        def Tfunc(x):
            Ul=[]
            ei_old=Inf
            for i in xrange(self.nband):
                ei=efuncs[i](x)
                #The degeneracy must be handled,
                #Checking degeneracy is a critical step to ensure the correctness.
                if abs(ei_old-ei)>1e-12:
                    Ui=eigh(self.rhofunc(ei))[1]
                    ei_old=ei
                Ul.append(Ui[:,i:i+1])
            Ux=concatenate(Ul,axis=1)
            Td=concatenate([tfuncs[i](x)*Ux[:,i:i+1] for i in xrange(self.nband)],axis=1)
            T=Td.conj().T
            return T
        return Tfunc,Efunc

def check_disc(rhofunc,discmodel,wlist,smearing=0.02,mode='eval'):
    '''
    check the discretization quality by eigenvalues - the multiple-band Green's function version.

    Parameters:
        :rhofunc: function, The original hybridization function.
        :discmodel: <DiscModel>, The discretized model.
        :wlist: 1D array, the frenquncy space.
        :smearing: float, smearing constant.
    '''
    print 'Start checking the mapping of discretized model!'
    t0=time.time()
    nband=discmodel.nband
    N=discmodel.N_pos
    Elist,Tlist=discmodel.Elist,discmodel.Tlist
    zlist=discmodel.z
    nz=len(zlist)
    assert(mode=='pauli' or mode=='eval')
    GL=array([sum([sum([dot(Ti.T.conj(),dot(H2G(Ei,w=w,geta=smearing*1./nz*max(1e-3,abs(w))),Ti))\
            for Ei,Ti in zip(Elist[:,i],Tlist[:,i])],axis=0) for i in xrange(nz)],axis=0)/nz for w in wlist])
    if ndim(GL)==1:  #the scalar case.
        AL=-GL.imag[:,newaxis]/pi
    else:
        AL=1j*(GL-transpose(GL,axes=(0,2,1)).conj())/(pi*2.)
    odatas=array([rhofunc(w) for w in wlist])
    if mode=='pauli':
        if nband!=2:
            raise Exception('Check pauli is for 2 band system, not %s!'%nband)
        plot_pauli_components(wlist,odatas,label='rho')
        plot_pauli_components(wlist,AL,method='scatter',label=r"\rho'")
    else:
        AV=array([eigvalsh(A) for A in AL]) if nband>1 else AL.reshape(AL.shape[:2])
        odatas=array([eigvalsh(o) for o in odatas]) if nband>1 else odatas.reshape(AL.shape[:2])
        colormap=cm.rainbow(linspace(0,0.8,4))
        plts=[]
        for i in xrange(nband):
            plts+=plot(wlist,odatas[:,i],lw=3,color=colormap[i])
        for i in xrange(nband):
            sct=scatter(wlist,AV[:,i],s=30,edgecolors=colormap[i],facecolors='none')
            plts.append(sct)
        legend(plts,[r'$\rho_%s$'%i for i in xrange(nband)]+[r"$\rho'_%s$"%i for i in xrange(nband)],ncol=2)
    xlabel('$\\omega$',fontsize=16)
    print 'Time Elapsed: %s s'%(time.time()-t0)

def quick_map(rhofunc,wlist,N,z=1.,Nx=500000,tick_params=None,autofix=1e-5):
    '''
    Perform quick mapping(All in one suit!) for nband x nband hybridization matrix.

    Parameters:
        :rhofunc: function, hybridization function.
        :wlist: 1D array, frequency space holding this hybridization function.
        :N: integer, number of discrete sites for each z number.
        :z: float/1D array, twisting parameters.
        :Nx: integer, number of samples for integration over rho(epsilon(x)).
        :tick_params: dict, parameters for ticks, the following keys could be available,

            * `tick_type` -> The type of ticks
                'log': logarithmic tick,
                'sclog': logarithmic ticks suited for superconductor.
                'adaptive': adaptive ticks(default).
                'linear': linear ticks.
                'adaptive_linear': adaptive linear ticks.
            * `r` -> Adaptive ratio for `adaptive`/`adaptive_linear` ticks(default: 1.0).
            * `wn` -> typical frequency to achieve the best fit for `ed` ticks(default: pi/200).
            * `Lambda` -> scaling factor for `log`, `adaptive`, `sclog` ticks(default: 2.0).
            * `Gap` -> gap region of the hybridization function, it is used in discretization schemes
            that do not allow zeros in hybridization function: `log`, `sclog`, `linear`(default: 0).

        :autofix: float, Automatically fix the occational small negative eigenvalue(>=autofix) of rho(w) to 0.

    Return:
        Tuple of (<Ticker>,<DiscModel>).
    '''
    D=[wlist[0],wlist[-1]]
    z=array(z)
    if tick_params is None: tick_params={}
    if not (all(z>0) and all(z<=1)): raise ValueError('twisting parameter out of range!')
    tick_type=tick_params.get('tick_type','adaptive')
    r=tick_params.get('r',1.)
    wn=tick_params.get('wn',pi/200)
    Lambda=tick_params.get('Lambda',2.)
    Gap=tick_params.get('Gap',0)
    if ndim(Gap)==0:Gap=[-Gap,Gap]
    rholist=array([rhofunc(w) for w in wlist])
    pmask=wlist>0
    wlists=[-wlist[~pmask][::-1],wlist[pmask]]
    branches=[0,1]  #negative and positive branches

    if ndim(rholist)==1:
        is_scalar=True
        weights=rholist
    else:
        is_scalar=False
        weights=(rholist*rholist.swapaxes(1,2)).sum(axis=(1,2)).real
    weights=[weights[~pmask][::-1],weights[pmask]]

    t0=time.time()
    print 'Start discretization. Using %s discretization points(ticks).'%tick_type
    tickers=[get_ticker(tick_type,r=r,Lambda=Lambda,D=abs(D[sgn]),Gap=abs(Gap[sgn]),\
            wlist=wlists[sgn],rholist=weights[sgn]) for sgn in branches]

    #check for precision
    if -wlist[wlist<0].max()>tickers[0](N+1) or wlist[wlist>0].min()>tickers[1](N+1):
        warnings.warn('Precision Warning, the minimum energy scale `w0` is \
supposed to be smaller than %s.'%min(tickers[1](N+1),tickers[0](N+1)))

    if is_scalar:
        handler=SingleBandDiscHandler(wlist=wlist,rholist=rholist,tickers=tickers)
    else:
        handler=MultiBandDiscHandler(rhofunc=rhofunc,wlist=wlist,tickers=tickers,autofix=autofix)
    print 'Getting functions of representative energies and hoppings.'
    funcs=[handler.get_TEfunc(sgn=sgn,Nx=Nx,xmax=N+2) for sgn in branches]
    print 'Time Elapsed: %s s'%(time.time()-t0)
    print 'Done.'

    print 'Generating discrete model ...'
    if ndim(z)==0:
        z=array([z])
    nz=len(z)

    ticks=array([arange(1+iz,N+1+iz-0.1) for iz in z]).T
    Tlists=[array([[funcs[sgn][0](x) for x in xi] for xi in ticks]) for sgn in branches]
    Elists=[array([[funcs[sgn][1](x) for x in xi] for xi in ticks]) for sgn in branches]
    print 'Done.'
    model=DiscModel(Elists,Tlists,z)
    return tickers,model

