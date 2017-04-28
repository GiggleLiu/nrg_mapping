'''
The library for deciding discretization mesh points.
'''

from numpy import *
from numpy.linalg import norm
from scipy.integrate import cumtrapz
from scipy.interpolate import InterpolatedUnivariateSpline as spline1d
from scipy.interpolate import interp1d
from matplotlib.pyplot import *
import time,pdb

__all__=['Ticker','AdaptiveTickerBase','AdaptiveLinearTicker','AdaptiveLogTicker','LogTicker','ScLogTicker','LinearTicker','EDTicker','get_ticker']

class Ticker(object):
    '''
    Base class for discretization tick generator.
    Use Ticker(x) to get the tick position of x.

    Construct:
        Ticker(tp)

    Attributes:
        :tp: The type string of this tick class.
    '''
    def __init__(self,tp):
        self.tp=tp

    def __call__(self,x,**kwargs):
        '''Get the tick position at specific site index - x.'''
        raise Exception('Not Implemented!')

class LogTicker(Ticker):
    '''
    Logarithmic ticks.
    :math:`\epsilon(x)=\Lambda^{(2-x)}`

    Construct:
        LogTicker(Lambda,D,Gap=0.,N=Inf)

    Attributes:
        :Lambda: Scaling factor, a number between 1 and infinity(optimally choose as 1.5-10).
        :D/Gap: The bandwidth/Gap range, then the hybridization function is defined in the interval Gap <= abs(w) <= bandwidth.
    '''
    def __init__(self,Lambda,D,Gap=0.):
        super(LogTicker,self).__init__('log')
        self.Lambda=Lambda
        self.D=D
        self.Gap=Gap

    def __call__(self,x):
        D=self.D
        Gap=self.Gap
        Lambda=self.Lambda
        if ndim(x)==0:
            if x<=2:
                return D
            else:
                return Lambda**(2-x)*(D-Gap)+Gap
        else:
            res=D*ones(len(x),dtype='float64')
            fit_mask=x>2
            res[fit_mask]=Lambda**(2-x[fit_mask])*(D-Gap)+Gap
            return res
            #return concatenate([D*ones(len(xs)),Lambda**(2-xm)*(D-Gap)+Gap])

class ScLogTicker(Ticker):
    '''
    Logarithmic tick suited for superconductor(logarithmic for normal state).

    Construct:
        ScLogTicker(Lambda,D,Gap)

    Attributes:
        :Lambda: Scaling factor.
        :D/Gap: The bandwidth/Gap range.

    **Note:**
    It is in fact, a mimic for the discretization scheme given by JPSJ 61.3239
    '''
    def __init__(self,Lambda,D,Gap):
        super(ScLogTicker,self).__init__('sclog')
        self.Lambda=Lambda
        self.D=D
        self.Gap=Gap

    def __call__(self,x):
        D=self.D
        Gap=self.Gap
        Lambda=self.Lambda
        D0=sqrt(D**2-Gap**2)
        res=sqrt((Lambda**(2-x)*D0)**2+Gap**2)
        if ndim(x)==0:
            if x<=2:
                return D
            else:
                return sqrt((Lambda**(2-x)*D0)**2+Gap**2)
        else:
            #inds=searchsorted(x,2)
            #xs,xm=split(x,(inds,))
            #return concatenate([D*ones(len(xs)),sqrt((Lambda**(2-xm)*D0)**2+Gap**2)])
            res=D*ones(len(x),dtype='float64')
            fit_mask=x>2
            res[fit_mask]=sqrt((Lambda**(2-x[fit_mask])*D0)**2+Gap**2)
            return res

class LinearTicker(Ticker):
    '''
    Linear scale tick.

    Construct:
        LinearTicker(N,D,Gap=0.)

    Attributes:
        :N: The number of intervals.
        :D/Gap: The bandwidth/Gap range.
    '''
    def __init__(self,N,D,Gap=0.):
        self.N=N
        self.D=D
        self.Gap=Gap
        super(LinearTicker,self).__init__('linear')

    def __call__(self,x):
        Gap=self.Gap
        D=self.D
        N=self.N
        step=(D-Gap)/N
        res=D-(x-2)*step
        if ndim(x)==0:
            if x<=2:
                return D
            elif x>N+2:
                return Gap
            else:
                return res
        else:
            res[x<=2]=D
            res[x>N+2]=Gap
            return res

class AdaptiveTickerBase(Ticker):
    '''
    Base class of adaptive tick(Ref: Comp. Phys. Comm. 180.1271).

    Construct:
        AdaptiveTickerBase(tp,xf,wlist,rlist),
        `wlist` is the base space :math:`\\rho(\omega)`, and `rlist` is the weight function of rho(w).

    Attributes:
        :xf: A function defined on index-space(to decide logarithmic or linear...).
        :RD: The integration over rho(w) from 0 to D.
        :iRfunc: Inverse function of :math:`\int \\rho(\omega)`, :math:`iRfunc(x)=\int_0^\omega \\rho(\omega)`
    '''
    def __init__(self,tp,xf,wlist,rlist):
        super(AdaptiveTickerBase,self).__init__(tp)
        self.D=wlist[-1]
        self.xf=xf

        Rlist=concatenate([[0],cumtrapz(rlist,wlist)])
        self.RD=Rlist[-1]
        self.iRfunc=interp1d(Rlist,wlist)

    def __call__(self,x):
        D=self.D
        RD=self.RD
        iRfunc=self.iRfunc
        if ndim(x)==0:
            if x<=2:
                return D
            else:
                return iRfunc(RD*self.xf(x))
        else:
            #inds=searchsorted(x,2)
            #xs,xm=split(x,(inds,))
            #res=concatenate([D*ones(len(xs)),iRfunc(RD*self.xf(xm))])
            res=D*ones(len(x),dtype='float64')
            fit_mask=x>2
            res[fit_mask]=iRfunc(RD*self.xf(x[fit_mask]))
            return res

class AdaptiveLogTicker(AdaptiveTickerBase):
    '''
    Zitko's adaptive log scale tick(Ref: Comp. Phys. Comm. 180.1271).

    Construct:
        AdaptiveLogTicker(Lambda,wlist,rholist,r=1.), `rholist` is the weight of rho defined frequency space `wlist`.

    Attributes:
        :Lambda: Scaleing factor.
        :r: Adaptive ratio, 1.0 for typical adaptive-log and 0 for traditional log.
    '''
    def __init__(self,Lambda,wlist,rholist,r=1.):
        self.Lambda=Lambda
        self.r=r
        if ndim(rholist)!=1:
            raise ValueError('rholist should be no more than 1D, \
        it is recommended to get the norm before proceeding in the multi-band case.')
        super(AdaptiveLogTicker,self).__init__('adaptive_log',xf=lambda x:Lambda**(2-x),wlist=wlist,rlist=rholist)


class AdaptiveLinearTicker(AdaptiveTickerBase):
    '''
    Adaptive linear scale tick, with hopping terms constant.

    Construct:
        AdaptiveLinearTicker(Lambda,wlist,rholist,r=1.), `rholist` is the weight of rho defined frequency space `wlist`.

    Attributes:
        :N: The number of intervals.
        :r: Adaptive ratio, 1.0 for traditional adaptive and 0 for linear.
    '''
    def __init__(self,N,wlist,rholist,r=1.):
        self.r=r
        self.N=N
        if ndim(rholist)!=1:
            raise ValueError('rholist should be no more than 1D, \
        it is recommended to get the norm before proceeding in the multi-band case.')
        super(AdaptiveLinearTicker,self).__init__('adaptive_linear',xf=lambda x:(N+2.-x)/N,wlist=wlist,rlist=rholist)


class EDTicker(Ticker):
    '''
    Scale tick optimized for Exact-diagonalization cost function(PRL 72.1545).

    Construct
        EDTicker(N,wlist,rholist,wn),
        `rholist` is the weight of rho defined frequency space `wlist`.

    Attributes:
        :N: The number of intervals.
        :D: The bandwidth.
        :wn: The typical frequency to achieve the best fit.
        :Lambda: The complex scaling factor.
        :RD: The integration over rho(w) from 0 to D.
        :iRfunc: Inverse function of :math:`\int \\rho(\omega)`, :math:`iRfunc(x)=\int_0^\omega \\rho(\omega)`
        :xf: A auxiliary function defined on index-space.

    **Note:**
    The derivation of this tick distribution function involves many approximations,
    and is tested only for limited cases(like Bethe latttice, constant DOS).

    The main theme is to lower down the cost function while keeping the number of intervals fixed,
    which is different from NRG's infinite intervals.

    If it fails to get satisfactory mapping, please contact the author: dg1422033@smail.nju.edu.cn
    '''
    def __init__(self,N,wlist,rholist,wn):
        if ndim(rholist)!=1:
            raise ValueError('rholist should be no more than 1D, \
        it is recommended to get the norm before proceeding in the multi-band case.')
        super(EDTicker,self).__init__('ed')
        self.D=wlist[-1]
        self.N=N
        self.wn=wn
        Z=(1j*wn-self.D)/(1j*wn)
        self.Lambda=Z**(1./N)
        Rlist=concatenate([[0],cumtrapz(rholist,wlist)])
        self.RD=Rlist[-1]
        self.iRfunc=interp1d(Rlist,wlist)
        self.xf=lambda x:(1j*self.wn-self.D)*self.Lambda**(2-x)

    def __call__(self,x):
        D=self.D
        RD=self.RD
        iRfunc=self.iRfunc
        if ndim(x)==0:
            if x<=2:
                return D
            else:
                #res=iRfunc(RD*self.xf(x))
                #res=1j*self.wn-res
                ei=-self.xf(x).real
                if ei<0:
                    return 0
                else:
                    return ei
        else:
            inds=searchsorted(x,2)
            xs,xm=split(x,(inds,))
            res=concatenate([D*ones(len(xs)),-self.xf(xm).real])
            res[res<0]=0
            return res

def get_ticker(tick_type,D,**kwargs):
    '''
    Get specific <Ticker>.

    Parameters:
        :tick_type: The type of discretization ticks,

            * `log` -> logarithmic tick, kwargs: Lambda, Gap(optional)
            * `sclog` -> logarithmic ticks suited for superconductor, kwargs: Lambda, Gap(optional)
            * `adaptive` -> adaptive ticks, kwargs: Lambda, wlist, rholist, r(optional)
            * `linear` -> linear ticks, kwargs: N, Gap(optional)
            * `adaptive_linear` -> adaptive linear ticks, kwargs: N, wlist, rholist, r(optional)
            * `ed` -> ticks suited for fix number of intervals, kwargs: N, wlist, rholist, wn(optional)
        :D: The bandwidth.
        :kwargs: see tick_type.

    Return:
        A <Ticker> instance.
    '''
    Lambda=kwargs.get('Lambda')
    N=kwargs.get('N')
    wlist=kwargs.get('wlist')
    rholist=kwargs.get('rholist')
    if Lambda is None and (tick_type=='log' or tick_type=='sclog' or tick_type=='adaptive'):
        raise Exception('`log/sclog/adaptive` Type tick Needs keyword parameter Lambda!')
    if N is None and (tick_type=='ed' or tick_type=='linear' or tick_type=='adaptive_linear'):
        raise Exception('`ed/linear/adaptive_linear` Type tick Needs keyword parameter N!')
    if (wlist is None or rholist is None) and (tick_type=='adaptive' or tick_type=='adaptive_linear' or tick_type=='ed'):
        raise Exception('`ed/adaptive/adaptive_linear` Type tick Needs keyword parameter wlist and rholist!')
    r=kwargs.get('r',1)
    wn=kwargs.get('wn',pi/200)
    Gap=kwargs.get('Gap',0)

    if tick_type=='log':
        ticker=LogTicker(Lambda,D=D,Gap=Gap)
    elif tick_type=='sclog':
        ticker=ScLogTicker(Lambda,D=D,Gap=Gap)
    elif tick_type=='adaptive':
        ticker=AdaptiveLogTicker(Lambda,wlist=wlist,rholist=rholist,r=r)
    elif tick_type=='linear':
        ticker=LinearTicker(N,D=D,Gap=Gap)
    elif tick_type=='adaptive_linear':
        ticker=AdaptiveLinearTicker(N=N,wlist=wlist,rholist=rholist,r=r)
    elif tick_type=='ed':
        ticker=EDTicker(N=N,wlist=wlist,rholist=rholist,wn=wn)
    else:
        raise Exception('Error','Undefined tick type %s'%tick_type)
    return ticker


