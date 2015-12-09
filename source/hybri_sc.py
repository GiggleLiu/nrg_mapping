'''
Get the hybridization function of a conventional superconductor bath.
wide-band approximation(get_hybri_wideband) version and finite-band width(get_hybri) version are available.
'''

from numpy import *
from utils import sx,sz

__all__=['get_hybri','get_hybri_wideband']

def get_hybri(Gap,Gamma,D0=1.,mu=0.,eta=1e-10):
    '''
    Get the hybridization function D(w) for superconducting surface.

    Parameters
    ------------------
    Gap:
        The Gap value.
    Gamma:
        The overall hybridization strength.
    D0:
        The band width of normal DOS(The true bandwidth is sqrt(D0**2+Gap**2)).
    mu:
        Chemical potential.
    eta:
        The smearing factor.

    Return
    -------------------
    A function, which is the hybridization function for superconductor.
    '''
    def hybri(w):
        z=(w+1j*eta)
        sqc=sqrt(Gap**2-mu**2-z**2)
        res=-Gamma/pi*((arctan(D0/sqc)-arctan(-D0/sqc))/sqc)*(z*identity(2)-Gap*sx)
        return 1j/2./pi*(res-res.conj().T)
    return hybri

def get_hybri_wideband(Gap,Gamma,D=1.,mu=0.,eta=1e-10):
    '''
    Get the hybridization function D(w) for superconducting surface using wide-band approximation.

    Parameter
    ---------------------
    Gap:
        The Gap value.
    Gamma:
        The overall hybridization strength.
    D:
        The band width.
    mu:
        Chemical potential.
    eta:
        The smearing factor.

    Return
    -------------------
    A function, which is the hybridization function for superconductor with wideband approximation.
    '''
    def hybri_wideband(w):
        E=sqrt(Gap**2-mu**2-(w+1j*eta)**2)
        if abs(w)<=D:
            if abs(mu)>0:
                res=-Gamma*(((w+1j*eta)/E)*identity(2)-(mu/E)*sz-(Gap/E)*sx)
            else:
                res=-Gamma*(((w+1j*eta)/E)*identity(2)-(Gap/E)*sx)
        else:
            res=zeros([2,2])
        return 1j/2./pi*(res-res.conj().T)
    return hybri_wideband

