#!/usr/bin/python
'''
Get the hybridization function of a conventional superconductor.
wide-band approximation(get_hybri_wideband) version and finite-band width(get_hybri) version are available.
'''
from numpy import *
from utils import sx,sz

def get_hybri(Gap,Gamma,D0=1.,mu=0.,eta=1e-10):
    '''
    D(w) for superconducting surface.

    Gap:
        the Gap value.
    Gamma:
        the overall strength.
    D0:
        the band width of normal part.
    mu:
        chemical potential.
    eta:
        the smearing factor.
    *return*:
        hybridization function for superconductor.
    '''
    def hybri(w):
        z=(w+1j*eta)
        sqc=sqrt(Gap**2-mu**2-z**2)
        res= -Gamma/pi*((arctan(D/sqc)-arctan(-D/sqc))/sqc)*(z*identity(2)-Gap*sx)
        return 1j/2./pi*(res-res.conj().T)
    return hybri

def get_hybri_wideband(Gap,Gamma,D=1.,mu=0.,eta=1e-10):
    '''
    D(w) for superconducting surface, taking wide-band approximation.

    Gap:
        the Gap value.
    Gamma:
        the overall strength.
    D:
        the band width.
    mu:
        chemical potential.
    eta:
        the smearing factor.
    *return*:
        hybridization function for superconductor with wideband approximation.
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

