'''
Get the hybridization function of a conventional superconductor bath.
wide-band approximation(get_hybri_wideband) version and finite-band width(get_hybri) version are available.
'''

from numpy import *
from utils import sx,sz

__all__=['get_hybri','get_hybri_wideband','get_hybri_skew']

def get_hybri(Gap,Gamma,D0=1.,mu=0.,eta=1e-10):
    '''
    Get the hybridization function D(w) for superconducting surface.

    Parameters:
        :Gap: The Gap value.
        :Gamma: The overall hybridization strength.
        :D0: The band width of normal DOS(The true bandwidth is sqrt(D0**2+Gap**2)).
        :mu: Chemical potential.
        :eta: The smearing factor.

    Return:
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

    Parameter:
        :Gap: The Gap value.
        :Gamma: The overall hybridization strength.
        :D: The band width.
        :mu: Chemical potential.
        :eta: The smearing factor.

    Return:
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

def get_hybri_skew(Gap,Gamma,skew,D=1.,eta=1e-10,g=False):
    '''
    Get the skewed hybridization function for superconductor.

    Parameters:
        :Gap: The gap value.
        :Gamma/skew: The overall strength, skew of hybridization function.
        :D: The band-width.
        :eta: Smearing factor, None for matsubara Green's function.
        :g: Get self energy instead of hybridization function.

    Return:
        A function, which is the hybridization function for superconductor with sz term.
    '''
    one=identity(2)
    N0=Gamma/pi
    if skew*D>1:
        raise ValueError('skew parameter is incorrect! it should be no more than 1/D = %s.'%(1./D))
    def gfunc(w):
        z=(w+1j*eta) if eta!=None else 1j*w
        sqc=sqrt(Gap**2-z**2)
        I0=-2*arctan(D/(sqc))/sqc
        I2=-2*D+2*sqc*arctan(D/sqc)
        return N0*(I0*z*one-I0*Gap*sx+skew*I2*sz)
    def dfunc(w):
        g=gfunc(w)
        return 1j/2./pi*(g-g.conj().T)
    if g:
        return gfunc
    else:
        return dfunc


