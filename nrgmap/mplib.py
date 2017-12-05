'''
Lib for High Precision(analytical) operations.
'''

from numpy import vectorize,dot,asarray,zeros,concatenate
import gmpy2

__all__=['mpconj','mpqr']

#conjugate
mpconj=vectorize(lambda x:gmpy2.mpc(x.real,-x.imag))

def mpqr(A):
    '''
    Analytically, get the QR decomposition of a matrix.

    Parameters:
        :A: 2d array, matrix of type object.

    Return:
        (Q,R), where QR=A, Q is columnwise orthogonal, and R is upper triangular.
    '''
    A=asarray(A)
    if A.shape[0]<A.shape[1]: raise NotImplementedError()
    ndim=A.shape[1]
    Q=zeros([A.shape[0],0])
    csqrt=vectorize(gmpy2.sqrt)
    for i in range(ndim):
        ui=A[:,i:i+1]
        if i>0:
            ui=ui-dot(Q,dot(mpconj(Q.T),ui))
        ui=ui/csqrt(dot(mpconj(ui.T),ui))
        Q=concatenate([Q,ui],axis=-1)
    R=dot(mpconj(Q.T),A)
    return Q,R
