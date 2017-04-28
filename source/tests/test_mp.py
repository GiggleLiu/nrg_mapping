from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
import gmpy2
import time,pdb,sys
sys.path.insert(0,'../')

from mplib import *

tocomplex=vectorize(gmpy2.mpc)
assert_mpclose=lambda x,y:assert_(abs(x-y).sum()<1e-10)

def test_conj():
    print 'Test conjugate.'
    c1=tocomplex(1+2j)
    assert_(mpconj(c1)==gmpy2.mpc(1-2j))
    print 'vectorized conjugate.'
    arr=array([1-2j,0.1,6j])
    carr=tocomplex(arr)
    assert_mpclose(mpconj(carr),tocomplex(arr.conj()))

def test_qr():
    for shp in [[3,3],[3,2],[2,3]]:
        print 'test qr for shape %s'%shp
        ai=random.random(shp)+1j*random.random(shp)
        ai=tocomplex(ai)
        if shp[0]<shp[1]:
            assert_raises(NotImplementedError,mpqr,ai)
            continue
        Q,R=mpqr(ai)
        #QR=ai
        assert_mpclose(dot(Q,R),ai)
        #orthogonality of Q
        assert_mpclose(dot(mpconj(Q.T),Q),identity(Q.shape[1]))
        #lower left part of R be empty.
        for i in xrange(R.shape[0]):
            for j in xrange(R.shape[1]):
                if i>j:
                    assert_almost_equal(R[i,j],0)

def test_all():
    test_conj()
    test_qr()

if __name__=='__main__':
    test_all()
