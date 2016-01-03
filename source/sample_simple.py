from numpy import *
from discretization import quick_map,get_wlist,check_disc
from chainmapper import map2chain,check_spec
from matplotlib import pyplot as plt

sx=array([[0,1],[1,0]])
sy=array([[0,-1j],[1j,0]])
sz=array([[1,0],[0,-1]])

#defined the hybridization function
wlist=get_wlist(Nw=5000,mesh_type='log',w0=1e-12,D=1.)
rhofunc=lambda w:identity(2)+0.4*sx+0.5*(0.1+w**2)*sy

#map it to a sun model
discmodel=quick_map(wlist=wlist,rhofunc=rhofunc,N=35,z=linspace(0.05,0.95,10),\
        tick_params={'tick_type':'adaptive','Lambda':2.})[1]

#map it to a Wilson chain
chains=map2chain(discmodel,prec=3000)

#do some checking
#check the sun model
plt.subplot(211)
check_disc(rhofunc=rhofunc,wlist=wlist[20::40],discmodel=discmodel,\
        smearing=0.7,mode='pauli')
plt.ylim(-0.1,1.1)
#check the chain
plt.subplot(212)
check_spec(rhofunc=rhofunc,chains=chains,wlist=wlist[20::40],mode='pauli',smearing=0.7)
plt.ylim(-0.1,1.1)
plt.show()
