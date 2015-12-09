from numpy import *
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

    def save_data(self,token):
        '''
        save data.

        token:
            the target filename token.
        '''
        fname='data/%s.npy'%token
        data=concatenate([self.Elist,self.Tlist])
        save(fname,data)

    def load_data(self,token):
        '''
        load specific data.

        token:
            the target filename token.
        '''
        fname='data/%s.npy'%token
        data=load(fname)
        self.Elist,self.Tlist=split(data,2)

