#from numpy.distutils.core import setup, Extension
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config=Configuration('nrgmap',parent_package,top_path)
    config.add_extension('lib.futils',['lib/futils.f90'],libraries=['lapack'])
    return config
