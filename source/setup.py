#from numpy.distutils.core import setup, Extension
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config=Configuration('nrgmap',parent_package,top_path,version='v1.0',author='JinGuo Leo',author_email='cacate0129@gmail.com')
    config.add_extension('lib.futils',['lib/futils.f90'],libraries=['lapack'])
    config.make_config_py()
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(configuration=configuration,package_data={'':['config-spec.ini']})
