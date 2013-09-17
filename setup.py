from distutils.core import setup

setup(
      name='PyDP',
      version='0.2.1',
      description='A Python library for implementing Dirichlet process mixture models.',
      author='Andrew Roth',
      author_email='andrewjlroth@gmail.com',
      url='https://bitbucket.org/aroth85/pydp',
      package_dir = {'': 'lib'},    
      packages=[ 
                'pydp',
                'pydp.samplers'
                ],
      license = 'GPL v3'
     )
