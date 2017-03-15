from setuptools import find_packages, setup

setup(
      name='PyDP',
      version='0.2.3',
      description='A Python library for implementing Dirichlet process mixture models.',
      author='Andrew Roth',
      author_email='andrewjlroth@gmail.com',
      url='https://bitbucket.org/aroth85/pydp',
      package_dir = {'': 'lib'},    
      packages=find_packages(),
      license = 'GPL v3'
     )
