import os

from setuptools import setup, find_packages

reqs = ['h5py>=2.6.0',
        'numpy>=1.12.1',
        'docopt>=0.6.2',
        'bigtiff>=0.1.1']

def readme():
    README_md = os.path.join(os.path.dirname(__file__), 'README.md')
    with open(README_md) as f:
        return f.read()


setup(name='pyilastik',
      version='0.0.5',
      description='Read ilastik labels in python',
      author='Manuel Schoelling',
      author_email='manuel.schoelling@dzne.de',
      include_package_data=True,
      packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
      zip_safe=False,
      install_requires=reqs,
      test_suite='nose.collector',
      tests_require=['nose', 'coverage', 'nose-timer', 'nose-deadline'])
