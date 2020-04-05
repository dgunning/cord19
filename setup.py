# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup

with open('requirements.txt') as f:
    REQUIRED = [l.strip() for l in f.readlines()]

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

EXTRAS_REQUIRES = {'nlp': ['allennlp', 'torch===1.4.0', 'scispacy'],
                   'ui': ['streamlit']}
EXTRAS_REQUIRES['all'] = EXTRAS_REQUIRES['nlp'] + EXTRAS_REQUIRES['ui']
setup(
    name='cord19',
    version='0.4.0',
    description='COVID-19 Open Dataset Research Tool',
    long_description=readme,
    author='Dwight Gunning',
    author_email='dgunning@gmail.com',
    url='https://github.com/dgunning/cord19.git',
    license=license,
    packages=['cord'],
    install_requires=REQUIRED,
    extras_require=EXTRAS_REQUIRES,
    include_package_data=True
)
