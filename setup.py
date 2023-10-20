from pip.req import parse_requirements
from setuptools import setup, find_packages



setup(
    name='MM4SVR',
    version='0.1.0',
    author="Xinyue Zhang",
    author_email="yocelyn464@gmail.com",
    packages=find_packages(include=['scripts', 'scripts.*']),
    install_requires=parse_requirements('requirements.txt', session='hack'),
    extras_require={
        'interactive': ['matplotlib>=2.2.0', 'jupyter'],
    },
    package_data={'MM4SVR': ['data/*', 'checkpoints/*']}
)



