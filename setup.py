from setuptools import setup
setup(
    name='ppim',
    version='1.0.3',
    author='jm12138',
    author_email='2286040843@qq.com',
    packages=['ppim', 'ppim.models', 'ppim.units'],
    license='Apache-2.0 License',
    description='Paddle Image Models',
    install_requires=['wget']
)
