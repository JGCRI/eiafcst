from distutils.core import setup

setup(
    name='eiafcst',
    version='0.1.0',
    author='Caleb Braun',
    author_email='caleb.braun@pnnl.gov',
    packages=['eiafcst'],
    package_data={'eiafcst': ['data/raw-data/*.h5']},
    license='LICENSE.txt',
    description='Model GDP from energy consumption data.',
    long_description=open('README.txt').read(),
    python_requires='>=3.6, <4',
)
