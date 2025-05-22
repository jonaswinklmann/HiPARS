from setuptools import setup

setup(
    name='HiPARS',
    version='0.0.1',    
    description='Package for sorting arrays of neutral atoms',
    url='https://github.com/jonaswinklmann/hipars',
    author='Jonas Winklmann',
    author_email='jonas.winklmann@tum.de',
    packages=['hipars'],
    install_requires=['numpy'],
    license_files='LICENSE',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: C++'
        'Programming Language :: Python :: 3',
    ],
    include_package_data=True,
    package_data={'': ['*.so','*.dll']}
)