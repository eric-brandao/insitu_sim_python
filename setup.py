import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'VERSION')) as version_file:
    version = version_file.read().strip()

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions)
            )

        if platform.system() == "Windows":
            cmake_version = LooseVersion(
                re.search(r'version\s*([\d.]+)', out.decode()).group(1)
            )
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext.name)))
        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            '-DPYTHON_EXECUTABLE=' + sys.executable
        ]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            cmake_args += [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(
                    cfg.upper(), extdir
                )
            ]
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j6']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        subprocess.check_call(
            ['cmake', ext.sourcedir] + cmake_args,
            cwd=self.build_temp, env=env
        )
        subprocess.check_call(
            ['cmake', '--build', '.'] + build_args,
            cwd=self.build_temp
        )


setup(
    name='insitu',
    version=version,
    description='In situ impedance',
    long_description=long_description,
    url="https://github.com/...",
    author='author',
    author_email='author@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Python :: 3',
    ],
    keywords="python",
    packages=find_packages(
        exclude=[
            "examples",
            "tests",
            ".vscode",
            ".venv"
        ]),
    data_files=[(
        '.', [
            'VERSION',
            'requirements.txt',
        ]
    )],
    include_package_data=True,
    # entry_points={
    #     "console_scripts": [
    #         "ra=ra.__main__:main",
    #     ],
    # },
    project_urls={
        "Bug Reports": "https://github.com/...",
        "Source": "https://github.com/...",
    },

    ext_modules=[CMakeExtension('insitu_cpp')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=[
        'matplotlib==3.1.1',
        'numpy==1.22.0',
        'numpy-quaternion==2019.7.23.15.26.49',
        'pycollada==0.6',
        'scipy==1.3.1',
        'toml==0.10.0',
        'xlrd==1.2.0',
        'xlwt==1.3.0',
    ],
)
