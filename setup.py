#!/usr/bin/env python3
import sys

import os
import subprocess
from setuptools import setup, find_packages
from setuptools.command.test import test

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        print(extdir)
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
        build_args += ['--', '-j2']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())

        cmake_args += ['-DCPU_ONLY=1', '-DUSE_OPENCV=0', '-DBUILD_docs=0', '-DUSE_LEVELDB=0', '-DUSE_LMDB=0', '-DUSE_HDF5=1',  '-DBUILD_SHARED_LIBS=0', '-DBOOST_ROOT=/opt/conda/']
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        try:
            subprocess.check_call(['git clone -b master --depth 1 https://github.com/BVLC/caffe.git .'], cwd=self.build_temp, env=env, shell=True)
        except subprocess.CalledProcessError:
            pass
        subprocess.check_call(['cmake', '.'] + cmake_args, cwd=self.build_temp, env=env, shell=False)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp, shell=False)
        
        import shutil
        shutil.copytree(os.path.join(self.build_temp, 'python', 'caffe'), os.path.join(extdir, 'caffe'))


def run_script(command):
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    while True:
        line = proc.stdout.readline()
        if line != b'':
            os.write(1, line)
        else:
            break


class TestCommand(test):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        test.initialize_options(self)
        self.pytest_args = '-v tests'

    def run_tests(self):
        # clean up caches
        run_script('find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf')

        import shlex
        # import here, cause outside the eggs aren't loaded
        import pytest
        run_script('pip install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-linux_x86_64.whl')
        run_script('pip install git+https://github.com/onnx/onnx.git')
        run_script('pip install torchvision')
        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)


# parse_requirements() returns generator of pip.req.InstallRequirement objects
reqs = ['scikit-image'] #for caffe

# Get the long description from the README file
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='onnx-caffe',
    version='0.0.0',
    description='a onnx toolkit that convert onnx model to caffe',
    long_description=long_description,
    url='https://github.com/0wu/onnx-caffe',
    author='Tingfan Wu',
    author_email='tingfan.wu@umbocv.com',
    python_requires='>=3',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=reqs,
    tests_require=['pytest==3.2.2',
                   'pytest-cov==2.5.1',
                   'pytest-flake8==0.9',
                   'pytest-sugar==0.9.0'],
    ext_modules=[CMakeExtension('caffe')],
    cmdclass={
        'test': TestCommand,
	    'build_ext': CMakeBuild
    }
)
