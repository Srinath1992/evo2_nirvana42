import os
import subprocess
import sys
from setuptools import setup, find_packages
from setuptools.command.build import build as _build  # use the top-level build command
from setuptools.command.develop import develop as _develop
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


def update_submodules():
    """No-op in development images.

    The `vortex` directory is now vendored or volume-mounted directly instead of
    being updated via `git submodule`.  Running git commands here clashes with
    local, uncommitted edits that developers frequently hot-reload inside the
    container, so we intentionally skip all sub-module operations.
    """
    print("Skipping git sub-module update for vortex (dev mode).")

def run_make_setup_full():
    base_dir = os.path.dirname(__file__)
    vortex_dir = os.path.join(base_dir, 'vortex')
    original_dir = os.getcwd()

    # Ensure submodules are updated before running the Makefile
    update_submodules()

    # Ensure the Makefile uses the current Python interpreter
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    print(f"Running 'make setup-full' in {vortex_dir} with PYTHON={sys.executable} ...")
    
    try:
        os.chdir(vortex_dir)
        subprocess.check_call(['make', 'setup-full'], env=env)
    finally:
        os.chdir(original_dir)

class CustomBuild(_build):
    def run(self):
        # Run egg_info to ensure metadata is available
        self.run_command('egg_info')
        # Update submodules and run the Makefile before building anything else
        run_make_setup_full()
        # Continue with the normal build process
        _build.run(self)

class CustomDevelop(_develop):
    def run(self):
        update_submodules()
        run_make_setup_full()
        _develop.run(self)

class CustomBDistWheel(_bdist_wheel):
    def run(self):
        self.run_command('egg_info')
        _bdist_wheel.run(self)

def parse_requirements(filename):
    requirements = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                requirements.append(line)
    return requirements


with open('evo2/version.py') as infile:
    exec(infile.read())

with open('README.md') as f:
    readme = f.read()

requirements = parse_requirements("requirements.txt")

setup(
    name='evo2',
    version=version,
    # Only include the evo2 package; the vortex submodule is used for build purposes.
    packages=find_packages(include=["evo2", "vortex/vortex"]),
    install_requires=requirements,
    cmdclass={
        'build': CustomBuild,
        'develop': CustomDevelop,
        'bdist_wheel': CustomBDistWheel,
    },
    package_data={'evo2': ['evo2/configs/*.yml']},
    include_package_data=True,
    # python_requires='>=3.11',
    python_requires='>=3.10',
    license="Apache-2.0",
    description='Genome modeling across all domains of life',
    long_description=readme,
    long_description_content_type='text/markdown',
    author='Team Evo 2',
    url='https://github.com/arcinstitute/evo2',
)
