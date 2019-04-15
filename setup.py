import os
import sys
from setuptools import setup, Command, find_packages
from setuptools.command.install import install
from subprocess import check_output


while True and len(sys.argv) >= 2 and sys.argv[1] == "install":
    try:
        prefix = sys.prefix
        if not any(['CONDA_PREFIX' in os.environ, 'VIRTUAL_ENV' in os.environ]):
            print "WARNING: You do not appear to be in a virtual environment!"
        user_input = raw_input("Current python environment is: {}\n"
                               "Proceed with setup install? (y/n) ".format(prefix))
        if user_input == "n":
            sys.exit()
        elif user_input == "y":
            break
        else:
            print "Incorrect input."
    except (AttributeError, KeyboardInterrupt):
        # if sys.prefix cannot be determined.
        raise


class CleanCommand(Command):
    """
    Custom clean command to tidy up the project root
    """
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


def req_file(filename):
    """
    We're using a requirements.txt file so that pyup.io can use this for security checks
    """
    with open(filename) as f:
        content = f.readlines()
        content = filter(lambda x: not x.startswith('#'), content)

    return [x.strip() for x in content]


def most_recent_tag():
    return check_output(["git", "describe", "--tags"]).strip().split('-').pop(0)


setup(
    name='MiMSI',
    version=most_recent_tag(),
    description='A deep, multiple instance learning based classifier for identifying Microsatellite Instability from NGS',
    url='https://github.com/mskcc/mimsi',
    author='John Ziegler',
    author_email='zieglerj@mskcc.org',
    license='GNU General Public License v3.0',
    install_requires=req_file('requirements.txt'),
    classifiers=[
            'Topic :: Scientific/Engineering :: Bio-Informatics',
            'Programming Language :: Python :: 2.7',
    ],
    packages=find_packages(exclude=['tests*']),
    py_modules=['analyze'],
    python_requires='==2.7.*, ==3.5.*, ==3.6.*',
    package_data={
        'utils': ['microsatellites.list.gz'],
        'model': ['*.model']
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'analyze = analyze:main',
            'create_data = data.generate_vectors.create_data:main',
            'evaluate_sample = main.evaluate_sample:main',
            'mi_msi_train_test = main.mi_msi_train_test:main'
        ]
    }
)
