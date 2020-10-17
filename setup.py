from setuptools import setup, find_packages

with open("requirements.txt", "r") as rq:
    packages_list = rq.read()

tests_require = [
    'pytest'
]

setup(
    name='braf',
    version="0.0.1",
    author="Jean",
    packages=find_packages('src'),
    package_dir={'': 'src'},
    description="",
    install_requires=[packages_list],
    setup_requires=[
        'pytest-runner'
    ],
    scripts=["bin/braf.sh", "bin/braf.bat"],
    tests_require=tests_require,
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'braf=braf.main:run_app',
        ]
    }
)
