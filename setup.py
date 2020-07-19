from setuptools import find_packages, setup

with open("reqs.txt") as f:
    reqs = f.read().splitlines()

setup(name="simulation", install_requires=reqs, packages=find_packages())
