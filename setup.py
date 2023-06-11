from setuptools import setup

# Read the contents of requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='bike_rental_package',
    version='1.0.0',
    author='IISc Team 5 Group',
    author_email='manikandan_np@yahoo.com',
    description='IISc Projects',
    packages=['bike_rental_model'],
    install_requires=requirements,
)