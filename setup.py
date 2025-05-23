import setuptools

# read the contents of requirements.txt
with open('./requirements.txt') as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name = 'adrd',
    version = '0.0.1',
    author = 'Sahana Kowhsik',
    author_email = 'skowshik@bu.edu',
    url = 'https://github.com/vkola-lab/ncomms2025/',
    # description = '',
    packages = setuptools.find_packages(),
    python_requires = '>=3.11',
    classifiers = [
        'Environment :: GPU :: NVIDIA CUDA',
        'Operating System :: OS Independent',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    install_requires = requirements,
)
