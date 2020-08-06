import setuptools

with open("README.md", "r") as fh:
    long_desc = fh.read()

setuptools.setup(
    name="bayes_perm_hmm",
    version="0.0.1",
    author="Shawn Geller",
    author_email="shawn.geller@colorado.edu",
    description="Computes misclassification rates for repeated measurement"
                " schemes",
    long_description=long_desc,
    url="https://gitlab.nist.gov/gitlab/sng13/bayes_perm_hmm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
