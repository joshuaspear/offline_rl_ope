from distutils.core import setup
import pathlib
import setuptools
import os


HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

# get __version__ variable
exec(open(os.path.join(HERE, '_version.py')).read())

setuptools.setup(
    name='offline_rl_ope',
    version=__version__,
    description="",
    long_description=README,
    packages=setuptools.find_packages(where="src"),
    author="Joshua Spear",
    author_email="josh.spear9@gmail.com",
    long_description_content_type="text/markdown",
    url="",
    license='MIT',
    classifiers=[],
    package_dir={"": "src"},
    python_requires="",
    install_requires=[""]
)