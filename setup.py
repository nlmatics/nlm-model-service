from setuptools import find_packages  # noreorder
from setuptools import setup  # noreorder

import platform
import sys
from distutils.core import Extension

import numpy
from Cython.Build import cythonize

NAME = "nlm-model-server"
VERSION = "v2.1.0"

REQUIRES = []

# for cython sif
includes = [numpy.get_include()]
extra_args = ["-ffast-math", "-O3"]

system = platform.system()

if system == "Linux":
    extra_args.append("-std=c++11")
elif system == "Darwin":
    extra_args.extend(["-stdlib=libc++", "-std=c++11"])

extensions = [
    Extension(
        "sif.cython_sif",
        sources=["sif/cython_sif.pyx"],
        language="c++",
        include_dirs=includes,
        extra_compile_args=extra_args,
        extra_link_args=extra_args,
    ),
]


setup(
    name=NAME,
    version=VERSION,
    description="NLM Model Server",
    author_email="info@nlmatics.com",
    url="",
    keywords=["NLM Model Server"],
    install_requires=REQUIRES,
    packages=find_packages(),
    include_package_data=True,
    scripts=[
        "sif/cython_sif.cpp",
        "sif/cython_sif.pyx",
        "sif/data_io.py",
        "sif/__init__.py",
        "sif/lasagne_average_layer.py",
        "sif/params.py",
        "sif/SIF_embedding.py",
        "sif/sif.py",
        "sif/tree.py",
        "nlp_server/modules/StopwordsList/stopwords_en.txt"
    ],
    long_description="""\
    API specification for nlm-service  # noqa: E501
    """,
    ext_modules=cythonize(
        extensions, compiler_directives={"language_level": sys.version_info[0]},
    ),
)