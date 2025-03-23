"""Utilities useful during the build."""


# =============================================================================
# Imports
# =============================================================================

# Standard
from distutils.core import Extension
import glob
import os
import sys

# Third party
from Cython.Build import cythonize
import numpy as np


# =============================================================================
# Constants
# =============================================================================

# The extension modules that need to compile against NumPy should locate
# the corresponding include directory (source directory or core headers)
if np.show_config is None:
    d = os.path.join(os.path.dirname(np.__file__), "core", "include")
else:
    d = os.path.join(os.path.dirname(np.core.__file__), "include")

NUMPY_HEADERS_PATH = d


# =============================================================================
# Functions
# =============================================================================

def create_extension(extension_path):
    """Create and return an extension module."""
    (extension_name, _) = os.path.splitext(extension_path)
    extension_name = extension_name.replace(os.path.sep, ".")

    (sources_head, _) = os.path.split(extension_path)
    sources_pattern = os.path.join(sources_head, "src", "**", "*.cpp")

    sources = glob.glob(sources_pattern, recursive=True)
    sources += [extension_path]

    include_dirs = [NUMPY_HEADERS_PATH]
    extra_link_args = ["-std=c++11"]
    extra_compile_args = ["-O3", "-std=c++11"]
    library_dirs = []
    libraries = []
    language = "c++"

    if extension_name == "sklr._highs_solver":
        language = "c"
        include_dirs.append("/home/jalfaro/Software/micromamba/envs/JiayiWang-LMU/include/highs")
        include_dirs.append("/home/jalfaro/Software/micromamba/envs/JiayiWang-LMU/include/highs/interfaces")
        library_dirs.append("/home/jalfaro/Software/micromamba/envs/JiayiWang-LMU/lib")
        libraries.append("highs")

    return Extension(extension_name,
                     sources,
                     language=language,
                     include_dirs=include_dirs,
                     library_dirs=library_dirs,
                     libraries=libraries,
                     extra_link_args=extra_link_args,
                     extra_compile_args=extra_compile_args)


def cythonize_extensions(module_name):
    """Find and cythonize the extension modules."""
    # Skip cythonization in the release tarballs since
    # the generated C++ source files are not necessary
    if "sdist" not in sys.argv:
        extensions_pattern = os.path.join(module_name, "**", "*.pyx")
        extensions_path = glob.glob(extensions_pattern, recursive=True)

        for (extension, extension_path) in enumerate(extensions_path):
            extensions_path[extension] = create_extension(extension_path)

        extensions = cythonize(extensions_path)

        return extensions
