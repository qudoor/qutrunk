import os
import sys
import pathlib
from setuptools import setup
from setuptools import extension
from distutils.command.build_ext import build_ext
from setuptools import Extension


class CMakeExtension(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])


simulator = CMakeExtension("qutrunk/sim/local")


ext_modules = [simulator]


class BuildFailed(Exception):
    pass


class ExtBuilder(build_ext):
    def run(self):
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                self.build_cmake(ext)
        super().run()

    def build_cmake(self, ext):
        cwd = pathlib.Path().absolute()

        build_temp = f"{pathlib.Path(self.build_temp)}/{ext.name}"

        os.makedirs(build_temp, exist_ok=True)
        extdir = pathlib.Path(self.get_ext_fullpath(ext.name))
        extdir.mkdir(parents=True, exist_ok=True)

        config = "Debug" if self.debug else "Release"
        cmake_args = [
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY="
            + str(extdir.parent.absolute())
            + "/local",
            "-DCMAKE_BUILD_TYPE=" + config,
        ]

        build_args = ["--config", config]

        os.chdir(build_temp)
        self.spawn(["cmake", f"{str(cwd)}/{ext.name}"] + cmake_args)
        if not self.dry_run:
            self.spawn(["cmake", "--build", "."] + build_args)
        os.chdir(str(cwd))


def build(setup_kwargs):
    """
    This function is mandatory in order to build the extensions.
    """
    setup_kwargs.update(
        {"ext_modules": ext_modules, "cmdclass": {"build_ext": ExtBuilder}}
    )
