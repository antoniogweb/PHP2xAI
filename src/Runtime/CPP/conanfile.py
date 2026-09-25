from conan import ConanFile

class PHP2xAI(ConanFile):
    settings = "os", "arch", "compiler", "build_type"
    generators = "PkgConfigDeps"

    def requirements(self):
        self.requires("nlohmann_json/3.12.0")
        self.requires("hdf5/1.14.6")
