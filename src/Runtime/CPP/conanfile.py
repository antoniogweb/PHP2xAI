from conan import ConanFile

class PHP2xAI(ConanFile):
    settings = "os", "arch", "compiler", "build_type"
    generators = "PkgConfigDeps"

    def requirements(self):
        self.requires("eigen/3.4.0")
        self.requires("nlohmann_json/3.12.0")
