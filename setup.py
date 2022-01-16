import setuptools

setuptools.setup(
    name="transprop",
    packages=["transprop"],
    version="0.0.1",
    description="Calculate transport properties of gasses",
    author="Robert Watt",
    install_requires=["numpy", "scipy", "uncertainties"]
)
