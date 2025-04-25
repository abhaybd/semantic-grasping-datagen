from setuptools import setup, find_namespace_packages

# Read requirements from requirements.txt
def read_requirements(filename="requirements.txt"):
    with open(filename) as f:
        requirements = []
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            requirements.append(line)
        return requirements

setup(
    name="semantic-grasping-datagen",
    version="0.1.0",
    description="Data generation tools for semantic grasping",
    author="Abhay Deshpande",
    packages=["semantic_grasping_datagen"] + find_namespace_packages(include=["semantic_grasping_datagen.*"]),
    python_requires=">=3.8",
    install_requires=read_requirements(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
