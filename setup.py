from setuptools import setup, find_packages

# Read requirements from requirements.txt
def read_requirements():
    with open("requirements.txt", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="tinyfabulist",
    version="0.1",
    packages=find_packages(),
    install_requires=read_requirements(),  # Load dependencies from requirements.txt
)
