from setuptools import setup, find_packages

setup(
    author="Ayush Raj Dahal",
    description="A library containing sequential algorithms for NLP.",
    name="sequentials",
    packages=find_packages(include=["sequentials", "sequentials.*"]),
    version="0.1.0",
    install_requirements=[
        'torch>=2.0.1'
    ],
)
