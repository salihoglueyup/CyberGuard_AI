# setup.py

from setuptools import setup, find_packages

setup(
    name="cyberguard-ai",
    version="1.0.0",
    description="AI-Powered Cybersecurity Platform",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open('requirements.txt').readlines()
        if not line.startswith('#')
    ],
    python_requires='>=3.10',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
    ],
)