from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="kb-nova-pipeline-models",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI/ML Pipeline Models for Knowledge Base Nova",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kb-nova-pipeline-models",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.4.1",
            "pre-commit>=3.3.3",
        ],
    },
    entry_points={
        "console_scripts": [
            "kb-nova-train=src.training.train:main",
            "kb-nova-predict=src.inference.predict:main",
        ],
    },
) 