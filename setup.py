import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kinref-pkg",
    version="1.0.0",
    author="Seokhyeon Hong",
    author_email="graphics.shong@gmail.com",
    description="Skinned Motion Retargeting via Artifact-driven Kinematic Prior Refinement",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
)
