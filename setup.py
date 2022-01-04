import setuptools

setuptools.setup(
    name="ForegroundObjectTracker",
    version="1.0.8",
    author="Benjamin MONSERAND",
    author_email="benjamin.monserand@utbm.fr",
    url='https://github.com/Xwaler/ForegroundObjectTracker',
    long_description=open('README.md').read(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "opencv-python>=4.5",
        "numpy>=1.20",
    ]
)
