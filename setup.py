import setuptools

setuptools.setup(
    name="ForegroundObjectTracker",
    version="1.0.3",
    author="Benjamin MONSERAND",
    author_email="benjamin.monserand@utbm.fr",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.0",
    install_requires=[
        "opencv-python",
        "numpy",
    ]
)
