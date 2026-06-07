from setuptools import setup, find_packages

setup(
    name="brain_tumor_detection",
    version="1.0.0",
    author="Shaili Jaiswal",
    author_email="your_email@gmail.com",
    description="Brain Tumor Detection using Deep Learning and Computer Vision",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "tensorflow",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "opencv-python",
        "scikit-learn",
        "pyyaml",
        "pytest"
    ],
    python_requires=">=3.10",
)