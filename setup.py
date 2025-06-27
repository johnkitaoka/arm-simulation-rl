from setuptools import setup, find_packages

setup(
    name="robot-arm-simulation",
    version="1.0.0",
    description="Native Desktop Robot Arm Simulation with Machine Learning Integration",
    author="Robot Simulation Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "PyOpenGL>=3.1.0",
        "PyOpenGL-accelerate>=3.1.0",
        "glfw>=2.5.0",
        "moderngl>=5.6.0",
        "pybullet>=3.2.0",
        "torch>=1.12.0",
        "transformers>=4.20.0",
        "stable-baselines3>=1.6.0",
        "gymnasium>=0.26.0",
        "opencv-python>=4.6.0",
        "Pillow>=9.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.64.0",
        "matplotlib>=3.5.0",
    ],
    entry_points={
        "console_scripts": [
            "robot-sim=main:main",
            "robot-gui=run_native_gui:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
