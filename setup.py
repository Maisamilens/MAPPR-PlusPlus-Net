from setuptools import setup, find_packages

setup(
    name='mapprpp-net',
    version='1.0.0',
    description='MAPPR++-Net: Multi-Path Attention and Progressive Perception Refinement Plus Plus Network for Infrared Small Target Detection',
    author='Maisam Abbas',
    author_email='maisam.abbas@example.com',
    url='https://github.com/Maisamilens/MAPPR-PlusPlus-Net',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.24.0',
        'opencv-python>=4.8.0',
        'matplotlib>=3.7.0',
        'scipy>=1.11.0',
        'tqdm>=4.65.0',
        'PyYAML>=6.0',
        'tensorboard>=2.14.0',
        'timm>=0.9.0',
        'albumentations>=1.3.0',
        'scikit-learn>=1.3.0',
        'pandas>=2.0.0',
    ],
)