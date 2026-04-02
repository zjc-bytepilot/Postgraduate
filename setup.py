# setup.py
from setuptools import setup, find_packages

setup(
    name='openpan',
    version='1.0.0',
    description='A Data-Adaptive and Band-Agnostic Unsupervised Pansharpening Framework',
    author='Your Name',
    packages=find_packages(), # 这一行会自动把 openpan 文件夹识别为一个 Python 模块
    install_requires=[
        # 依赖可以写在 requirements.txt 里，这里留空即可
    ],
)