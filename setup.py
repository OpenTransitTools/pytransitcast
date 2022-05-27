from setuptools import setup, find_packages
setup(
    name="updatedist",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'xgboost==1.5.2',
        'pandas==1.1.5',
        'holidays==0.13',
        'numpy==1.19.5',
        'scikit_learn==0.24.2',
        'psycopg2-binary==2.9.3'
    ]
)
