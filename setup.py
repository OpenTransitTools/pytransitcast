from setuptools import setup, find_packages

setup(
    name="updatedist",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "xgboost==1.6.1",
        "pandas==1.4.3",
        "holidays==0.14.2",
        "numpy==1.23.2",
        "scipy==1.9.0",
        "scikit_learn==1.1.2",
        "psycopg2-binary==2.9.3",
        "nats-py==2.1.4",
    ],
)
