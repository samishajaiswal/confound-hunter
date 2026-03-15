from setuptools import setup, find_packages

setup(
    name="confound-hunter",
    version="0.1",
    description="ML auditing tool to detect spurious correlations in tabular ML models",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "shap",
        "scipy",
        "statsmodels",
        "plotly",
        "jinja2",
        "click",
        "tqdm"
    ],
)