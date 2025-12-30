import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required_packages = [
    "pandas",
    "matplotlib",
    "numpy",
    "openpyxl",
    "plotly",
    "dash",
    "scipy",
    "seaborn",
    "scikit-learn",
    "statsmodels",
    "optbinning"
]

for package in required_packages:
    install(package)