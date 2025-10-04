\# MLOps Assignment 1



This project trains two models (DecisionTreeRegressor and KernelRidge) on the Boston Housing dataset.



\## Setup

conda activate mlops-a1

pip install -r requirements.txt



\## Run

python train.py      # DecisionTreeRegressor

python train2.py     # KernelRidge

\## CI Results (KernelRidge branch)

\- \[DecisionTreeRegressor] Test MSE: 10.4161

\- \[KernelRidge] Test MSE: 18.1512



\## Branches

\- main — merged work and documentation

\- dtree — DecisionTreeRegressor implementation (merged into main)

\- kernelridge — KernelRidge + GitHub Actions CI (runs on push)



