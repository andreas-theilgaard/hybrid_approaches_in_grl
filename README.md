Hybrid Approaches in Graph Representation Learning
==============================

About
------------
...

Environment Setup
------------
First create a virtual environment by
```
conda create -n <env_name> python=3.10
conda activate <env_name>
```

Then install the torch related packages using the following

Use the folloiwng is cuda is avaviable. This is recommended.
```
pip3 install --no-cache-dir torch==1.13.1+cu117 --index-url https://download.pytorch.org/whl/cu117
pip3 install --no-cache-dir torch-geometric==2.4.0
pip3 install -—no-cache-dir torch-sparse==0.6.16 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip3 install -—no-cache-dir torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip3 install -—no-cache-dir torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
```

Else use the following.
```
pip3 install --no-cache-dir torch==1.13.1
pip3 install --no-cache-dir torch-geometric==2.4.0
pip3 install -—no-cache-dir torch-sparse==0.6.16 -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
pip3 install -—no-cache-dir torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
pip3 install -—no-cache-dir torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
```

Finally, setup the environment and install any auxiliary dependencies using
```
pip3 install -r requirements.txt
```

```txt

├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
├── docs                 <- Documentation folder
│   │
│   ├── index.md         <- Homepage for your documentation
│   │
│   ├── mkdocs.yml       <- Configuration file for mkdocs
│   │
│   └── source/          <- Source directory for documentation files
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks            <- Jupyter notebooks.
│
├── pyproject.toml       <- Project configuration file
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
├── requirements_dev.txt <- The requirements file for reproducing the analysis environment
│
├── tests                <- Test files
│
├── src  <- Source code for use in this project.
│   │
│   ├── __init__.py      <- Makes folder a Python module
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── model.py
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```

Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
