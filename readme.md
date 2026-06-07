## Project Structure
project/
│
├── data/
│   ├── raw/
│   │   ├── cats/
│   │   ├── dogs/
│   │
│   ├── processed/
│
├── notebooks/
│   ├── cat_dog_classification.ipynb
│
├── src/
│   ├── data_ingestion.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│
├── models/
│   ├── best_model.keras
│
├── configs/
│   ├── config.yaml
│
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│
├── pipeline/
│   ├── train_pipeline.py
│   ├── inference_pipeline.py
│
├── Dockerfile
├── requirements.txt
├── setup.py
│
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── cd.yml
│
└── README.md

