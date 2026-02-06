### Environment Setup

This project is written in Python and follows a standard modular ML project structure.

### Prerequisites
- Python 3.9 or newer
- pip

### Setup Steps

1. Clone the repository:
    ```bash
    git clone git@github.com:sidcsebhu/kaggle.git
    cd titanic
    ```
2. Create Virtual Environment
    ```bash
    python -m venv .venv
    source .venv/bin/activate        # Linux / macOS
    .venv\Scripts\activate           # Windows
3. Install dependencies
    ```bash
    pip install -r requirements.txt
4. Verify Setup
    ```bash
    python -c "import pandas, sklearn; print('Setup OK')"
5. Running
    ```bash
    python src/main.py
    ```
This will:
- load raw data from data/raw/
- preprocess data (leak-free)
- train the model
- generate a submission CSV in submissions/

### Notebooks

Notebooks under `notebooks/` are used for exploration and experimentation only.

All reusable and score-affecting logic lives in `src/` to ensure:
- reproducibility
- clean version control
- leak-free experimentation

### Data

This project uses the Kaggle Titanic dataset.

- `data/raw/train.csv`
- `data/raw/test.csv`


## Project Status

This project is considered complete.

The focus was on building a clean, leak-free, reproducible ML pipeline rather than maximizing the Kaggle leaderboard score.
Several feature engineering ideas (e.g., Title extraction, HasDeck) were evaluated and accepted/dropped based on generalization benefit.

The repository is frozen in its current state.

Best Kaggle Score achieved: 0.77990

