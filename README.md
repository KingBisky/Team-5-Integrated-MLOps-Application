# IT3385 Team 5 – MLOps Project

This repository showcases Team 5's Machine Learning Operations (MLOps) project for IT3385. The project brings three individual machine learning applications together through a shared Flask portal, supported by a reproducible development environment and an integrated deployment workflow.

The project demonstrates how machine learning applications can be developed, integrated, tested, version-controlled and deployed using a consistent MLOps workflow.

## Project Overview

The integrated portal features three machine learning applications:

- **Employee Burnout Predictor** – Kang Bin's application at `/kang-bin/`
- **Employee Salary Predictor** – Long Chen's application at `/long-chen/`
- **Mental Health Risk Predictor** – Clifton's application at `/clifton/`, supporting single and batch predictions

The shared project also incorporates:

- Poetry for dependency and virtual-environment management
- Hydra for runtime configuration
- DVC for dataset version tracking
- Pytest for automated testing
- Git and GitHub for collaborative development
- GitHub Actions for Continuous Integration (CI)
- PyCaret for model development and saved pipelines
- Docker and Google Cloud Run for deployment

> **Dependency management:** Poetry is the central dependency manager. `pyproject.toml` declares project dependencies, while `poetry.lock` records the resolved versions for reproducible installation. The project does not use separate `requirements.txt` files.

---

# A. Team Information

## Team 5

| Team Member | Dataset / Individual Contribution |
|---|---|
| **Kang Bin** | Employee Burnout Prediction – EDA, machine learning model, Flask application, and shared MLOps setup covering Conda, Poetry, Hydra, DVC, Git branching, Pytest and CI |
| **Clifton** | Mental Health Risk Prediction – EDA, machine learning model and Flask prediction application |
| **Long Chen** | Global AI Jobs / Employee Salary Prediction – EDA, machine learning model, Flask application, containerisation and production deployment to Google Cloud Run |

Each member developed an individual machine learning component using a separate dataset. The applications were integrated into one Team 5 Flask portal while retaining their own models, preprocessing pipelines, templates and static files.

---

# B. Project / Folder Structure

```text
IT3385_Team5_Assignment/
│
├── .dvc/
│   ├── config
│   └── .gitignore
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── config/
│   ├── main.yaml
│   ├── app/
│   │   └── default.yaml
│   └── server/
│       ├── local.yaml
│       └── development.yaml
│
├── data/
│   ├── raw/
│   │   ├── Kang Bin/
│   │   │   ├── .gitignore
│   │   │   └── tech_mental_health_burnout.csv.dvc
│   │   ├── Clifton/
│   │   │   ├── .gitignore
│   │   │   └── mental_health_risk_dataset.csv.dvc
│   │   └── Long Chen/
│   │       ├── .gitignore
│   │       └── global_ai_jobs.csv.dvc
│   ├── processed/
│   └── final/
│
├── docs/
│
├── notebooks/
│   ├── Kang Bin/
│   │   └── KangBin_Task1&2_Final.ipynb
│   └── Long Chen/
│       └── LongChen_Task1&2_Final.ipynb.ipynb
│
├── src/
│   └── team5_app/
│       ├── app.py
│       ├── static/
│       │   └── portal.css
│       ├── templates/
│       │   ├── index.html
│       │   └── placeholder.html
│       │
│       ├── Kang Bin/
│       │   └── employee_burnout_app/
│       │       ├── app.py
│       │       ├── employee_burnout_final_model.pkl
│       │       ├── static/
│       │       └── templates/
│       │
│       ├── Clifton/
│       │   └── mental_health_risk_app/
│       │       ├── app.py
│       │       ├── clifton_mental_health_risk_final_model.pkl
│       │       ├── samples/
│       │       ├── static/
│       │       └── templates/
│       │
│       └── Long Chen/
│           └── salary_predictor_app/
│               ├── app.py
│               ├── employee_salary_final_model.pkl
│               ├── schema.py
│               ├── static/
│               └── templates/
│
├── tests/
│   └── test_environment.py
│
├── .gitignore
├── poetry.lock
├── pyproject.toml
└── README.md
```

## Important Files and Folders

| Path | Purpose |
|---|---|
| `src/team5_app/app.py` | Main entry point for the Team 5 portal and integrated Flask applications |
| `src/team5_app/Kang Bin/employee_burnout_app/` | Employee Burnout Predictor and trained classification model |
| `src/team5_app/Long Chen/salary_predictor_app/` | Employee Salary Predictor and trained regression model |
| `src/team5_app/Clifton/mental_health_risk_app/` | Mental Health Risk Predictor, including single and batch prediction |
| `config/` | Hydra configuration for application and server settings |
| `data/raw/` | Raw datasets and their DVC metadata |
| `notebooks/` | EDA and model-development notebooks |
| `tests/test_environment.py` | Automated environment, project-structure and Hydra configuration tests |
| `.github/workflows/ci.yml` | GitHub Actions CI workflow |
| `pyproject.toml` | Direct Python dependencies and version constraints |
| `poetry.lock` | Exact resolved dependency versions for reproducible installation |

## MLOps Tools Used

| Tool | Purpose |
|---|---|
| Cookiecutter | Initial standard ML project structure |
| Conda | Base Python 3.10 interpreter |
| Poetry | Dependency and project virtual-environment management |
| Hydra | Reusable server and application runtime configuration |
| DVC | Dataset version tracking through `.dvc` metadata |
| Git | Source-code version control |
| GitHub | Shared repository and Pull Request collaboration |
| GitHub Actions | Continuous Integration |
| Pytest | Automated environment and configuration testing |
| Flask | Portal and web application framework |
| PyCaret | Machine learning development and saved pipelines |
| Jupyter | EDA and model experimentation |
| Docker | Application containerisation |
| Google Cloud Run | Production hosting |

---

# C. Deployment Guide

This section documents how to recreate the development environment, run the integrated application and deploy updates.

## 1. Prerequisites

Install:

- Git
- Anaconda or Miniconda
- Python 3.10 through Conda

Check Git:

```bash
git --version
```

## 2. Clone the Repository

Clone the repository to retain its Git history, branches and collaboration workflow.

Navigate to your preferred directory, for example:

```bash
cd Desktop
```

Clone the repository:

```bash
git clone https://github.com/KingBisky/IT3385_Team5_Assignment.git
```

Enter the project folder:

```bash
cd IT3385_Team5_Assignment
```

Verify the repository:

```bash
git status
git remote -v
```

The configured remote should point to:

```text
https://github.com/KingBisky/IT3385_Team5_Assignment.git
```

## 3. Create the Conda Environment

Create and activate the Python 3.10 environment:

```bash
conda create -n mlops_assignment python=3.10
conda activate mlops_assignment
```

Verify the Python version:

```bash
python --version
```

Expected version:

```text
Python 3.10.x
```

### Conda and Poetry Responsibilities

The project separates interpreter management from package management:

```text
Conda
  └── Provides the Python 3.10 interpreter

Poetry
  ├── Manages the project virtual environment
  ├── Installs Python dependencies
  ├── Resolves package compatibility
  └── Reproduces versions recorded in poetry.lock
```

Conda provides the base interpreter; Poetry manages the project's Python packages.

## 4. Install Poetry

Inside the activated Conda environment:

```bash
conda install -c conda-forge poetry
```

Verify:

```bash
poetry --version
```

## 5. Link Poetry to the Conda Interpreter

Find the active Python interpreter:

```bash
python -c "import sys; print(sys.executable)"
```

Example Windows path:

```text
C:\Users\<USERNAME>\anaconda3\envs\mlops_assignment\python.exe
```

Configure Poetry to use the path returned on your computer:

```bash
poetry env use "C:\Users\<USERNAME>\anaconda3\envs\mlops_assignment\python.exe"
```

Verify the environment:

```bash
poetry env info
```

## 6. Dependency Management with Poetry

### 6.1 Dependency Files

Poetry is the central dependency manager for the Team 5 project.

| File | Purpose |
|---|---|
| `pyproject.toml` | Declares direct dependencies and version constraints |
| `poetry.lock` | Records the exact resolved dependency versions |

Install the committed environment with:

```bash
poetry install --no-root
```

The project does not use separate `requirements.txt` files. Avoid manually installing project dependencies with pip, as this can cause differences between local environments and the committed lock file.

Use Poetry to manage dependency changes:

```bash
poetry add <package>
poetry remove <package>
poetry update <package>
```

Whenever dependency resolution changes, commit both `pyproject.toml` and `poetry.lock`.

### 6.2 Dependency Compatibility

The repository contains saved PyCaret model pipelines (`.pkl` files). These pipelines rely on compatible versions of machine learning and preprocessing libraries, including scikit-learn, pandas, NumPy, LightGBM and category-encoders.

Important project versions include:

| Dependency | Project Version / Constraint |
|---|---:|
| Python | `>=3.10,<3.11` |
| Flask | `3.0.3` |
| PyCaret | `3.3.2` |
| pandas | `2.1.4` |
| NumPy | `1.26.4` |
| scikit-learn | `1.4.2` |
| category-encoders | `2.7.0` |
| LightGBM | `4.6.0` |
| CatBoost | `1.2.5` |

> `lightgbm==4.6.0` and `category-encoders==2.7.0` are intentional compatibility pins for the saved Employee Burnout pipeline. Changes to these versions should be followed by model-loading and prediction tests.

The committed `poetry.lock` remains the authoritative source for the complete dependency graph, including transitive packages.

### 6.3 Install and Verify Dependencies

Install:

```bash
poetry install --no-root
```

Verify the main packages:

```bash
poetry run python -c "import pycaret, flask, pandas, numpy, sklearn, hydra, lightgbm, category_encoders; print('TEAM 5 ENVIRONMENT OK'); print('LightGBM:', lightgbm.__version__); print('category-encoders:', category_encoders.__version__)"
```

Check the Poetry configuration:

```bash
poetry check
```

### 6.4 Dependency Management Guidelines

**Recommended practices**

- Use `poetry install --no-root` after cloning or pulling dependency changes.
- Use `poetry add` and `poetry remove` to manage packages.
- Pin versions when required for saved model compatibility.
- Commit both Poetry dependency files after changes.
- Run automated tests and check the integrated application after dependency updates.

**Avoid**

- Using pip as a replacement for Poetry.
- Manually editing or deleting `poetry.lock`.
- Running `poetry update` without considering compatibility.
- Upgrading PyCaret, scikit-learn, LightGBM, CatBoost, NumPy, pandas or encoders without retesting saved models.
- Creating separate component-level `requirements.txt` files.

### 6.5 Common Poetry Commands

| Action | Command |
|---|---|
| Install committed environment | `poetry install --no-root` |
| Add a dependency | `poetry add <package>` |
| Add an exact version | `poetry add "<package>==<version>"` |
| Add development dependency | `poetry add --group dev <package>` |
| Remove a dependency | `poetry remove <package>` |
| Update one dependency | `poetry update <package>` |
| Update allowed dependencies | `poetry update` |
| Show installed dependencies | `poetry show` |
| Show a specific package | `poetry show <package>` |
| Validate configuration | `poetry check` |
| Inspect environment | `poetry env info` |

### 6.6 Dependency Update Workflow

Create a feature branch before making dependency changes:

```bash
git switch main
git pull origin main
git switch -c feature/dependency-update
```

Make the change using Poetry. For example:

```bash
poetry add "lightgbm==4.6.0" "category-encoders==2.7.0"
```

Install and test:

```bash
poetry install --no-root
poetry run pytest tests -v
poetry run python src/team5_app/app.py
```

Review and commit both dependency files:

```bash
git diff -- pyproject.toml poetry.lock
git add pyproject.toml poetry.lock
git commit -m "Update model dependency compatibility"
git push -u origin feature/dependency-update
```

Create a Pull Request into `main` after pushing.

When pulling dependency changes made by another team member, synchronise your environment with:

```bash
git pull origin main
poetry install --no-root
poetry run pytest tests -v
```

### 6.7 Troubleshooting

Inspect the Python version and Poetry environment:

```bash
poetry run python --version
poetry env info
```

Check important ML package versions:

```bash
poetry run python -c "import pycaret, sklearn, lightgbm, category_encoders; print('PyCaret:', pycaret.__version__); print('scikit-learn:', sklearn.__version__); print('LightGBM:', lightgbm.__version__); print('category-encoders:', category_encoders.__version__)"
```

If the local environment differs from the repository:

1. Pull the latest changes.
2. Confirm both Poetry files are present.
3. Run `poetry install --no-root`.
4. Run the test suite.
5. Test an actual prediction through the application.

## 7. Run Automated Tests

Run the test suite:

```bash
poetry run pytest tests -v
```

The tests cover:

- Required project files and folders
- DVC metadata for all three datasets
- Hydra default configuration composition
- Default server configuration
- Application runtime configuration
- Hydra command-line overrides

The recorded test result was:

```text
3 passed
```

## 8. Load the Models and Start the Web Application

The trained model artefacts are stored in their respective application folders:

```text
src/team5_app/Kang Bin/employee_burnout_app/employee_burnout_final_model.pkl

src/team5_app/Clifton/mental_health_risk_app/clifton_mental_health_risk_final_model.pkl

src/team5_app/Long Chen/salary_predictor_app/employee_salary_final_model.pkl
```

The saved models allow the applications to generate predictions without retraining.

From the repository root, start the integrated portal:

```bash
poetry run python src/team5_app/app.py
```

Open the local application:

```text
http://127.0.0.1:5000
```

| Route | Application |
|---|---|
| `/` | Team 5 portal |
| `/kang-bin/` | Employee Burnout Predictor |
| `/long-chen/` | Employee Salary Predictor |
| `/clifton/` | Mental Health Risk Predictor |

## 9. Hydra Runtime Configuration

Hydra separates runtime settings from application code.

```text
config/
├── main.yaml
├── app/
│   └── default.yaml
└── server/
    ├── local.yaml
    └── development.yaml
```

### Default Application Settings

`config/app/default.yaml`:

```yaml
max_upload_mb: 32

batch:
  result_ttl_seconds: 21600
  chunk_size: 5000
  preview_rows: 20
```

These settings control the maximum CSV upload size, batch-result lifetime, processing chunk size and preview row count. They are applied to the integrated Employee Burnout Predictor through its runtime configuration function.

### Local Server Configuration

`config/server/local.yaml`:

```yaml
host: "127.0.0.1"
port: 5000
use_reloader: false
use_debugger: false
threaded: true
```

### Development Server Profile

Run the development profile with:

```bash
poetry run python src/team5_app/app.py server=development
```

This uses the Flask/Werkzeug reloader and debugger settings defined in `config/server/development.yaml`.

### Command-Line Overrides

Change the server port:

```bash
poetry run python src/team5_app/app.py server.port=5050
```

Change the batch preview size:

```bash
poetry run python src/team5_app/app.py app.batch.preview_rows=5
```

Combine multiple overrides:

```bash
poetry run python src/team5_app/app.py server.port=5050 app.max_upload_mb=64 app.batch.chunk_size=10000 app.batch.preview_rows=50
```

Hydra uses `hydra.job.chdir: false`, so starting the application does not change the process working directory.

## 10. Jupyter Notebook Setup

The notebooks use the same Poetry-managed environment as the web application.

Register a Jupyter kernel:

```bash
poetry run python -m ipykernel install --user --name it3385-team5-poetry --display-name "Python (IT3385 Team 5 - Poetry)"
```

Launch Jupyter:

```bash
poetry run jupyter notebook
```

Select the kernel:

```text
Python (IT3385 Team 5 - Poetry)
```

Project notebooks:

```text
notebooks/Kang Bin/KangBin_Task1&2_Final.ipynb

notebooks/Long Chen/LongChen_Task1&2_Final.ipynb.ipynb
```

Using the Poetry kernel helps keep notebook experiments consistent with the application's package environment.

## 11. DVC Data Version Control

DVC tracks the raw machine learning datasets through metadata files committed to Git. The full CSV files are excluded from normal Git tracking.

Tracked DVC metadata:

```text
data/raw/Kang Bin/tech_mental_health_burnout.csv.dvc
data/raw/Clifton/mental_health_risk_dataset.csv.dvc
data/raw/Long Chen/global_ai_jobs.csv.dvc
```

Corresponding local dataset files:

```text
data/raw/Kang Bin/tech_mental_health_burnout.csv
data/raw/Clifton/mental_health_risk_dataset.csv
data/raw/Long Chen/global_ai_jobs.csv
```

Check dataset tracking status:

```bash
poetry run dvc status
```

Example of updating a dataset:

```bash
poetry run dvc add "data/raw/Kang Bin/tech_mental_health_burnout.csv"
git add "data/raw/Kang Bin/tech_mental_health_burnout.csv.dvc"
git commit -m "Update Kang Bin dataset version"
```

### DVC Storage Limitation

The repository uses local DVC tracking and does not currently define a shared DVC remote. Cloning the Git repository therefore provides the `.dvc` metadata but does not automatically download the full raw datasets.

The saved `.pkl` model artefacts are stored with the applications, allowing predictions to run without retrieving the raw training datasets or retraining the models.

## 12. Git and Team Collaboration Workflow

The `main` branch serves as the stable integration branch. Development work is organised through feature branches.

Start new work:

```bash
git switch main
git pull origin main
git switch -c feature/<branch-name>
```

Example branches:

```bash
git switch -c feature/kang-bin
git switch -c feature/clifton
git switch -c feature/long-chen
```

Commit and push changes:

```bash
git status
git add .
git commit -m "Describe the changes made"
git push -u origin feature/<branch-name>
```

After pushing, create a Pull Request on GitHub:

1. Set `base: main`.
2. Set `compare: feature/<branch-name>`.
3. Review the changed files.
4. Allow GitHub Actions CI to run.
5. Merge after the checks pass.

Synchronise the local main branch after merging:

```bash
git switch main
git pull origin main
```

## 13. Continuous Integration

The GitHub Actions workflow is defined in:

```text
.github/workflows/ci.yml
```

It runs for:

- Pushes to `main`
- Pushes to `feature/**`
- Pull Requests targeting `main`

The CI workflow follows this process:

```text
Git push / Pull Request
        ↓
GitHub Actions
        ↓
Checkout repository
        ↓
Set up Python 3.10
        ↓
Install Poetry
        ↓
poetry install --no-root
        ↓
Verify MLOps packages and DVC
        ↓
poetry run pytest tests -v
        ↓
PASS / FAIL
```

This provides an automated check that the shared environment can be installed and the project tests can pass before changes are merged.

## 14. Deployment to Google Cloud Run

The integrated Team 5 application is containerised with Docker and deployed to Google Cloud Run. Its Python dependencies are managed through Poetry and the committed lock file.

The deployment environment uses:

- Python 3.10
- Poetry-managed dependencies
- Saved `.pkl` model artefacts
- An application host and port configured for the deployment platform
- Google Cloud Run for production hosting

### Redeploying After Changes

The following steps describe how to build and deploy an updated application.

**Step 1: Install the Google Cloud CLI**

Installation guide:

https://cloud.google.com/sdk/docs/install

**Step 2: Navigate to the project root**

```bash
cd [your-path]/IT3385_Team5_Assignment
```

**Step 3: Confirm the required files are present**

```bash
dir Dockerfile pyproject.toml poetry.lock
```

**Step 4: Authenticate and select the Google Cloud project**

```bash
gcloud auth login
gcloud config set project it3385-team5-assignment1
```

**Step 5: Build the container image**

```bash
gcloud builds submit --tag gcr.io/it3385-team5-assignment1/team5-app
```

**Step 6: Deploy to Cloud Run**

```bash
gcloud run deploy team5-app \
  --image gcr.io/it3385-team5-assignment1/team5-app \
  --platform managed \
  --region asia-southeast1 \
  --memory 2Gi \
  --cpu 2 \
  --port 8080 \
  --allow-unauthenticated
```

Cloud Run creates a new revision using the updated image. The service retains its existing production URL.

---

# D. Users Manual

## 1. Access the Team 5 Portal

<https://team5-app-873480729550.asia-southeast1.run.app>

The home page provides access to all three integrated machine learning applications.

The portal can also be run locally:

```bash
poetry run python src/team5_app/app.py
```

Local URL:

```text
http://127.0.0.1:5000
```

See Section C for environment setup instructions.

## 2. Kang Bin – Employee Burnout Predictor

**Application URL:**

<https://team5-app-873480729550.asia-southeast1.run.app/kang-bin/>

Local route:

```text
http://127.0.0.1:5000/kang-bin/
```

Model artefact:

```text
src/team5_app/Kang Bin/employee_burnout_app/employee_burnout_final_model.pkl
```

### Single Prediction

1. Open the Employee Burnout Predictor.
2. Complete the employee profile, workload/environment and wellbeing fields.
3. Select categorical values from the available options.
4. Enter numeric values within the displayed ranges.
5. Submit the form to generate a prediction.

The application validates all 23 model inputs before passing them to the saved classification pipeline.

**Prediction output:**

- Low
- Moderate
- High
- Prediction confidence, when provided by the model pipeline

The output is a model prediction based on the submitted features and should not be interpreted as a medical diagnosis.

### Batch Prediction

The application also supports CSV-based batch prediction:

1. Open the batch prediction section.
2. Download or use the provided CSV template.
3. Enter one employee per row, keeping the required column names unchanged.
4. Upload the CSV file.
5. Start batch prediction.
6. Review the result preview and summary.
7. Download the generated result CSV.

Batch output can include:

```text
predicted_burnout_class
predicted_burnout_level
prediction_confidence
```

Default Hydra batch settings:

| Setting | Value |
|---|---:|
| Maximum upload | 32 MB |
| Chunk size | 5,000 rows |
| Preview | 20 rows |
| Result TTL | 21,600 seconds (6 hours) |

## 3. Long Chen – Employee Salary Predictor

**Application URL:**

<https://team5-app-873480729550.asia-southeast1.run.app/long-chen/>

Local route:

```text
http://127.0.0.1:5000/long-chen/
```

Model artefact:

```text
src/team5_app/Long Chen/salary_predictor_app/employee_salary_final_model.pkl
```

### Single Prediction

1. Open the Employee Salary Predictor.
2. Enter or select the requested role, experience, company, compensation and market-related inputs.
3. Submit or change the values as supported by the interface.
4. The saved regression pipeline generates the salary estimate.

**Prediction output**

The application returns an estimated salary in USD, formatted as a dollar value.

Example:

```text
$120,000
```

The result is a model estimate based on the supplied features and training data, not a guaranteed salary offer or market quote.

### Batch Prediction

1. Open the Long Chen batch page.
2. Upload a CSV containing the required model columns.
3. Start batch prediction.
4. Review the generated results.
5. Download the salary prediction CSV.

The batch output includes:

```text
predicted_salary_usd
```

## 4. Clifton – Mental Health Risk Predictor

**Application URL:**

<https://team5-app-873480729550.asia-southeast1.run.app/clifton/>

Local route:

```text
http://127.0.0.1:5000/clifton/
```

Model artefact:

```text
src/team5_app/Clifton/mental_health_risk_app/clifton_mental_health_risk_final_model.pkl
```

### Single Prediction

1. Open Clifton's Mental Health Risk Predictor.
2. Enter the nine validated mental-health and lifestyle signals.
3. Submit the profile.
4. The saved PyCaret classification pipeline predicts the risk category.
5. Review the predicted category, confidence and per-class probabilities when available.

Possible prediction categories:

- Low
- Moderate
- High

Numeric and binary inputs are validated on the server before inference.

### Batch Prediction

The application also supports batch prediction through:

```text
/clifton/batch
```

Steps:

1. Open the batch prediction route.
2. Download the CSV template or use the sample file:

   ```text
   src/team5_app/Clifton/mental_health_risk_app/samples/clifton_batch_demo.csv
   ```

3. Upload the CSV.
4. The application validates the schema and individual rows.
5. Predictions are generated in configured chunks.
6. Review the Low/Moderate/High counts, mean confidence and preview.
7. Download the full result CSV.

The output includes the predicted class, label and confidence.

### Health Check and Smoke Testing

The endpoint:

```text
/clifton/health
```

performs a deeper saved-model loading check.

A deployment smoke-test script is also available:

```bash
python scripts/verify_clifton_deployment.py https://team5-app-873480729550.asia-southeast1.run.app
```

It provides repeatable checks for Clifton's single and batch prediction endpoints.

---

# E. Project URLs

## Source Code Repository

**GitHub Repository:**

<https://github.com/KingBisky/IT3385_Team5_Assignment>

The repository contains the source code, project configuration, model application files and MLOps workflow.

## Deployed Team 5 Application

**Google Cloud Run:**

<https://team5-app-873480729550.asia-southeast1.run.app>

The deployed portal provides access to the three integrated machine learning applications.

---

# MLOps Lifecycle Summary

The project brings together the following workflow:

```text
Raw datasets
    ↓
DVC metadata and version tracking
    ↓
Jupyter + PyCaret model development
    ↓
Conda Python 3.10 interpreter
    ↓
Poetry dependency management
    ↓
pyproject.toml + poetry.lock
    ↓
Hydra runtime configuration
    ↓
Saved PyCaret pipelines
    ↓
Integrated Flask applications
    ↓
Pytest validation
    ↓
Feature branches
    ↓
GitHub push / Pull Request
    ↓
GitHub Actions CI
    ↓
Merge to main
    ↓
Docker containerisation
    ↓
Google Cloud Run deployment
    ↓
Live Team 5 application
```

The completed project demonstrates the integration of three machine learning applications with shared dependency management, configurable runtime settings, dataset version tracking, automated testing, collaborative Git workflows, Continuous Integration and cloud deployment.
