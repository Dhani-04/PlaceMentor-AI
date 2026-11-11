# PlaceMentor AI

**Student Skill Profiling and Placement Readiness Analysis Using K-Means Clustering**

PlaceMentor AI is an interactive machine learning mini project built with Python and Streamlit. It groups students with similar academic and skill profiles using **K-Means clustering**, shows how a new student compares with those groups, and suggests practical skill-development steps.

> **Important:** The dataset is **simulated** (not real institutional placement data), and the app performs **exploratory profile clustering**. It does **not** predict placement selection or probability.

## Problem statement
Students often lack a structured way to understand their academic performance, technical skills and employability-related strengths. This project uses unsupervised learning to discover groups of students with similar profiles and provides understandable recommendations.

## Objectives
- Analyse seven academic and skill indicators per student.
- Group similar profiles with K-Means.
- Evaluate the clustering (elbow method, silhouette score, PCA).
- Identify improvement areas and give recommendations based on the student's own values.

## Features
| Module | What it does |
|---|---|
| Overview | Live dataset statistics, silhouette score, cluster distribution, PCA plot, quick navigation |
| Student Assessment | Form with validation, nearest-cluster assignment, centroid distance, recommendations, session history |
| Cluster Explorer | Cluster sizes, averages, comparison charts, student records, CSV download |
| Model Evaluation | Elbow method (k = 2 to 8), silhouette scores, PCA, centroids, cluster sizes |
| Skill Gap Analysis | Compares a profile with illustrative benchmarks and gives value-specific advice |
| Student History | Session-only records, CSV export, clear with confirmation |
| About the Project | Problem, objectives, algorithm, workflow, stack, dataset, limitations, future work |

## Technology stack
Python, Streamlit, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib, Git, GitHub.

## Machine learning methodology
1. Load the dataset and validate columns, ranges and missing values.
2. Select seven features: CGPA, Aptitude, Coding, Communication, Projects, Internships, Attendance.
3. Standardise with `StandardScaler`.
4. Compute inertia (WCSS) and silhouette score for k = 2 to 8.
5. Train the final K-Means model with k = 3.
6. Name the clusters **after** clustering by comparing centroids: *Career-ready profile*, *Developing talent*, *Skill-building priority*. K-Means itself learns no labels.
7. Use PCA on the scaled data for 2-D visualisation only.
8. Save the scaler and model together as one scikit-learn `Pipeline` with Joblib (`models/kmeans_pipeline.joblib`).

The scaler and K-Means live in a single pipeline, so new input is transformed with the training mean and standard deviation (never refitted), which avoids preprocessing mismatch and leakage. Because there are no ground-truth labels, accuracy, precision, recall and confusion matrices are intentionally **not** reported.

If `k = 3` is not the highest-silhouette choice in the tested range, the Model Evaluation page says so and explains why three interpretable groups were kept.

## Dataset
`dataset/student_data.csv` holds 180 simulated records generated with a fixed random seed (42) from three overlapping student archetypes with noise. Columns: `CGPA, Aptitude, Coding, Communication, Projects, Internships, Attendance`. There are no missing values and all values are within valid ranges. Regenerate it with `python train_model.py --regenerate`.

## Installation and running in VS Code (Windows)
```powershell
cd PlaceMentor-AI
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```
On macOS or Linux, activate with `source venv/bin/activate`. The app opens at http://localhost:8501.

`python train_model.py` is optional: if the dataset or model is missing, or the saved model was built with a different scikit-learn version, **the app regenerates and retrains automatically** on start-up.

## Project structure
```
PlaceMentor-AI/
├── app.py                   Streamlit application
├── train_model.py           Training script
├── requirements.txt
├── .streamlit/config.toml   Theme
├── dataset/student_data.csv
├── models/kmeans_pipeline.joblib
├── outputs/clustered_students.csv
└── utils/
    ├── preprocessing.py     Features, validation, simulated data
    ├── clustering.py        Training, evaluation, prediction, persistence
    └── recommendations.py   Skill-gap analysis and advice text
```

## Upload to GitHub
Create an empty repository named `PlaceMentor-AI` on GitHub, then run:
```bash
git init
git add .
git commit -m "Initial PlaceMentor AI project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/PlaceMentor-AI.git
git push -u origin main
```

## Deploy on Streamlit Community Cloud
1. Sign in at https://share.streamlit.io with your GitHub account.
2. Choose **Create app** and connect the `PlaceMentor-AI` repository.
3. Branch: `main`. Main file path: `app.py`.
4. Click **Deploy**.

The trained model is committed to the repository. If it is ever missing or incompatible with the cloud's scikit-learn version, the app retrains it automatically in a few seconds.

## Limitations
- Simulated data: results describe artificial students only.
- Clusters show profile similarity, not placement outcomes.
- K-Means assumes roughly round, similar-sized groups and is sensitive to scaling and outliers.
- Silhouette scores indicate moderate separation, so group borders overlap.
- Skill-gap benchmarks are illustrative, not official criteria.
- History is stored only for the current browser session (no database).

## Future enhancements
- Use anonymised real data with consent.
- Compare with hierarchical clustering or Gaussian mixtures.
- Add a database for permanent history.
- Add indicators such as certifications and soft-skill assessments.
- Track a student's cluster movement across semesters.
