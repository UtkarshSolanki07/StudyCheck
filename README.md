# 📚 Student Success Companion

A practical Streamlit application to explore how student habits relate to academic performance. The project includes:

- 🌐 An interactive web app to predict an exam score from daily habits
- 📊 A synthetic dataset of student habits and outcomes
- 📓 A notebook for data exploration and model development
- 🤖 A packaged model with a robust fallback so the app always runs

## 📋 Table of Contents

- [Repository Structure](#-repository-structure)
- [App Features](#-app-features)
- [Inputs and Features](#-inputs-and-features)
- [Dataset](#-dataset)
- [Notebook](#-notebook)
- [Getting Started](#-getting-started)
- [Model Details](#-model-details)
- [Privacy](#-privacy)
- [Reproducibility Checklist](#-reproducibility-checklist)
- [Deployment Options](#-deployment-options)
- [Screenshots](#-screenshots)


## 📁 Repository Structure

- `app.py` — Streamlit app with prediction, target planning, and habit tracking
- `notebook.ipynb` — EDA, feature engineering, and model training workflow
- `student_habits_performance.csv` — Synthetic dataset used for EDA/modeling
- `exported.csv` — Any exported, intermediate data from the notebook (optional)
- `final_student_performance_model.pkl` — Trained model artifact used by the app
- `requirements.txt` — Python dependencies


## ✨ App Features

### 1. 📊 Dashboard
- Instant prediction of a student performance score (0–100) based on current inputs
- Quick KPIs for study time, attendance, sleep
- Actionable recommendations on study routines, sleep hygiene, and leisure management

### 2. 🔮 Predict
- Simple form-based predictor; shows the predicted performance as a single score

### 3. 🎯 Target Planner
- Compares your current predicted score to a user-selectable target
- What-if sliders to increase study hours, attendance, and sleep to see impact
- Generates a concise action plan to close the gap

### 4. 📈 Habit Tracker
- Lightweight in-session log of daily habits (study, sleep, attendance, leisure, mental health)
- Auto charts for trends and a quick correlation view

### 5. ℹ️ About
- Summarizes model, features, and privacy approach (all-local)


## Inputs and features

The app uses the following features as inputs:
- study_hours: daily study time (hours)
- attendance: attendance percentage (0–100)
- mental_health: self-rating 1–10
- sleep_hours: nightly sleep (hours)
- part_time_job: Yes/No
- leisure_platforms: multi-select of platforms weighted by distraction potential
- leisure_hours: daily leisure (hours)
- caffeine_per_day: number of caffeinated drinks per day

Internally, the app builds a feature vector that adapts to the loaded model’s expected shape. If the trained model is missing or incompatible, the app falls back to an interpretable heuristic that emphasizes study time, attendance, sleep, and mental health, with penalties for heavy leisure, distractions, caffeine, and part-time work under low study hours.


## 📈 Dataset

**File:** `student_habits_performance.csv`
- 2,000 synthetic student records
- Columns include demographics (age, gender), lifestyle and habit variables (`study_hours_per_day`, `social_media_hours`, `netflix_hours`, `part_time_job`, `attendance_percentage`, `sleep_hours`, `diet_quality`, `exercise_frequency`, `internet_quality`, `mental_health_rating`, `extracurricular_participation`), and the target `exam_score`.
- Use this file for EDA and for training/validating models in the notebook.

> **Note:** The Streamlit app focuses on a practical subset of features for user inputs. Your trained model may use engineered features derived from the full dataset.


## Notebook

File: notebook.ipynb
- Typical flow:
  - Load and inspect dataset
  - Clean/encode features, engineer inputs relevant to performance
  - Train and evaluate candidate models
  - Persist the final model to final_student_performance_model.pkl
- You can export intermediate tables to exported.csv if convenient.

Tip: Ensure the model’s expected input dimension aligns with the app. The app gracefully adapts to models using between 5 and 8 features; if your model expects more, the app pads/truncates to fit safely.


## 🚀 Getting Started

1. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. **Using the app**
   - Adjust sliders and selections in the sidebar to reflect current habits
   - Visit **Dashboard** for an overview and quick recommendations
   - Use **Predict** for a single-score output
   - Use **Target Planner** to simulate changes and generate an action plan
   - Use **Habit Tracker** to log a few days and inspect trend charts and correlation


## Model details

- Default model artifact: final_student_performance_model.pkl
- The app checks for a predict method and n_features_in_. If anything fails, a built-in heuristic model is used and a non-blocking warning is shown in the UI.
- This design ensures the app is demo-friendly and does not break if the artifact is absent.

Feature adaptation logic (high-level):
- Primary 8-feature vector: [study_hours, attendance, mental_health, sleep_hours, ptj_encoded, leisure_hours, distractions_index, caffeine_per_day]
- If the model reports n_features_in_ == 5, the app uses [study, attendance, mental, sleep, ptj]
- If n_features_in_ <= 8, the app slices the first n features
- If n_features_in_ > 8, the app pads/truncates to match, ensuring safe inference


## 🔒 Privacy

All computation happens locally in your browser session via Streamlit. No external network calls are made by the app.


## Reproducibility checklist

- Seed and reproducibility controls are handled in the notebook during modeling, if required
- Requirements are pinned in requirements.txt
- Model artifact is committed (for demo), but you can regenerate it from the notebook


## 🌐 Deployment Options

- **Local:** `streamlit run app.py`
- **Streamlit Community Cloud:** Push this repo to GitHub and point Streamlit to `app.py`; ensure `requirements.txt` is present
- **Container:** Create a minimal Dockerfile (Python base image) and run streamlit in container

**Example Dockerfile (optional):**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```


## Screenshots



