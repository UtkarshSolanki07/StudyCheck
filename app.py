import streamlit as st
import numpy as np
import pandas as pd
import joblib
import warnings
from typing import Dict, List

warnings.filterwarnings('ignore')

st.set_page_config(page_title='Student Success Companion', page_icon='🎓', layout='wide')

@st.cache_resource(show_spinner=False)
def load_model(path: str = 'final_student_performance_model.pkl'):
    """Load model with graceful fallback if missing/invalid."""
    try:
        model = joblib.load(path)
        _ = getattr(model, 'predict')
        return model
    except Exception:
        st.warning('Model could not be loaded. Using a simple heuristic fallback.')

        class FallbackModel:
            n_features_in_ = 8

            def predict(self, X):
                
                X = np.asarray(X)
                study = X[:, 0]
                attend = X[:, 1]
                mental = X[:, 2]
                sleep = X[:, 3]
                ptj = X[:, 4]
                leisure = X[:, 5]
                distract = X[:, 6]
                caffeine = X[:, 7]
                score = (
                    0.45 * (study / 6.0 * 100).clip(0, 100)
                    + 0.35 * attend
                    + 0.10 * (mental / 10.0 * 100)
                    + 0.10 * (np.clip(sleep, 6, 9) / 9.0 * 100)
                )
                
                score -= np.clip(leisure - 2, 0, None) * 3
                score -= np.clip(distract - 1, 0, None) * 5
                score -= np.clip(caffeine - 2, 0, None) * 2
                score -= (ptj * np.where(study < 3, 5, 0))
                return np.clip(score, 0, 100)

        return FallbackModel()


def platform_distraction_score(selected: List[str]) -> float:
    if not selected:
        return 0.0
    weights = [LEISURE_TO_FEATURE.get(x, 1.0) for x in selected]
    return float(np.mean(weights))


DEFAULT_LEISURE_PLATFORMS = [
    'YouTube', 'Netflix', 'Prime Video', 'Disney+', 'Local TV', 'Gaming', 'Social Media', 'Other'
]

LEISURE_TO_FEATURE = {
    'YouTube': 1.0,
    'Netflix': 1.0,
    'Prime Video': 0.9,
    'Disney+': 0.9,
    'Local TV': 0.6,
    'Gaming': 1.2,
    'Social Media': 1.1,
    'Other': 0.8,
}

RECOMMENDATIONS = {
    'study_hours': [
        'Block 90-minute deep work sessions with 10–15 minute breaks.',
        'Use Pomodoro (25/5) and ramp to longer focus blocks over 2 weeks.',
        'Move study sessions to high-energy periods (morning if possible).',
    ],
    'attendance': [
        'Schedule classes and commute in a calendar with reminders.',
        'Coordinate with a buddy for accountability and shared notes.',
        'If remote, attend live when possible to ask questions early.',
    ],
    'mental_health': [
        'Add 10–15 minutes of mindfulness or journaling daily.',
        'Avoid doomscrolling 1 hour before bed to reduce anxiety.',
        'Seek counseling if persistent low mood or high stress.',
    ],
    'sleep_hours': [
        'Aim for a fixed sleep schedule with 7–9 hours.',
        'Avoid caffeine after 2 PM and heavy screens 1 hour before bed.',
        'Use blue-light filters and keep room cool and dark.',
    ],
    'leisure_hours': [
        'Batch leisure into a defined window after study blocks.',
        'Set app timers/limits for high-distraction platforms.',
        'Convert passive watch time into active learning occasionally.',
    ],
}


model = load_model()


def prepare_features_full(inputs: Dict) -> np.ndarray:
    """Return the extended 8-feature vector used by the fallback and optionally by richer models."""
    return np.array([
        [
            inputs['study_hours'],
            inputs['attendance'],
            inputs['mental_health'],
            inputs['sleep_hours'],
            1 if inputs['part_time_job'] == 'Yes' else 0,
            inputs['leisure_hours'],
            platform_distraction_score(inputs['leisure_platforms']),
            inputs['caffeine_per_day'],
        ]
    ])


def vector_for_model(model_obj, inputs: Dict) -> np.ndarray:
    """Adapt features to the model's expected shape.
    - If model has n_features_in_, slice/expand accordingly.
    - Default to using first 5 features for classic model: [study, attendance, mental, sleep, ptj].
    """
    full = prepare_features_full(inputs)
    cols_full = [
        'study_hours', 'attendance', 'mental_health', 'sleep_hours',
        'ptj_encoded', 'leisure_hours', 'distractions', 'caffeine'
    ]

    def five_feat():
        return full[:, [0, 1, 2, 3, 4]]

    n = getattr(model_obj, 'n_features_in_', None)
    if n is None:
        return five_feat()

    if n == 5:
        return five_feat()
    elif n <= 8:
        return full[:, :n]
    else:
        if full.shape[1] < n:
            pad = np.zeros((full.shape[0], n - full.shape[1]))
            return np.concatenate([full, pad], axis=1)
        return full[:, :n]




def input_form(defaults: Dict = None) -> Dict:
    defaults = defaults or {}

    with st.sidebar.expander('Leisure Platform Catalog', expanded=False):
        st.write('Platforms are weighted by distraction potential. Add custom items below:')
        custom = st.text_input('Add custom platform (press Enter to add)', '')
        if custom:
            if custom not in DEFAULT_LEISURE_PLATFORMS:
                DEFAULT_LEISURE_PLATFORMS.append(custom)

    col1, col2, col3 = st.columns(3)
    with col1:
        study_hours = st.slider('Study Hours per Day', 0.0, 12.0, float(defaults.get('study_hours', 2.0)), 0.5)
        sleep_hours = st.slider('Sleep Hours per Night', 0.0, 12.0, float(defaults.get('sleep_hours', 6.0)), 0.5)
        caffeine = st.slider('Caffeine drinks/day', 0, 10, int(defaults.get('caffeine_per_day', 1)))
    with col2:
        attendance = st.slider('Attendance Percentage', 0.0, 100.0, float(defaults.get('attendance', 85.0)), 1.0)
        mental_health = st.slider('Mental Health Score (1-10)', 1, 10, int(defaults.get('mental_health', 5)))
        part_time_job = st.selectbox(
            'Part-time Job', ['Yes', 'No'], index=(0 if defaults.get('part_time_job', 'No') == 'Yes' else 1)
        )
    with col3:
        leisure_platforms = st.multiselect(
            'Leisure Platforms Used', DEFAULT_LEISURE_PLATFORMS, defaults.get('leisure_platforms', ['YouTube', 'Social Media'])
        )
        leisure_hours = st.slider('Leisure Hours per Day', 0.0, 10.0, float(defaults.get('leisure_hours', 2.0)), 0.5)
        distractions = platform_distraction_score(leisure_platforms)
        st.caption(f'Distraction index from platforms: {distractions:.2f}')

    return {
        'study_hours': study_hours,
        'attendance': attendance,
        'mental_health': mental_health,
        'sleep_hours': sleep_hours,
        'part_time_job': part_time_job,
        'leisure_platforms': leisure_platforms,
        'leisure_hours': leisure_hours,
        'caffeine_per_day': caffeine,
    }




st.sidebar.title('Student Success Companion')
section = st.sidebar.radio('Navigate', ['Dashboard', 'Predict', 'Target Planner', 'Habit Tracker', 'About'])


if section == 'Dashboard':
    st.title('🎓 Student Success Companion')
    st.write('A practical toolkit to predict performance, plan targets, and optimize habits.')

    st.subheader('Quick Summary')
    inputs = input_form()
    X = vector_for_model(model, inputs)
    pred = float(model.predict(X)[0])
    pred = max(0, min(100, pred))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Predicted Score', f'{pred:.1f}/100')
    m2.metric('Study Hours', f"{inputs['study_hours']:.1f} h/day")
    m3.metric('Attendance', f"{inputs['attendance']:.0f}%")
    m4.metric('Sleep', f"{inputs['sleep_hours']:.1f} h/night")

    st.divider()
    st.subheader('Recommendations')

    def pick_recs(key: str, n: int = 2):
        items = RECOMMENDATIONS.get(key, [])
        return items[:n]

    rec_cols = st.columns(3)
    with rec_cols[0]:
        st.markdown('• ' + '\n• '.join(pick_recs('study_hours')))
    with rec_cols[1]:
        st.markdown('• ' + '\n• '.join(pick_recs('sleep_hours')))
    with rec_cols[2]:
        st.markdown('• ' + '\n• '.join(pick_recs('leisure_hours')))



elif section == 'Predict':
    st.title('Performance Prediction')
    inputs = input_form()
    if st.button('Predict Performance'):
        X = vector_for_model(model, inputs)
        prediction = float(model.predict(X)[0])
        prediction = max(0, min(100, prediction))
        st.success(f'Predicted Student Performance Score: {prediction:.2f}')


elif section == 'Target Planner':
    st.title('Target Planner')
    st.write('Estimate how to reach your target score from your current habits.')

    inputs = input_form()
    X_current = vector_for_model(model, inputs)
    current = float(model.predict(X_current)[0])
    current = max(0, min(100, current))

    target = st.slider('Target Score', 0.0, 100.0, min(95.0, max(50.0, current + 5)), 1.0)

    st.info(f'Current predicted: {current:.1f}. Target: {target:.1f}. Gap: {max(0.0, target - current):.1f}')

    st.subheader('What-if adjustments')
    colA, colB, colC = st.columns(3)
    with colA:
        delta_study = st.slider('Increase Study Hours by', 0.0, 6.0, 1.0, 0.5)
    with colB:
        delta_att = st.slider('Improve Attendance by', 0.0, 20.0, 5.0, 1.0)
    with colC:
        delta_sleep = st.slider('Improve Sleep by', 0.0, 3.0, 1.0, 0.5)

    adj_inputs = inputs.copy()
    adj_inputs['study_hours'] = min(12.0, inputs['study_hours'] + delta_study)
    adj_inputs['attendance'] = min(100.0, inputs['attendance'] + delta_att)
    adj_inputs['sleep_hours'] = min(12.0, inputs['sleep_hours'] + delta_sleep)
    adj_inputs['leisure_hours'] = max(0.0, inputs['leisure_hours'] - 0.5)

    X_adj = vector_for_model(model, adj_inputs)
    adj_pred = float(model.predict(X_adj)[0])
    adj_pred = max(0, min(100, adj_pred))

    c1, c2 = st.columns(2)
    with c1:
        st.metric('New Predicted Score', f'{adj_pred:.1f}', f'{adj_pred - current:+.1f}')
    with c2:
        st.metric('Remaining Gap', f'{max(0.0, target - adj_pred):.1f}')

    st.subheader('Actionable Plan')
    plan = []
    if target > current:
        if delta_study < 2:
            plan.append('Add at least 2 hours of focused study across the week (e.g., 4x 30-min blocks).')
        if inputs['attendance'] < 90 and delta_att < 10:
            plan.append('Raise attendance to 90%+ by scheduling and accountability partner.')
        if inputs['sleep_hours'] < 7 and delta_sleep < 1.5:
            plan.append('Move bedtime earlier by 30 minutes for the next 5 days.')
        if inputs['leisure_hours'] > 3:
            plan.append('Cap leisure to <= 2.5h/day using app timers; move gaming to weekends.')
        if inputs['part_time_job'] == 'Yes' and inputs['study_hours'] < 3:
            plan.append('Negotiate 1 fewer shift or create protected study blocks on off days.')
    else:
        plan.append('You already meet or exceed your target. Maintain habits for 2 weeks and reassess.')

    if not plan:
        plan.append('Tune one habit at a time and evaluate next week to avoid overload.')

    st.markdown('\n'.join([f'- {p}' for p in plan]))


elif section == 'Habit Tracker':
    st.title('Habit Tracker')
    st.write('Log daily habits to visualize trends that impact performance. Data stays in-session.')

    if 'habit_log' not in st.session_state:
        st.session_state.habit_log = []

    with st.form('habit_form'):
        date = st.date_input('Date')
        study = st.number_input('Study Hours', 0.0, 16.0, 2.0, 0.5)
        sleep = st.number_input('Sleep Hours', 0.0, 16.0, 7.0, 0.5)
        attendance = st.number_input('Attendance %', 0.0, 100.0, 90.0, 1.0)
        leisure = st.number_input('Leisure Hours', 0.0, 12.0, 2.0, 0.5)
        mh = st.slider('Mental Health (1-10)', 1, 10, 6)
        submitted = st.form_submit_button('Add Entry')
        if submitted:
            st.session_state.habit_log.append(
                {
                    'date': pd.to_datetime(date),
                    'study_hours': study,
                    'sleep_hours': sleep,
                    'attendance': attendance,
                    'leisure_hours': leisure,
                    'mental_health': mh,
                }
            )

    if st.session_state.habit_log:
        df = pd.DataFrame(st.session_state.habit_log).sort_values('date')
        st.dataframe(df, use_container_width=True)
        st.line_chart(df.set_index('date')[['study_hours', 'sleep_hours', 'leisure_hours']])
        st.bar_chart(df.set_index('date')[['attendance']])

        st.caption('Simple correlation (higher may indicate stronger relationship):')
        corr = df[['study_hours', 'sleep_hours', 'leisure_hours', 'attendance', 'mental_health']].corr()
        st.dataframe(corr)
    else:
        st.info("No entries yet. Add today's habits above.")


else:
    st.title('About This App')
    st.write(
        'This app provides predictions using your trained model (or a safe heuristic fallback), '
        'and translates them into actionable planning. Configure leisure platforms beyond Netflix, '
        'simulate what-if improvements, and track habits locally. '
        'No external services required.'
    )

    st.subheader('Data and Model')
    st.write('- Model file: final_student_performance_model.pkl (if missing, a heuristic is used)')
    st.write('- Features used: study hours, attendance, mental health, sleep, part-time job, leisure, platform distraction index, caffeine drinks/day')

    st.subheader('Privacy')
    st.write('Inputs are processed locally in your browser session. No network calls are made by this app.')
