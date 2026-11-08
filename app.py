
import streamlit as st
import pandas as pd
import pickle
import html
from pathlib import Path


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="MindWell | Mental Wellness",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. CUSTOM CSS (YOUR ORIGINAL CSS)
# ==========================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F4F7FB;
}

/* Main content text */
[data-testid="stMain"] {
    color: #1F2937 !important;
}

[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4,
[data-testid="stMain"] h5,
[data-testid="stMain"] h6 {
    color: #172B4D !important;
}

[data-testid="stMain"] p,
[data-testid="stMain"] span,
[data-testid="stMain"] label,
[data-testid="stMain"] li {
    color: #1F2937;
}

/* Main title */
.main-title {
    font-size: 38px;
    font-weight: 700;
    color: #172B4D;
    margin-bottom: 10px;
}

.subtitle {
    font-size: 18px;
    color: #52647A;
    margin-bottom: 25px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #172B4D;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: #64748B;
}

/* Buttons */
.stButton button,
.stFormSubmitButton button {
    background-color: #172B4D;
    color: #FFFFFF !important;
    border-radius: 10px;
    border: none;
    font-weight: 600;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background-color: #294B7A;
    color: #FFFFFF !important;
}

/* Input fields */
[data-testid="stMain"] input,
[data-testid="stMain"] textarea {
    color: #1F2937 !important;
}

/* Select boxes and dropdowns */
[data-testid="stMain"] [data-baseweb="select"] {
    color: #FFFFFF !important;
}

/* Alert messages */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] {
    color: #1F2937 !important;
}

/* Alert icons */
[data-testid="stAlert"] svg {
    color: #1F2937 !important;
}

/* Result cards */
.result-card {
    background-color: #FFFFFF;
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #DCE3ED;
    box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    text-align: center;
    margin-bottom: 15px;
}

.result-title {
    font-size: 18px;
    font-weight: 600;
    color: #52647A;
    margin-bottom: 15px;
}

.result-value {
    font-size: 27px;
    font-weight: 700;
    color: #172B4D;
}

/* Feature cards */
.feature-card {
    background-color: #FFFFFF;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #DCE3ED;
}

.feature-card h3,
.feature-card p {
    color: #172B4D !important;
}

/* =====================================
   FIX TEXT VISIBILITY INSIDE INPUT BOXES
===================================== */

/* Number input text */
[data-testid="stNumberInput"] input {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Dropdown selected text */
[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    color: #FFFFFF !important;
}

/* Dropdown text */
[data-testid="stSelectbox"] div[data-baseweb="select"] span {
    color: #FFFFFF !important;
}

/* Dropdown selected value */
[data-testid="stSelectbox"] input {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Number input plus and minus buttons */
[data-testid="stNumberInput"] button {
    color: #FFFFFF !important;
}

/* Slider value labels */
[data-testid="stSlider"] {
    color: #1F2937 !important;
}

/* Text inside text input fields */
[data-testid="stTextInput"] input {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Placeholder text */
[data-testid="stMain"] input::placeholder {
    color: #CBD5E1 !important;
    -webkit-text-fill-color: #CBD5E1 !important;
}

/* Dropdown options when opened */
[data-baseweb="popover"] li,
[data-baseweb="menu"] li {
    color: #1F2937 !important;
    background-color: #FFFFFF !important;
}

/* Dropdown option hover */
[data-baseweb="menu"] li:hover {
    background-color: #E2E8F0 !important;
}

/* Predict Button Text Color */
.stFormSubmitButton button {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    font-weight: 600 !important;
}
//* Change alert text only */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] strong,
[data-testid="stAlert"] li {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

/* Preserve original alert background */
[data-testid="stAlert"] {
    background-color: #FFF4E5 !important;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model():

    model_path = Path(__file__).resolve().parent / "mhpred_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Please upload mhpred_model.pkl to your project folder."
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


try:
    log_model = load_model()

except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()


# ==========================================
# 4. SIDEBAR / PROFILE
# ==========================================

with st.sidebar:

    st.markdown(
        "<h2 style='text-align:center;'>🧠 MindWell</h2>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    profile_path = Path(__file__).resolve().parent / "profile.png"

    if profile_path.exists():

        st.image(
            str(profile_path),
            width=150
        )

    else:
        st.info("Add your profile.png to the project folder.")

    st.markdown("Sein Win @ Jannat")

    st.write("Senior Data Science Student")
    st.write("Let Yar Chaung, Maungdaw, Myanmar")

    st.markdown("---")

    st.markdown("### About This Project")

    st.write("""
    MindWell is a machine learning-based
    mental wellness prediction application.

    It explores how screen time, sleep,
    physical activity and lifestyle habits
    relate to mental wellness.
    """)

    st.markdown("---")

    st.caption("Developed as a Data Science Project")


# ==========================================
# 5. MAIN NAVIGATION FUNCTION
# ==========================================

def show_navigation():

    st.markdown("### Explore MindWell")

    st.radio(
        "Choose a page:",
        options=[
            "Home",
            "Mental Wellness Prediction",
            "About"
        ],
        format_func=lambda x: {
            "Home": "🏠 Home",
            "Mental Wellness Prediction": "🧠 Prediction",
            "About": "ℹ️ About"
        }[x],
        horizontal=True,
        key="page_navigation",
        label_visibility="collapsed"
    )

    st.markdown("---")


# Set default page

if "page_navigation" not in st.session_state:
    st.session_state.page_navigation = "Home"

page = st.session_state.page_navigation


# ==========================================
# 6. HOME PAGE
# ==========================================

if page == "Home":

    st.markdown(
        '<div class="main-title">Welcome to MindWell 🧠</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Understand your lifestyle. Explore your mental wellness.'
        '</div>',
        unsafe_allow_html=True
    )

    # Navigation directly under homepage heading

    show_navigation()

    st.info("""
    Your daily habits can influence your mental
    wellness. This application uses a machine
    learning model to estimate a mental wellness
    category based on lifestyle information.
    """)

    st.markdown("## What Can You Explore?")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            """
            <div class="feature-card">
                <h3>📱 Screen Time</h3>
                <p>Explore your daily digital habits.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="feature-card">
                <h3>😴 Sleep Habits</h3>
                <p>Understand the role of sleep.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:

        st.markdown(
            """
            <div class="feature-card">
                <h3>🏃 Lifestyle</h3>
                <p>Explore exercise and social activities.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    st.markdown("### Get Started")

    st.write(
        "Select Mental Wellness Prediction from the "
        "navigation menu above to enter your information."
    )


# ==========================================
# 7. MENTAL WELLNESS PREDICTION
# ==========================================

elif page == "Mental Wellness Prediction":

    st.markdown(
        '<div class="main-title">Mental Wellness Assessment</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Enter your information below to receive a '
        'model-generated prediction.'
        '</div>',
        unsafe_allow_html=True
    )

    # Navigation menu

    show_navigation()

    with st.form("prediction_form"):

        # ==================================
        # PERSONAL INFORMATION
        # ==================================

        st.markdown("### 👤 Personal Information")

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=14,
                max_value=100,
                value=22
            )

            gender = st.selectbox(
                "Gender",
                ["Male", "Female", "Other"]
            )

        with col2:

            occupation = st.selectbox(
                "Occupation",
                [
                    "Student",
                    "Employed",
                    "Self-employed",
                    "Unemployed"
                ]
            )

            work_mode = st.selectbox(
                "Work Mode",
                [
                    "Remote",
                    "Hybrid",
                    "On-site"
                ]
            )

        st.markdown("---")

        # ==================================
        # SCREEN TIME
        # ==================================

        st.markdown("### 📱 Screen Time Habits")

        col1, col2, col3 = st.columns(3)

        with col1:

            screen_time_hours = st.number_input(
                "Total Screen Time (hours/day)",
                min_value=0.0,
                max_value=24.0,
                value=6.0,
                step=0.5
            )

        with col2:

            work_screen_hours = st.number_input(
                "Work Screen Time (hours/day)",
                min_value=0.0,
                max_value=24.0,
                value=3.0,
                step=0.5
            )

        with col3:

            leisure_screen_hours = st.number_input(
                "Leisure Screen Time (hours/day)",
                min_value=0.0,
                max_value=24.0,
                value=3.0,
                step=0.5
            )

        st.markdown("---")

        # ==================================
        # SLEEP AND HEALTH
        # ==================================

        st.markdown("### 😴 Sleep and Health Habits")

        col1, col2 = st.columns(2)

        with col1:

            sleep_hours = st.number_input(
                "Sleep Hours per Day",
                min_value=0.0,
                max_value=24.0,
                value=7.0,
                step=0.5
            )

            sleep_quality_1_5 = st.slider(
                "Sleep Quality (1 = Poor, 5 = Excellent)",
                min_value=1,
                max_value=5,
                value=3
            )

            stress_level_0_10 = st.slider(
                "Stress Level (0 = Lowest, 10 = Highest)",
                min_value=0,
                max_value=10,
                value=5
            )

        with col2:

            exercise_minutes_per_week = st.number_input(
                "Exercise (minutes/week)",
                min_value=0,
                max_value=1000,
                value=150
            )

            social_hours_per_week = st.number_input(
                "Social Activities (hours/week)",
                min_value=0.0,
                max_value=168.0,
                value=10.0,
                step=0.5
            )

            productivity_0_100 = st.slider(
                "Productivity (0-100)",
                min_value=0,
                max_value=100,
                value=50
            )

        st.write("")

        submitted = st.form_submit_button(
            "Predict Mental Wellness",
            use_container_width=True
        )


    # ======================================
    # 8. PREDICTION RESULT
    # ======================================

    if submitted:

        # ==================================
        # INPUT VALIDATION
        # ==================================

        if work_screen_hours + leisure_screen_hours > screen_time_hours:

            st.error("""
            Invalid screen time information!

            Your work screen time plus leisure screen time
            cannot exceed your total screen time.

            Please correct your values and try again.
            """)

        else:

            # ==================================
            # CREATE INPUT DATAFRAME
            # ==================================

            input_data = pd.DataFrame({

                "age": [age],

                "gender": [gender],

                "occupation": [occupation],

                "work_mode": [work_mode],

                "screen_time_hours": [screen_time_hours],

                "work_screen_hours": [work_screen_hours],

                "leisure_screen_hours": [leisure_screen_hours],

                "sleep_hours": [sleep_hours],

                "sleep_quality_1_5": [sleep_quality_1_5],

                "stress_level_0_10": [stress_level_0_10],

                "exercise_minutes_per_week": [
                    exercise_minutes_per_week
                ],

                "social_hours_per_week": [
                    social_hours_per_week
                ],

                "productivity_0_100": [
                    productivity_0_100
                ]

            })

            try:

                # Check model feature names

                if hasattr(log_model, "feature_names_in_"):

                    expected_columns = list(
                        log_model.feature_names_in_
                    )

                    missing_columns = [
                        col for col in expected_columns
                        if col not in input_data.columns
                    ]

                    if missing_columns:

                        raise ValueError(
                            f"Missing model columns: {missing_columns}. "
                            "Check your training and preprocessing code."
                        )

                    # Match model column order

                    input_data = input_data[expected_columns]

                # Make prediction

                prediction = log_model.predict(input_data)[0]

                st.markdown("---")

                st.markdown("## 📊 Your Prediction Results")

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        f"""
                        <div class="result-card">
                            <div class="result-title">
                                Predicted Mental Wellness
                            </div>
                            <div class="result-value">
                                {html.escape(str(prediction))}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with col2:

                    st.markdown(
                        """
                        <div class="result-card">
                            <div class="result-title">
                                Assessment Status
                            </div>
                            <div class="result-value">
                                Completed ✓
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # ==================================
                # PREDICTION PROBABILITIES
                # ==================================

                if hasattr(log_model, "predict_proba"):

                    probabilities = log_model.predict_proba(
                        input_data
                    )[0]

                    classes = log_model.classes_

                    st.markdown("### Prediction Probabilities")

                    probability_df = pd.DataFrame({

                        "Mental Wellness Category": classes,

                        "Probability (%)": probabilities * 100

                    })

                    st.bar_chart(
                        probability_df.set_index(
                            "Mental Wellness Category"
                        )
                    )

                    st.dataframe(
                        probability_df.style.format({
                            "Probability (%)": "{:.2f}%"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

                # Disclaimer

                st.info("""
                This result is an estimate generated by a
                machine learning model. It is not a clinical
                diagnosis or a substitute for professional
                mental health assessment.
                """)

            except Exception as e:

                st.error(f"Prediction error: {e}")

                st.warning(
                    "Please check whether your input categories "
                    "and column names match the training dataset."
                )


# ==========================================
# 9. ABOUT PAGE
# ==========================================

elif page == "About":

    st.markdown(
        '<div class="main-title">About MindWell</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Exploring the relationship between lifestyle '
        'and mental wellness through machine learning.'
        '</div>',
        unsafe_allow_html=True
    )

    # Navigation menu

    show_navigation()

    st.write("""
    MindWell is a machine learning project designed
    to explore the relationship between digital
    screen time, lifestyle habits and mental wellness.

    The application uses a trained Logistic Regression
    model to predict mental wellness categories based
    on user-provided information.
    """)

    st.markdown("---")

    st.markdown("### 🎯 Project Objectives")

    st.write("""
    - Explore the relationship between screen time and mental wellness.
    - Understand how sleep and exercise relate to wellness.
    - Explore the role of stress and social activities.
    - Apply machine learning to lifestyle-related data.
    - Present model predictions through an interactive web application.
    """)

    st.markdown("---")

    st.markdown("### Technology Used")

    st.write("""
    - Python
    - Pandas
    - Scikit-learn
    - Logistic Regression
    - Streamlit
    - Pickle
    """)

    st.markdown("---")

    st.markdown("### Important Disclaimer")

    st.warning("""
    This application is intended for educational
    and informational purposes only.

    It does not diagnose, treat or prevent mental
    health conditions.
    """)

    st.markdown("---")

    st.caption("MindWell | Machine Learning Data Science Project")
