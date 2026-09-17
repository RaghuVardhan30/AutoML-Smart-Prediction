import streamlit as st
import requests
import os

# =========================================================
# API CONFIGURATION
# =========================================================


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AutoML Smart Student Prediction",
    page_icon="🎓",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 40px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        color: #AAAAAA;
        margin-bottom: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="main-title">🎓 AutoML Smart Student Prediction System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    "Predict a student's final academic performance using an AutoML-trained model"
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# =========================================================
# FORM
# =========================================================

with st.form("student_prediction_form"):

    # =====================================================
    # BASIC INFORMATION
    # =====================================================

    st.header("👤 Basic Student Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        school_display = st.selectbox(
            "🏫 School",
            [
                "Gabriel Pereira (GP)",
                "Mousinho da Silveira (MS)"
            ]
        )

        school = {
            "Gabriel Pereira (GP)": "GP",
            "Mousinho da Silveira (MS)": "MS"
        }[school_display]

    with col2:

        sex_display = st.selectbox(
            "Gender",
            [
                "Female",
                "Male"
            ]
        )

        sex = {
            "Female": "F",
            "Male": "M"
        }[sex_display]

    with col3:

        age = st.number_input(
            "Age",
            min_value=15,
            max_value=22,
            value=17,
            step=1
        )

    col1, col2, col3 = st.columns(3)

    with col1:

        address_display = st.selectbox(
            "📍 Area of Residence",
            [
                "Urban",
                "Rural"
            ]
        )

        address = {
            "Urban": "U",
            "Rural": "R"
        }[address_display]

    with col2:

        family_size_display = st.selectbox(
            "👨‍👩‍👧 Family Size",
            [
                "More than 3 members",
                "3 or fewer members"
            ]
        )

        famsize = {
            "More than 3 members": "GT3",
            "3 or fewer members": "LE3"
        }[family_size_display]

    with col3:

        parent_status_display = st.selectbox(
            "👪 Parents' Living Situation",
            [
                "Living together",
                "Separated / Apart"
            ]
        )

        Pstatus = {
            "Living together": "T",
            "Separated / Apart": "A"
        }[parent_status_display]


    # =====================================================
    # EDUCATION & FAMILY
    # =====================================================

    st.divider()

    st.header("🎓 Education & Family Information")

    education_options = {
        "No formal education": 0,
        "Primary education": 1,
        "5th–9th grade": 2,
        "Secondary education": 3,
        "Higher education": 4
    }

    col1, col2, col3 = st.columns(3)

    with col1:

        mother_education_display = st.selectbox(
            "Mother's Education",
            list(education_options.keys())
        )

        Medu = education_options[mother_education_display]

    with col2:

        father_education_display = st.selectbox(
            "Father's Education",
            list(education_options.keys())
        )

        Fedu = education_options[father_education_display]

    with col3:

        guardian_display = st.selectbox(
            "Main Guardian",
            [
                "Mother",
                "Father",
                "Other"
            ]
        )

        guardian = {
            "Mother": "mother",
            "Father": "father",
            "Other": "other"
        }[guardian_display]


    # =====================================================
    # PARENT OCCUPATION
    # =====================================================

    col1, col2 = st.columns(2)

    job_options = {
        "Teacher": "teacher",
        "Healthcare professional": "health",
        "Services sector": "services",
        "Homemaker": "at_home",
        "Other": "other"
    }

    with col1:

        mother_job_display = st.selectbox(
            "Mother's Occupation",
            list(job_options.keys())
        )

        Mjob = job_options[mother_job_display]

    with col2:

        father_job_display = st.selectbox(
            "Father's Occupation",
            list(job_options.keys())
        )

        Fjob = job_options[father_job_display]


    # =====================================================
    # ACADEMIC INFORMATION
    # =====================================================

    st.divider()

    st.header("📚 Academic Information")

    reason_options = {
        "Course preference": "course",
        "Close to home": "home",
        "School reputation": "reputation",
        "Other reason": "other"
    }

    col1, col2, col3 = st.columns(3)

    with col1:

        reason_display = st.selectbox(
            "Reason for Choosing School",
            list(reason_options.keys())
        )

        reason = reason_options[reason_display]

    with col2:

        travel_options = {
            "Less than 15 minutes": 1,
            "15–30 minutes": 2,
            "30–60 minutes": 3,
            "More than 60 minutes": 4
        }

        travel_display = st.selectbox(
            "🚗 Travel Time to School",
            list(travel_options.keys())
        )

        traveltime = travel_options[travel_display]

    with col3:

        study_options = {
            "Less than 2 hours per week": 1,
            "2–5 hours per week": 2,
            "5–10 hours per week": 3,
            "More than 10 hours per week": 4
        }

        study_display = st.selectbox(
            "📖 Weekly Study Time",
            list(study_options.keys())
        )

        studytime = study_options[study_display]


    # =====================================================
    # ACADEMIC HISTORY
    # =====================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        failures_display = st.selectbox(
            "📉 Previous Class Failures",
            [
                "No failures",
                "1 failure",
                "2 failures",
                "3 or more failures"
            ]
        )

        failures = {
            "No failures": 0,
            "1 failure": 1,
            "2 failures": 2,
            "3 or more failures": 3
        }[failures_display]

    with col2:

        higher_display = st.selectbox(
            "🎓 Interested in Higher Education?",
            [
                "Yes",
                "No"
            ]
        )

        higher = {
            "Yes": "yes",
            "No": "no"
        }[higher_display]

    with col3:

        internet_display = st.selectbox(
            "🌐 Internet Access at Home?",
            [
                "Yes",
                "No"
            ]
        )

        internet = {
            "Yes": "yes",
            "No": "no"
        }[internet_display]


    # =====================================================
    # SUPPORT & ACTIVITIES
    # =====================================================

    st.divider()

    st.header("🤝 Support & Activities")

    col1, col2, col3, col4 = st.columns(4)

    yes_no = {
        "Yes": "yes",
        "No": "no"
    }

    with col1:

        schoolsup_display = st.selectbox(
            "🏫 Extra School Support",
            ["Yes", "No"]
        )

        schoolsup = yes_no[schoolsup_display]

    with col2:

        famsup_display = st.selectbox(
            "👨‍👩‍👧 Family Educational Support",
            ["Yes", "No"]
        )

        famsup = yes_no[famsup_display]

    with col3:

        paid_display = st.selectbox(
            "📘 Paid Extra Classes",
            ["Yes", "No"]
        )

        paid = yes_no[paid_display]

    with col4:

        activities_display = st.selectbox(
            "⚽ Extra-Curricular Activities",
            ["Yes", "No"]
        )

        activities = yes_no[activities_display]


    col1, col2 = st.columns(2)

    with col1:

        nursery_display = st.selectbox(
            "🧸 Attended Nursery School?",
            ["Yes", "No"]
        )

        nursery = yes_no[nursery_display]

    with col2:

        # Relationship removed from the UI.
        # The existing trained model still expects famrel,
        # so a neutral value is supplied internally.
        famrel = 3


    # =====================================================
    # LIFESTYLE INFORMATION
    # =====================================================

    st.divider()

    st.header("🌱 Lifestyle Information")

    quality_options = {
        "Very low": 1,
        "Low": 2,
        "Average": 3,
        "High": 4,
        "Very high": 5
    }

    col1, col2 = st.columns(2)

    with col1:

        freetime_display = st.select_slider(
            "⏰ Free Time After School",
            options=list(quality_options.keys()),
            value="Average"
        )

        freetime = quality_options[freetime_display]

    with col2:

        goout_display = st.select_slider(
            "👥 Frequency of Going Out",
            options=list(quality_options.keys()),
            value="Average"
        )

        goout = quality_options[goout_display]


    col1, col2 = st.columns(2)

    with col1:

        health_display = st.select_slider(
            "❤️ Current Health Condition",
            options=[
                "Very poor",
                "Poor",
                "Average",
                "Good",
                "Excellent"
            ],
            value="Good"
        )

        health = {
            "Very poor": 1,
            "Poor": 2,
            "Average": 3,
            "Good": 4,
            "Excellent": 5
        }[health_display]

    with col2:

        absences = st.number_input(
            "📅 Number of School Absences",
            min_value=0,
            max_value=100,
            value=4,
            step=1
        )


    # =====================================================
    # PREDICTION BUTTON
    # =====================================================

    st.divider()

    submitted = st.form_submit_button(
        "🔮 Predict Final Grade",
        use_container_width=True
    )


# =========================================================
# PREDICTION
# =========================================================

if submitted:

    student_data = {

        "school": school,
        "sex": sex,
        "age": age,
        "address": address,
        "famsize": famsize,
        "Pstatus": Pstatus,

        "Medu": Medu,
        "Fedu": Fedu,

        "Mjob": Mjob,
        "Fjob": Fjob,

        "reason": reason,
        "guardian": guardian,

        "traveltime": traveltime,
        "studytime": studytime,
        "failures": failures,

        "schoolsup": schoolsup,
        "famsup": famsup,
        "paid": paid,
        "activities": activities,
        "nursery": nursery,

        "higher": higher,
        "internet": internet,

        # Hidden model-required field.
        # Removed from the user interface.
        "famrel": famrel,

        "freetime": freetime,
        "goout": goout,

        "health": health,
        "absences": absences
    }

    try:

        with st.spinner("🤖 AutoML model is analyzing the student..."):

            response = requests.post(
                f"{API_URL}/predict",
                json=student_data,
                timeout=30
            )

        if response.status_code == 200:

            result = response.json()

            predicted_grade = result["predicted_final_grade"]
            category = result["performance_category"]

        else:

            st.error(
                f"Prediction failed ({response.status_code}): "
                f"{response.text}"
            )

            st.stop()


        # =================================================
        # RESULT
        # =================================================

        st.success("✅ Prediction completed successfully!")

        st.divider()

        st.header("📊 Prediction Result")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Predicted Final Grade",
                f"{predicted_grade:.2f} / 20"
            )

        with col2:

            st.metric(
                "Performance Category",
                category
            )

        st.progress(
            min(max(predicted_grade / 20, 0.0), 1.0)
        )


        # =================================================
        # INTERPRETATION
        # =================================================

        st.subheader("📌 Interpretation")

        if predicted_grade >= 15:

            st.success(
                "🌟 The predicted performance is excellent. "
                "The student is expected to perform very well."
            )

        elif predicted_grade >= 12:

            st.info(
                "👍 The predicted performance is good. "
                "The student is expected to achieve a satisfactory result."
            )

        elif predicted_grade >= 10:

            st.warning(
                "⚠️ The predicted performance is average. "
                "Additional academic support may help improve performance."
            )

        else:

            st.error(
                "⚠️ The predicted performance indicates that "
                "the student may need additional academic support."
            )


        st.caption(
            "Note: This prediction is an estimate generated by a "
            "machine learning model and should not be considered "
            "an official academic assessment."
        )


    except requests.exceptions.ConnectionError:

        st.error(
            "❌ Could not connect to the FastAPI backend."
        )

        st.info(
            "Please start FastAPI first using: "
            "uvicorn app.api:app --reload"
        )

    except requests.exceptions.Timeout:

        st.error(
            "❌ The prediction request timed out."
        )

    except Exception as e:

        st.error(
            f"❌ Prediction failed: {e}"
        )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "AutoML Smart Prediction System | Student Performance Prediction"
)