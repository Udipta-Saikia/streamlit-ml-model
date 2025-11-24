import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ------------------------------
# Load trained model
# ------------------------------
pipeline = joblib.load("career_recommender_model.joblib")

# ------------------------------
# Prediction function
# ------------------------------
def predict_top_3_careers(age, education, skills_list, interests_list):
    combined_text = ";".join(skills_list) + ";" + ";".join(interests_list)

    user_df = pd.DataFrame([{
        "Age": age,
        "Education": education,
        "Combined_Text": combined_text
    }])

    proba = pipeline.predict_proba(user_df)[0]
    classes = pipeline.classes_
    top3_idx = np.argsort(proba)[-3:][::-1]

    return [(classes[i], float(proba[i])) for i in top3_idx]


# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Career Recommendation System", layout="centered")

st.title("🎯 AI-Powered Career Recommendation System")
st.write("Enter your details to get the **top 3 career suggestions** based on your skills & interests.")

# --- Inputs ---
age = st.number_input("Enter Age", min_value=15, max_value=60, value=22)

education = st.selectbox("Select Education Level", [
    "High School", "Bachelor's", "Master's", "PhD"
])

skills = st.multiselect(
    "Select Your Skills",
    [
        "Accounting",
"Algorithms",
"Analysis",
"Analytics",
"Auditing",
"AutoCAD",
"Behavior Analysis",
"Branding",
"Business Analysis",
"Business Strategy",
"C++",
"CAD",
"CRM",
"Circuits",
"Communication",
"Conflict Resolution",
"Construction",
"Coordination",
"Counseling",
"Creativity",
"Critical Thinking",
"Cybersecurity",
"Data Analysis",
"Data Science",
"Design",
"Documentation",
"Electronics",
"Excel",
"Finance",
"Firewalls",
"Illustrator",
"Java",
"Leadership",
"Learning",
"Linux",
"Logistics",
"Machine Learning",
"Management",
"Manufacturing",
"Marketing",
"MATLAB",
"Mechanical Design",
"Modeling",
"Negotiation",
"Networking",
"NoSQL",
"Numerical Methods",
"NumPy",
"Observation",
"Operations",
"Patience",
"Photoshop",
"Planning",
"Power BI",
"Power Systems",
"Predictive Modeling",
"Problem Solving",
"Prototyping",
"Public Speaking",
"Python",
"R",
"Recruitment",
"Reporting",
"Research",
"Risk Taking",
"Roadmapping",
"SEO",
"SQL",
"Statistics",
"Structural Analysis",
"Surveying",
"Teaching",
"Team Management",
"Teamwork",
"Thermodynamics",
"Threat Analysis",
"Tally",
"UI/UX",
"Visualization",
"Wireframing",
"Writing"

    ]
)

interests = st.multiselect(
    "Select Your Interests",
    [
        
"Art",
"Design",
"Entrepreneurship",
"Finance",
"Management",
"Marketing",
"Problem Solving",
"Public Speaking",
"Research",
"Science",
"Teaching",
"Technology",
"Travel"

    ]
)

# --- Predict Button ---
if st.button("🔍 Predict My Career"):
    if len(skills) == 0 or len(interests) == 0:
        st.warning("Please select at least one skill and one interest.")
    else:
        results = predict_top_3_careers(age, education, skills, interests)

        st.subheader("✨ Top 3 Career Recommendations")

        for rank, (career, prob) in enumerate(results, 1):
            st.write(f"**{rank}. {career}** — Probability: `{prob:.4f}`")


st.markdown("---")
st.caption("Created by Udipta • Powered by Machine Learning 🚀")
