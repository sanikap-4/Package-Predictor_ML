import streamlit as st
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CGPA → Package Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)


MODEL_PATH = "model.joblib"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #f8fafc 0%, #eef2ff 100%);
}

.hero {
    padding: 2rem;
    border-radius: 20px;
    background: linear-gradient(135deg, #111827, #312e81);
    color: white;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.12);
}

.hero h1 {
    font-size: 2.6rem;
    margin-bottom: 0.3rem;
}

.hero p {
    font-size: 1.05rem;
    opacity: 0.88;
}

.prediction-card {
    padding: 1.8rem;
    border-radius: 18px;
    background: white;
    border: 1px solid #e5e7eb;
    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
    text-align: center;
}

.package-value {
    font-size: 2.8rem;
    font-weight: 800;
    color: #312e81;
}

.small-label {
    color: #6b7280;
    font-size: 0.9rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    return joblib.load(MODEL_PATH)


try:

    model = load_model()

    model_loaded = True

except Exception as e:

    model = None
    model_loaded = False
    load_error = str(e)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    if model_loaded:

        st.success("Model loaded successfully")

        st.write(
            f"**Model:** `{type(model).__name__}`"
        )

        st.write("**Input:** CGPA")

        st.write("**Output:** Package")

    else:

        st.error("Model could not be loaded.")

        st.code(load_error)


    st.divider()

    st.subheader("📌 Prediction Flow")

    st.write("1. Enter CGPA")

    st.write("2. Model processes input")

    st.write("3. Package is predicted")

    st.write("4. Analyze different CGPAs")


    st.divider()

    st.caption(
        "Machine Learning Prediction System"
    )


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<div class="hero">

<h1>🎓 CGPA → Package Predictor</h1>

<p>
An interactive Machine Learning application that predicts
an expected placement package based on a student's CGPA.
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# INPUT SECTION
# ============================================================

left, right = st.columns(
    [1, 1.4],
    gap="large"
)


with left:

    st.subheader("🎯 Student Input")

    cgpa = st.number_input(

        "Enter your CGPA",

        min_value=0.0,

        max_value=10.0,

        value=7.5,

        step=0.1,

        format="%.1f"

    )


    # CGPA progress

    st.progress(cgpa / 10)


    # Academic level

    if cgpa < 5:

        level = "Needs Improvement"

    elif cgpa < 7:

        level = "Average"

    elif cgpa < 8.5:

        level = "Good"

    elif cgpa < 9.5:

        level = "Excellent"

    else:

        level = "Outstanding"


    st.metric(
        "Academic Level",
        level
    )


    predict_clicked = st.button(

        "🚀 Predict Package",

        type="primary",

        use_container_width=True

    )


# ============================================================
# PREDICTION SECTION
# ============================================================

with right:

    st.subheader("📊 Prediction Result")


    if not model_loaded:

        st.warning(
            "Please keep model.joblib in the same folder as this app."
        )


    elif predict_clicked:

        try:

            # Convert input into 2D NumPy array

            input_data = np.array(
                [[cgpa]],
                dtype=float
            )


            # Make prediction

            prediction_raw = model.predict(input_data)

            # Safely extract a scalar from the model output
            prediction_array = np.asarray(prediction_raw)

            if prediction_array.size == 0:
                raise ValueError("The model returned an empty prediction.")

            prediction = float(prediction_array.ravel()[0])


            # Display prediction

            st.markdown(
                f"""
                <div class="prediction-card">

                    <div class="small-label">
                        Predicted Package
                    </div>

                    <div class="package-value">
                        ₹ {prediction:.2f} LPA
                    </div>

                    <div class="small-label">
                        Based on CGPA: {cgpa:.1f}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            st.success(
                "Prediction generated successfully!"
            )


            # ====================================================
            # MODEL INTERPRETATION
            # ====================================================

            if hasattr(model, "coef_") and hasattr(
                model,
                "intercept_"
            ):

                coefficient_array = np.asarray(model.coef_)

                if coefficient_array.size == 0:
                    raise ValueError("Model coefficient is empty.")

                coefficient = float(
                    coefficient_array.ravel()[0]
                )


                intercept_array = np.asarray(model.intercept_)

                if intercept_array.size == 0:
                    raise ValueError("Model intercept is empty.")

                intercept = float(
                    intercept_array.ravel()[0]
                )


                c1, c2, c3 = st.columns(3)


                c1.metric(
                    "CGPA",
                    f"{cgpa:.1f}"
                )


                c2.metric(
                    "Predicted Package",
                    f"{prediction:.2f} LPA"
                )


                c3.metric(
                    "CGPA Effect",
                    f"{coefficient:.2f}"
                )


                st.markdown(
                    "### 🧠 Model Insight"
                )


                if coefficient > 0:

                    st.info(
                        f"""
                        According to this trained Linear Regression
                        model, higher CGPA is associated with a higher
                        predicted package.

                        The model estimates approximately
                        **{coefficient:.2f} LPA change**
                        for every 1-point increase in CGPA.
                        """
                    )


                elif coefficient < 0:

                    st.warning(
                        f"""
                        The trained model learned a negative relationship
                        between CGPA and package.

                        Coefficient:
                        **{coefficient:.2f}**
                        """
                    )


                else:

                    st.info(
                        "The model learned almost no linear relationship "
                        "between CGPA and package."
                    )


                # Regression equation

                st.markdown(
                    "### 📐 Regression Equation"
                )


                st.code(

                    f"Package = "
                    f"({coefficient:.4f} × CGPA) "
                    f"+ ({intercept:.4f})",

                    language="text"

                )


        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# ============================================================
# WHAT-IF ANALYSIS
# ============================================================

if model_loaded:

    st.divider()

    st.subheader(
        "🔬 What-If Analysis"
    )

    st.caption(
        "See how predicted package changes for different CGPAs."
    )


    # Generate CGPA values

    cgpa_values = np.arange(
        5.0,
        10.1,
        0.5
    )


    # Convert to model input

    inputs = cgpa_values.reshape(
        -1,
        1
    )


    try:

        # Predict packages

        package_predictions = np.asarray(
            model.predict(inputs)
        ).reshape(-1)


        # Create DataFrame

        comparison = pd.DataFrame({

            "CGPA": cgpa_values,

            "Predicted Package (LPA)":
                np.round(
                    package_predictions,
                    2
                )

        })


        # Display table

        st.dataframe(

        comparison,

        use_container_width=True,
        hide_index=True

        )


        # Prepare chart

        chart_data = comparison.set_index(
            "CGPA"
        )


        # Display chart

        st.line_chart(
            chart_data
        )


    except Exception as e:

        st.warning(
            f"What-if analysis is unavailable: {e}"
        )


# ============================================================
# TECHNICAL DETAILS
# ============================================================

with st.expander(
    "🔍 Technical Details"
):

    st.write(
        "### Machine Learning Pipeline"
    )


    st.write(
        """
        **Input → CGPA → Trained ML Model → Predicted Package**

        The application accepts CGPA as a numerical feature,
        converts it into a 2-dimensional NumPy array and sends
        it to the trained model using `model.predict()`.
        """
    )


    if model_loaded and hasattr(
        model,
        "coef_"
    ):

        st.write(
            "### Learned Parameters"
        )


        coefficient_array = np.asarray(model.coef_)

        if coefficient_array.size == 0:
            raise ValueError("Model coefficient is empty.")

        coefficient = float(
            coefficient_array.ravel()[0]
        )


        intercept_array = np.asarray(model.intercept_)

        if intercept_array.size == 0:
            raise ValueError("Model intercept is empty.")

        intercept = float(
            intercept_array.ravel()[0]
        )


        parameters = pd.DataFrame({

            "Parameter": [
                "Coefficient",
                "Intercept"
            ],

            "Value": [
                coefficient,
                intercept
            ]

        })


        st.dataframe(

            parameters,

            use_container_width=True,

            hide_index=True

        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🎓 CGPA → Package Predictor | "
    "Built with Python, Streamlit, NumPy, Pandas & Machine Learning"
)