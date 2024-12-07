import streamlit as st
import extra_streamlit_components as stx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from cv_chooser import cv_chooser
import regressors as rgr
import classifiers as clf

# Initialize session state variables if they are not already set
if 'DF' not in st.session_state:
    st.session_state.DF = None
if 'result_col' not in st.session_state:
    st.session_state.result_col = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'cv_type' not in st.session_state:
    st.session_state.cv_type = None
if 'cv_args' not in st.session_state:
    st.session_state.cv_args = {}
if 'cv' not in st.session_state:
    st.session_state.cv = None
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'cv_confirmed' not in st.session_state:
    st.session_state.cv_confirmed = False
if 'previous_scores' not in st.session_state:
    st.session_state.previous_scores = {}

def display_comparison_results(results):
    """
    Display model comparison results with changes and visualizations.
    """
    st.header("Model Comparison Results")

    # Calculate changes compared to previous scores
    changes = {}
    for model, score in results.items():
        # Extract the first element if score is a tuple
        score = score[0] if isinstance(score, tuple) else score
        
        previous_score = st.session_state.previous_scores.get(model, None)
        # Extract the first element if previous_score is a tuple
        previous_score = previous_score[0] if isinstance(previous_score, tuple) else previous_score
        
        changes[model] = score - previous_score if previous_score is not None else 0.0

    # Update the previous scores in session state
    st.session_state.previous_scores = results

    # Define formatting functions
    def format_arrow(val):
        return f"{'↑' if val > 0 else '↓'} {abs(val):.0f}%" if val != 0 else f"{val:.0f}%"

    def color_arrow(val):
        return "color: green" if val > 0 else "color: red" if val < 0 else "color: yellow"

    # Convert results and changes to a DataFrame
    results_df = pd.DataFrame(
        # Extract first element if result is a tuple
        [(model, (score[0] if isinstance(score, tuple) else score) * 100, 
          changes[model] * 100) for model, score in results.items()],
        columns=["Model", "Accuracy Score", "Change"]
    )

    # Sort results by score (descending)
    results_df = results_df.sort_values(by="Accuracy Score", ascending=False).reset_index(drop=True)

    # Display the DataFrame with custom formatting
    st.subheader("Results Table")
    
    # Apply styling to the DataFrame
    styled_df = results_df.style.format({
        "Accuracy Score": "{:.2f}%",
        "Change": format_arrow
    }).applymap(color_arrow, subset=["Change"])
    
    st.dataframe(styled_df, use_container_width=True)

    # Visualize results with a bar chart
    st.subheader("Results Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(results_df["Model"], results_df["Accuracy Score"], color="skyblue")
    ax.set_ylabel("Accuracy Score (%)")
    ax.set_xlabel("Model")
    ax.set_title("Model Performance Comparison")
    plt.xticks(rotation=45, ha="right")

    # Add annotations to bars
    for bar, score in zip(bars, results_df["Accuracy Score"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{score:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color="black",
        )

    st.pyplot(fig)

def main():
    # Create a stepper bar with three steps: Upload Data, Select Sampling Method, Train Models
    step = stx.stepper_bar(steps=["Upload Data", "Select Sampling Method", "Train Models"])
    
    # Update the current step based on user interaction
    if step != st.session_state.step:
        st.session_state.step = step

    # Step 0: Upload Data
    if st.session_state.step == 0:
        uploaded_file = st.file_uploader("Upload Data", type=['csv'])
        if uploaded_file is not None:
            st.session_state.DF = pd.read_csv(uploaded_file)
        if st.session_state.DF is not None:
            columns = list(st.session_state.DF.columns)
            st.session_state.result_col = st.selectbox(
                "Select Y Column", 
                columns,
                index=columns.index(st.session_state.result_col) if st.session_state.result_col in columns else 0
            )
            st.write(st.session_state.DF)
            if st.session_state.result_col:
                st.info("You can now proceed to the next step by clicking on 'Select Sampling Method' above.")

    # Step 1: Select Sampling Method
    elif st.session_state.step == 1:
        if st.session_state.DF is not None:
            cv_chooser()
            if st.button("Confirm Sampling Method"):
                st.session_state.cv_confirmed = True
            if st.session_state.cv_confirmed and st.session_state.cv_type:
                st.info("You can now proceed to the next step by clicking on 'Train Models' above.")
        else:
            st.warning("Please upload data first")

    # Step 2: Train Models
    elif st.session_state.step == 2:
        if st.session_state.DF is not None and st.session_state.cv is not None:
            with st.sidebar:
                model_type = st.radio("Select model type:", ["Regression", "Classification"])
                models = []
                pkg = None
                if model_type == "Regression":
                    pkg = rgr
                    models = rgr.MODEL_MAPPING.keys()
                else:
                    pkg = clf
                    models = clf.MODEL_MAPPING.keys()
                for model_name in models:
                    expander = st.expander(model_name)
                    with expander:
                        pkg.MODEL_PARAMS[model_name]()

            X = st.session_state.DF.drop(columns=[st.session_state.result_col])
            y = st.session_state.DF[st.session_state.result_col]
            results = {}
            for model_name in models:
                model = pkg.create_model(model_name)
                accuracy = pkg.score_model(model, X, y, st.session_state.cv)
                results[model_name] = accuracy

            display_comparison_results(results)
        else:
            st.warning("Please upload data and select sampling method first")

if __name__ == "__main__":
    main()
