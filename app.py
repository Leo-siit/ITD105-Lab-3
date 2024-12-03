import streamlit as st
import extra_streamlit_components as stx
import pandas as pd
from cv_chooser import cv_chooser
import regressors as rgr
import classifiers as clf

# Initialize session state variables if they are not already set
if 'DF' not in st.session_state:
    st.session_state.DF = None
if 'result_col' not in st.session_state:
    st.session_state.result_col = None
if 'model_type' not in st.session_state:  # Indicates whether we are using a regressor or classifier
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
            # Load the uploaded CSV file into the session state
            st.session_state.DF = pd.read_csv(uploaded_file)
        if st.session_state.DF is not None:
            # Allow the user to select the target column (Y column)
            columns = list(st.session_state.DF.columns)
            st.session_state.result_col = st.selectbox(
                "Select Y Column", 
                columns,
                index=columns.index(st.session_state.result_col) if st.session_state.result_col in columns else 0
            )
            st.write(st.session_state.DF)
            # Display note only if conditions are met
            if st.session_state.result_col:
                st.info("You can now proceed to the next step by clicking on 'Select Sampling Method' above.")

    # Step 1: Select Sampling Method
    elif st.session_state.step == 1:
        if st.session_state.DF is not None:
            # Use the cv_chooser component to select cross-validation parameters
            cv_chooser()
            # Add a button to confirm the selection
            if st.button("Confirm Sampling Method"):
                st.session_state.cv_confirmed = True
            # Display note only if cross-validation is selected and confirmed
            if st.session_state.cv_confirmed and st.session_state.cv_type:
                st.info("You can now proceed to the next step by clicking on 'Train Models' above.")
        else:
            st.warning("Please upload data first")

    # Step 2: Train Models
    elif st.session_state.step == 2:
        if st.session_state.DF is not None and st.session_state.cv is not None:
            with st.sidebar:
                # Allow the user to choose between regression and classification models
                model_type = st.radio("Select model type:", ["Regression", "Classification"])
                models = []
                pkg = None
                # Set the appropriate package and model list based on the selected model type
                if model_type == "Regression":
                    pkg = rgr
                    models = rgr.MODEL_MAPPING.keys()
                else:
                    pkg = clf
                    models = clf.MODEL_MAPPING.keys()
                # Display model parameter options for each model in an expander
                for model_name in models:
                    expander = st.expander(model_name)
                    with expander:
                        pkg.MODEL_PARAMS[model_name]()

                # Train models and evaluate their performance
                X = st.session_state.DF.drop(columns=[st.session_state.result_col])
                y = st.session_state.DF[st.session_state.result_col]
                results = {}
                for model_name in models:
                    # Create and train the model, then evaluate using cross-validation
                    model = pkg.create_model(model_name)
                    accuracy = pkg.score_model(model, X, y, st.session_state.cv)
                    results[model_name] = accuracy
                
                # Display the results as a table and a bar chart
                results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
                st.write(results_df)
                st.bar_chart(results_df.set_index('Model'))
        else:
            st.warning("Please upload data and select sampling method first")

if __name__ == "__main__":
    main()
