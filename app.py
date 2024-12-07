import streamlit as st
import extra_streamlit_components as stx
import pandas as pd
import altair as alt
import os
import tempfile
from cv_chooser import cv_chooser
import regressors as rgr
import classifiers as clf
import joblib

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
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'current_results' not in st.session_state:
    st.session_state.current_results = None
if 'current_model_type' not in st.session_state:
    st.session_state.current_model_type = None

def download_model(model, model_name):
    """
    Create a downloadable model file using temporary file with proper cleanup
    """
    temp_file = None
    try:
        # Create temp file with a unique name
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.joblib')
        temp_path = temp_file.name
        temp_file.close()  # Close file handle immediately
        
        # Save model
        joblib.dump(model, temp_path)
        
        # Read file for download
        with open(temp_path, 'rb') as f:
            model_bytes = f.read()
            
        # Create download button
        st.download_button(
            label=f"Download {model_name}",
            data=model_bytes,
            file_name=f"{model_name.replace(' ', '_')}_model.joblib",
            mime="application/octet-stream"
        )
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
    finally:
        # Clean up temp file in finally block
        if temp_file:
            try:
                os.unlink(temp_path)
            except:
                pass  # Ignore cleanup errors

def display_comparison_results(results, model_type):
    """
    Display model comparison results with changes and visualizations using Altair.
    """
    metric_name = "Accuracy (%)" if model_type == "Classification" else "Mean Absolute Error (MAE)"
    st.header(f"Model Comparison Results ({metric_name})")

    # Calculate changes compared to previous scores
    changes = {}
    for model, score in results.items():
        score_value = score[0] if isinstance(score, tuple) else score
        previous_score = st.session_state.previous_scores.get(model, None)
        previous_value = previous_score[0] if isinstance(previous_score, tuple) else previous_score
        changes[model] = score_value - (previous_value if previous_value is not None else 0.0)

    # Update the previous scores in session state
    st.session_state.previous_scores = results

    # Convert results and changes to a DataFrame
    results_df = pd.DataFrame(
        [(model,
          (score[0] if isinstance(score, tuple) else score) * (100 if model_type == "Classification" else 1),
          changes[model] * (100 if model_type == "Classification" else 1))
            for model, score in results.items()],
        columns=["Model", metric_name, "Change"]
    )

    # Sort results by metric value in descending order
    results_df = results_df.sort_values(by=metric_name, ascending=(model_type == "Regression")).reset_index(drop=True)

    # Create a formatted change column with arrows and colors
    def format_change(change):
        if change > 0:
            return f"↑ {change:.2f}", "green"
        elif change < 0:
            return f"↓ {abs(change):.2f}", "red"
        else:
            return "→ 0.00", "gray"

    # Apply formatting to the Change column
    results_df["Formatted_Change"] = results_df["Change"].apply(format_change)

    # Create a styled dataframe
    styled_df = results_df.copy()
    styled_df["Change"] = styled_df["Formatted_Change"].apply(lambda x: x[0])

    # Display the DataFrame
    st.subheader("Results Table")
    st.dataframe(styled_df[["Model", metric_name, "Change"]], use_container_width=True)

    # Visualize results with Altair bar chart
    st.subheader("Results Visualization")
    base = alt.Chart(results_df).encode(
        x=alt.X('Model', sort=None),
        tooltip=['Model', metric_name, 'Change']
    )

    bars = base.mark_bar().encode(
        y=alt.Y(metric_name, sort='-x'),
        color=alt.Color('Change:Q', 
            scale=alt.Scale(
                domain=[-max(abs(results_df['Change'])), 0, max(abs(results_df['Change']))],
                range=['red', 'lightgray', 'green']
            )
        )
    )

    text = base.mark_text(
        align='center',
        baseline='bottom',
        dy=-5
    ).encode(
        text=alt.Text(f'{metric_name}:Q', format='.2f')
    )

    changes = base.mark_text(
        align='center',
        baseline='top',
        dy=10
    ).encode(
        text=alt.Text('Change:Q', format='+.2f'),
        color=alt.condition(
            'datum.Change > 0',
            alt.value('#90EE90'),
            alt.value('maroon')
        )
    )

    st.altair_chart((bars + text + changes).properties(width=600, height=400, title=f"Model Performance Comparison ({metric_name})"), use_container_width=True)

def main():
    step = stx.stepper_bar(steps=["Upload Data", "Select Sampling Method", "Train Models"])

    if step != st.session_state.step:
        st.session_state.step = step

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

    elif st.session_state.step == 1:
        if st.session_state.DF is not None:
            cv_chooser()
            if st.button("Confirm Sampling Method"):
                st.session_state.cv_confirmed = True
            if st.session_state.cv_confirmed and st.session_state.cv_type:
                st.info("You can now proceed to the next step by clicking on 'Train Models' above.")
        else:
            st.warning("Please upload data first")

    # Modify step 2 section
    elif st.session_state.step == 2:
        if st.session_state.DF is not None and st.session_state.cv is not None:
            # Model type selection in sidebar
            model_type = st.sidebar.radio("Select model type:", ["Regression", "Classification"])
            pkg = rgr if model_type == "Regression" else clf
            
            # Create tabs for training and downloading
            train_tab, download_tab = st.tabs(["Train Models", "Download Models"])
            
            with train_tab:
                with st.form(key="training_form"):
                    st.subheader("Model Parameters")
                    selected_models = []
                    for model_name in pkg.MODEL_MAPPING.keys():
                        if st.checkbox(f"Train {model_name}", key=f"train_{model_name}"):
                            selected_models.append(model_name)
                            with st.expander(f"{model_name} Parameters"):
                                pkg.MODEL_PARAMS[model_name]()
                    
                    train_button = st.form_submit_button("Train Selected Models")
                    
                    if train_button and selected_models:
                        X = st.session_state.DF.drop(columns=[st.session_state.result_col])
                        y = st.session_state.DF[st.session_state.result_col]
                        results = {}
                        
                        st.session_state.trained_models = {}
                        
                        with st.spinner('Training models...'):
                            for model_name in selected_models:
                                model = pkg.create_model(model_name)
                                accuracy_or_mae = pkg.score_model(model, X, y, st.session_state.cv)
                                results[model_name] = accuracy_or_mae
                                
                                trained_model = model.fit(X, y)
                                st.session_state.trained_models[model_name] = trained_model
                            
                            # Store results in session state
                            st.session_state.current_results = results
                            st.session_state.current_model_type = model_type
                            display_comparison_results(results, model_type)

            with download_tab:
                if st.session_state.current_results:
                    st.header("Model Performance Comparison")
                    display_comparison_results(st.session_state.current_results, st.session_state.current_model_type)
                    
                    st.header("Download Trained Model")
                    selected_model = st.selectbox(
                        "Select model to download",
                        options=list(st.session_state.trained_models.keys()),
                        key="download_select"
                    )
                    if selected_model:
                        download_model(
                            st.session_state.trained_models[selected_model],
                            selected_model
                        )
                else:
                    st.info("Train models first to see comparison and enable downloading")
        else:
            st.warning("Please upload data and select sampling method first")

if __name__ == "__main__":
    main()
