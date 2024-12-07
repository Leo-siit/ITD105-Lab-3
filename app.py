import streamlit as st
import extra_streamlit_components as stx
import pandas as pd
import altair as alt
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

    elif st.session_state.step == 2:
        if st.session_state.DF is not None and st.session_state.cv is not None:
            with st.sidebar:
                model_type = st.radio("Select model type:", ["Regression", "Classification"])
                models = []
                pkg = rgr if model_type == "Regression" else clf
                models = pkg.MODEL_MAPPING.keys()

                for model_name in models:
                    expander = st.expander(model_name)
                    with expander:
                        pkg.MODEL_PARAMS[model_name]()

            X = st.session_state.DF.drop(columns=[st.session_state.result_col])
            y = st.session_state.DF[st.session_state.result_col]
            results = {}
            for model_name in models:
                model = pkg.create_model(model_name)
                accuracy_or_mae = pkg.score_model(model, X, y, st.session_state.cv)
                results[model_name] = accuracy_or_mae

            display_comparison_results(results, model_type)
        else:
            st.warning("Please upload data and select sampling method first")


if __name__ == "__main__":
    main()
