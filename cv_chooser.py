import streamlit as st
from cv_types import *

@st.fragment
def cv_chooser():
    with st.container():
        with st.form("choose_cv"):
            options = ["KFold", "Leave One Out", "Shuffle Split", "Train Test Split"]
            if st.session_state.cv_type is not None:
                options = [st.session_state.cv_type] + [o for o in options if o != st.session_state.cv_type]
            c1, c2 = st.columns(2)
            cv_type = c1.selectbox("Select Sampling Technique", options)
            c2.markdown("<h3 style='font-size: 3px;'></h3>", unsafe_allow_html=True)
            if c2.form_submit_button("Confirm"):
                st.session_state.cv_args = {}
                st.session_state.cv_type = cv_type
                st.rerun(scope="fragment")

        with st.form("cv_params"):
            if st.session_state.cv_type == "KFold":
                params = kfold_view(st, st.session_state.cv_args)
            elif st.session_state.cv_type == "Leave One Out":
                params = leave_one_out_view(st, st.session_state.cv_args)
            elif st.session_state.cv_type == "Shuffle Split":
                params = shuffle_split_view(st, st.session_state.cv_args)
            elif st.session_state.cv_type == "Train Test Split":
                params = train_test_split_view(st, st.session_state.cv_args)
            if st.form_submit_button("Update"):
                st.session_state.cv_args = params
                st.session_state.cv = create_cross_val(st.session_state.cv_type, **st.session_state.cv_args)
