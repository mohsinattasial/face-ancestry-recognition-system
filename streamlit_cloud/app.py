"""Streamlit Community Cloud portfolio adapter for face-ancestry-recognition-system.

This does not replace the original implementation.
"""
from pathlib import Path

import streamlit as st

TITLE = "Image Classification Research Notes"
ICON = "🔬"
DESC = "Coursework code from a Keras image classification experiment and its documented methodological limitations."
CATEGORY = "Ethics / CV experiment"
NOTICE = "Ancestry, race or ethnicity cannot be reliably identified from facial appearance. This app intentionally does not offer such prediction."
GITHUB = "https://github.com/mohsinattasial/face-ancestry-recognition-system"
VERCEL = "https://mohsin-image-research.vercel.app"
LOCAL_COMMAND = "See README.md before running experimental scripts."
FEATURES = ["Model training pipeline","Preprocessing scripts","Limitations and fairness considerations"]

st.set_page_config(page_title=TITLE + " | Mohsin Atta", page_icon=ICON, layout="wide")
st.caption("Mohsin Atta · GitHub Portfolio · Streamlit Cloud edition")
st.title(ICON + " " + TITLE)
st.write(DESC)
st.caption(CATEGORY)

left, right = st.columns(2)
with left:
    st.link_button("GitHub source code", GITHUB, use_container_width=True)
with right:
    st.link_button("Existing browser version", VERCEL, use_container_width=True)

st.info(NOTICE)
with st.expander("Original project and features", expanded=False):
    for feature in FEATURES:
        st.write("• " + feature)
    st.code(LOCAL_COMMAND, language="bash")

st.subheader("Project overview")
st.write("This page is a deployable portfolio presentation of the original repository. Use the GitHub link above to inspect the code and run it in its intended local environment.")

st.divider()
st.caption("© 2026 Mohsin Atta · Educational portfolio · MIT-licensed original code where applicable")
