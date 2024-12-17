import streamlit as st

from ui.view_result import render_view_result
from ui.test_model import render_test_model

# Set page to wide mode
st.set_page_config(layout="wide", page_title="AI Validation Tool")

# --- SIDEBAR ---
st.sidebar.title("Configuration")


# Get the selected section from session state or default to "Test Model"
if "selected_section" not in st.session_state:
    st.session_state.selected_section = "Test Model"  # Default to "Test Model"

# Radio button to switch between "Test Model", "Validation Rule", "View Result"
selected_section = st.sidebar.radio("Select Section", ("Test Model", "Validation Rule", "View Result"),
                                    help="Chọn phần muốn kiểm tra: Test Model (mặc định) hoặc Validation Rule",
                                    index=("Test Model" == st.session_state.selected_section) * 0 +
                                          ("Validation Rule" == st.session_state.selected_section) * 1 +
                                          ("View Result" == st.session_state.selected_section) * 2)

# Set session state to remember the selected section
st.session_state.selected_section = selected_section

# Test Model Section
if selected_section == "Test Model":
    render_test_model(st)
if selected_section == "View Result":
    render_view_result(st)
# Validation Rule Section
elif selected_section == "Validation Rule":
    pass
