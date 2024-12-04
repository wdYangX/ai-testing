import streamlit as st
import urllib.parse

from ai_services import model_metrics

# Set page to wide mode
st.set_page_config(layout="wide", page_title="AI Validation Tool")


# --- Utility Functions ---
def display_metrics(metric):
    """Displays model validation metrics."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", metric[0])
        st.metric("F1-Score", metric[3])
    with col2:
        st.metric("Precision", metric[1])
        st.metric("Recall", metric[2])
    with col3:
        st.metric("IOU", "-")
        st.metric("mAP50", "-")


def display_results(results):
    """Displays results in a table format with clickable URLs."""
    st.dataframe(
        results,
        hide_index=True,
    )


def display_export_button(filename="results.csv"):
    """Displays a button to download results in CSV format."""
    st.download_button("Export Results", "Exported data in CSV format", file_name=filename)


# --- SIDEBAR ---
st.sidebar.title("Configuration")

# Using st.query_params to get query parameters
query_params = st.query_params
detail_view = query_params.get("page", ["main"])[0]

# --- MAIN PAGE ---
if detail_view == "main":
    # Radio button to switch between "Test Model" and "Validation Rule"
    selected_section = st.sidebar.radio("Select Section", ("Test Model", "Validation Rule"),
        help="Chọn phần muốn kiểm tra: Test Model (mặc định) hoặc Validation Rule", )

    # Test Model Section
    if selected_section == "Test Model":
        st.title("Testing AI: VALIDATION MODEL")

        # Configuration in the sidebar
        model = st.sidebar.selectbox("Select Model", ["yolov8m_g"], help="Chọn model để validate")
        data_zip = st.sidebar.file_uploader("Import data (Zip files)", type=["zip"], help="Nhập dữ liệu ảnh (dạng zip)")

        label_file = st.sidebar.file_uploader("Import Label", type=["zip"], help="Nhập label tương ứng")
        if st.sidebar.button("Run Validation"):
            # Detailed results with clickable URL
            if data_zip and label_file:
                st.sidebar.success("Validation started!")
                output, metric = model_metrics(data_zip, label_file)
                if output:
                    # Display results
                    st.subheader("Validation Model Results")
                    display_metrics(metric)
                    st.write("#### Detailed Results")
                    display_results(output)
                    display_export_button("model_results.csv")
                    st.success("Validation completed!")
            else:
                st.error('Invalid Input')

    # Validation Rule Section
    elif selected_section == "Validation Rule":
        st.title("Testing AI: VALIDATION RULE")

        # Configuration in the sidebar
        rule = st.sidebar.selectbox("Select Rule", ["Rule A", "Rule B", "Rule C"], help="Chọn rule để kiểm tra")
        validation_model = st.sidebar.file_uploader("Import data validation model (json)", type=["json"],
                                                    help="Nhập validation model")
        expected_result = st.sidebar.text_input("Expected Result", "", help="Kết quả mong muốn")

        if st.sidebar.button("Run Validation Rule"):
            st.sidebar.success("Validation Rule executed!")
            st.success("Validation Rule completed!")

        # Display results
        st.subheader("Validation Rule Results")
        display_metrics()

        # Detailed results with clickable URL
        st.write("#### Detailed Results")
        results_rule = {"IM name": ["IM1"], "Obj": ["3METER / No Obj"], "Bbox": ["Bbox"],
            "URL": ["https://via.placeholder.com/640x480.png?text=IM1"], }
        display_results(results_rule)
        display_export_button("rule_results.csv")

# --- DETAIL PAGE ---
elif detail_view == "detail":
    st.title("Image Detail View")

    # Get the image URL from the query parameters
    img_url = query_params.get("img_url", [""])[0]

    if img_url:
        # Display the image
        st.image(img_url, use_column_width=True)
        # Link to go back to the main page
        st.markdown(f"[Back to main page](?page=main)", unsafe_allow_html=True)
    else:
        st.error("No image URL provided!")
