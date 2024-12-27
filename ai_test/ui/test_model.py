from services.ai_services import model_metrics
from utils.extract_zip import extract_zip
from constant import cfg
from utils.helpers import display_metrics, display_results


def render_test_model(st):
    st.title("Testing AI: VALIDATION MODEL")
    # Configuration in the sidebar
    model = st.sidebar.selectbox("Select Model", ["yolov8m_g"], help="Chọn model để validate")
    data_zip = st.sidebar.file_uploader("Import data (Zip files)", type=["zip"], help="Nhập dữ liệu ảnh (dạng zip)")

    label_file = st.sidebar.file_uploader("Import Label", type=["zip"], help="Nhập label tương ứng")
    if st.sidebar.button("Run Validation"):
        # Detailed results with clickable URL
        if data_zip and label_file:
            file_pth = extract_zip(data_zip)
            st.sidebar.success("Validation started!")
            output, metric = model_metrics(data_zip, label_file)
            if output:
                # Display results
                st.subheader("Validation Model Results")
                display_metrics(st, metric)
                st.write("#### Detailed Results")
                display_results(st, output, file_pth)
                st.success("Validation completed!")
        else:
            st.error('Invalid Input')