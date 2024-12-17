import pandas as pd
import urllib.parse
from constant import cfg


# --- Utility Functions ---
def display_metrics(st, metric):
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


def display_results(st, results, s3_prefix):
    """Displays results in a table format with clickable URLs."""
    data_df = pd.DataFrame(results)
    data_df['confirm'] = True
    url = urllib.parse.urljoin(cfg.S3_ENDPOINT.rstrip("/") + "/", cfg.BUCKET_NAME.lstrip("/") + "/")
    url = urllib.parse.urljoin(url, s3_prefix.lstrip("/") + "/")
    # Prepare DataFrame for display
    data_df['view'] = data_df['image_name'].apply(lambda x: urllib.parse.urljoin(url, x))
    data_df.to_csv('latest_result.csv')

    data_df = data_df[['image_name', 'view'] + [col for col in data_df.columns if col not in ['image_name', 'view']]]
    del data_df['scores']
    del data_df['boxes']
    del data_df['g_boxes']
    del data_df['predict']
    del data_df['ground_truth']

    # Display data editor
    st.data_editor(data_df, column_config={
        "view": st.column_config.ImageColumn("Preview Image", help="Streamlit app preview screenshots", )},
                   hide_index=False, )


def display_export_button(st, filename="results.csv"):
    """Displays a button to download results in CSV format."""
    st.download_button("Export Results", "Exported data in CSV format", file_name=filename)
