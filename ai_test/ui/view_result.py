import os
import ast
import requests
import pandas as pd
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

# Define labels and their corresponding colors
class_labels = {0: ("Forklift", "red"), 1: ("Hand pallet jack", "green"), 2: ("Electric pallet jack", "blue"),
                3: ("Reach truck", "orange"), 4: ("Truck", "purple"), 5: ("Pallet", "yellow"),
                6: ("Product box", "pink"),  # 7: ("Product package", "cyan"),
                8: ("Fallen package", "brown"), 9: ("Person", "gray"), }


# Map class indices to class names
def map_classes_with_colors(class_indices):
    out = []
    for idx in class_indices:
        if class_labels.get(int(idx)):
            out += [(class_labels.get(int(idx), ("Unknown", "black"))[0],
                     class_labels.get(int(idx), ("Unknown", "black"))[1])]
    return out


# Function to fetch and display an image with bounding boxes and growth boxes
def display_image_with_boxes(st, index, selected_classes, show_predictions, show_ground_truth, show_text_predictions,
                             show_text_ground_truth):
    # Fetch image from URL
    response = requests.get(st.session_state.image_urls[index])
    image = Image.open(BytesIO(response.content)).convert("RGB")

    # Parse bounding box and growth box data
    box_data = ast.literal_eval(st.session_state.bounding_boxes[index])  # Predictions
    growth_data = ast.literal_eval(st.session_state.growth_boxes[index])  # Ground truth
    pred_data = ast.literal_eval(st.session_state.predictions[index])  # Predicted class IDs
    gt_data = ast.literal_eval(st.session_state.ground_truths[index])  # Ground truth class IDs

    # Draw bounding boxes
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype('ai_test/ui/static/font/Arial_Bold.ttf', 25)  # Use default font

    # Draw predicted boxes (solid lines) and text if the checkbox is selected
    if show_predictions:
        for box, pred_class in zip(box_data, pred_data):
            if pred_class in selected_classes:
                x_min, y_min, x_max, y_max = box
                class_name, color = class_labels.get(pred_class, ("Unknown", "black"))
                draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=3)  # Solid color
                if show_text_predictions:
                    draw.text((x_min, y_min), class_name, fill="blue", font=font)  # Add text label

    # Draw ground truth boxes (dashed lines) and text if the checkbox is selected
    if show_ground_truth:
        for growth, gt_class in zip(growth_data, gt_data):
            if gt_class in selected_classes:
                x_min, y_min, x_max, y_max = growth
                class_name, color = class_labels.get(gt_class, ("Unknown", "black"))
                # Draw dashed lines for ground truth boxes in green
                draw.line([x_min, y_min, x_max, y_min], fill=color, width=3)  # Top edge
                draw.line([x_max, y_min, x_max, y_max], fill=color, width=3)  # Right edge
                draw.line([x_max, y_max, x_min, y_max], fill=color, width=3)  # Bottom edge
                draw.line([x_min, y_max, x_min, y_min], fill=color, width=3)  # Left edge
                if show_text_ground_truth:
                    draw.text((x_min, y_min), f"{class_name}", fill="red", font=font)  # Add GT text label

    return image


# Function to update the image index
def update_image(st, direction):
    if direction == "next":
        st.session_state.img_index = (st.session_state.img_index + 1) % len(st.session_state.image_urls)
    elif direction == "prev":
        st.session_state.img_index = (st.session_state.img_index - 1) % len(st.session_state.image_urls)


# Function to save validation status to the CSV
def save_validation_status(st, status):
    if status == 'Approved':
        status = True
    elif status == 'Reject':
        status = False
    st.session_state.df.at[st.session_state.img_index, 'confirm'] = status
    st.session_state.df.to_csv(st.session_state.csv_file_path, index=False)  # Save the updated DataFrame


def create_html_with_color(mapped_data):
    return " &nbsp; ".join([f'<span style="color:{color};">{label}</span>' for label, color in mapped_data])


def views(st):
    # Check if session_state has image index, set default if not
    if "img_index" not in st.session_state:
        st.session_state.img_index = 0
    data = []
    for class_id, (label, color) in class_labels.items():
        data.append({"Class": label,
                     "Prediction": f"""<div style="display: inline-block; width: 15px; height: 15px; background-color: {color}; margin-right: 5px;"></div>Solid Line""",
                     "Ground Truth": f"""<div style="display: inline-block; width: 15px; height: 15px; background-color: {color}; margin-right: 5px; border: 2px dashed;"></div>Dashed Line"""})
    df = pd.DataFrame(data)
    # Display the current image with bounding boxes and predictions
    col1, col2 = st.columns([3, 1])  # First column for the image, second for the form
    # Display image name and related parameters
    image_name = st.session_state.image_names[st.session_state.img_index]
    # Add a multi-select dropdown for filtering classes
    with col2:

        st.subheader("Filter Classes")
        selected_classes = st.multiselect("Select classes to display", options=list(class_labels.keys()),
                                          format_func=lambda x: class_labels[x][0],
                                          # Display the label instead of the class ID
                                          default=list(class_labels.keys()),  # Show all classes by default
                                          )
        # Checkboxes for showing predictions, ground truth, and text labels
        st.subheader("Display")
        c11, c12 = st.columns([1, 1])
        with c11:
            show_predictions = st.checkbox("Show Predictions", value=True)
        with c12:
            show_text_predictions = st.checkbox("Show Prediction Text", value=True)
        c21, c22 = st.columns([1, 1])
        with c21:
            show_ground_truth = st.checkbox("Show Ground Truth", value=True)
        with c22:
            show_text_ground_truth = st.checkbox("Show Ground Truth Text", value=True)
        st.subheader("Confirmation Image")
        # Form for manual validation
        with st.form(key="Streamlit Image Comparison"):
            status = st.radio("Choose manual image testing 👉", key="Approved", options=["Approved", "Reject"], )
            submit = st.form_submit_button("Save 🔥")

            if submit:
                # Save the validation status
                save_validation_status(st, status)
                st.success(f"Validation status for image '{image_name}' has been saved as '{status}'")
        # "Prev" and "Next" buttons
        c1, _, c2, c3 = st.columns([1, 0.6, 1.5, 1])
        with c1:
            if st.button("⬅️ Prev"):
                update_image(st, "prev")
        with c2:
            # Display total images
            st.write(f"Image {st.session_state.img_index + 1}/{st.session_state.total_images}")
        with c3:
            if st.button("Next ➡️"):
                update_image(st, "next")

        st.download_button(label="Export Result", data=df.to_csv(index=False), file_name="validation_results.csv",
                           mime="text/csv", )
        # Display color-coded legend
        st.markdown("### Legend")
        st.write(df.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Display the image in the main column
    with col1:
        st.image(
            display_image_with_boxes(st, st.session_state.img_index, selected_classes, show_predictions, show_ground_truth,
                                     show_text_predictions, show_text_ground_truth),
            caption=f"Image {st.session_state.img_index + 1}/{st.session_state.total_images}", use_container_width=True, )
        st.markdown(f"### Image Name: {image_name}")
        c111, c222, c333, c444 = st.columns([1, 1, 1, 1])
        mapped_predictions = map_classes_with_colors(eval(f'{st.session_state.predictions[st.session_state.img_index]}'))
        mapped_ground_truths = map_classes_with_colors(eval(f'{st.session_state.ground_truths[st.session_state.img_index]}'))

        # Display the predictions and ground truths with color-coded class names
        st.markdown(f"**Predictions (Bounding Boxes):** {create_html_with_color(mapped_predictions)}",
                    unsafe_allow_html=True)
        st.markdown(f"**Ground Truth (Bounding Boxes):** {create_html_with_color(mapped_ground_truths)}",
                    unsafe_allow_html=True)
        with c111:
            st.markdown(f"**Accuracy:** {st.session_state.accuracy[st.session_state.img_index]}")
        with c222:
            st.markdown(f"**Precision:** {st.session_state.precision[st.session_state.img_index]}")
        with c333:
            st.markdown(f"**Recall:** {st.session_state.recall[st.session_state.img_index]}")
        with c444:
            st.markdown(f"**F1-Score:** {st.session_state.f1_score[st.session_state.img_index]}")


def render_view_result(st):
    st.markdown("""
        <style>
        [data-testid="stSidebar"], [data-testid="collapsedControl"] {
            display: none;
        }
        </style>
        """, unsafe_allow_html=True)
    st.title("Testing AI: View Result")

    # Add "Back Home" button with section reset
    if st.button("Back Home"):
        st.session_state.selected_section = "Test Model"  # Reset to default section
        st.rerun()

    csv_file_path = './latest_result.csv'  # Update this path as needed
    # Check if CSV file exists
    if os.path.isfile(csv_file_path):
        if "data_loaded" not in st.session_state:
            if os.path.isfile(csv_file_path):
                df = pd.read_csv(csv_file_path)
                st.session_state.update({
                    'csv_file_path': csv_file_path,
                    'df': df,
                    'image_urls': df['view'].tolist(),
                    'image_names': df['image_name'].tolist(),
                    'bounding_boxes': df['boxes'].tolist(),
                    'growth_boxes': df['g_boxes'].tolist(),
                    'predictions': df['predict'].tolist(),
                    'ground_truths': df['ground_truth'].tolist(),
                    'accuracy': df['accuracy'].tolist(),
                    'precision': df['precision'].tolist(),
                    'recall': df['recall'].tolist(),
                    'f1_score': df['f1-Score'].tolist(),
                    'total_images': len(df),
                    'data_loaded': True
                })
            else:
                st.error(f"CSV file '{csv_file_path}' not found.")
                return
        views(st)
