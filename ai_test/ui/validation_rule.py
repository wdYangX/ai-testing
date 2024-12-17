def render_validation_rule(st):
    st.title("Testing AI: VALIDATION RULE")

    # Sidebar configuration
    rule = st.sidebar.selectbox("Select Rule", ["Rule A", "Rule B", "Rule C"], help="Choose a rule for validation")
    validation_model = st.sidebar.file_uploader("Upload Validation Model (JSON)", type=["json"], help="Upload validation model")
    expected_result = st.sidebar.text_input("Expected Result", "", help="Enter the expected result")

    if st.sidebar.button("Run Validation Rule"):
        st.sidebar.success("Validation Rule executed!")
        st.success("Validation Rule completed!")

        # Display results
        # st.subheader("Validation Rule Results")
        # display_metrics([])
        # st.write("#### Detailed Results")
        # results_rule = {
        #     "IM name": ["IM1"],
        #     "Obj": ["3METER / No Obj"],
        #     "Bbox": ["Bbox"],
        #     "URL": ["https://via.placeholder.com/640x480.png?text=IM1"]
        # }
        # display_results(results_rule)
        # display_export_button("rule_results.csv")
