import streamlit as st
from PIL import Image
from src.schemas import Landmark, LandmarkType

st.title("Interactive Landmark Creator")

if "landmarks" not in st.session_state:
    st.session_state.landmarks = []
if "uuid_counter" not in st.session_state:
    st.session_state.uuid_counter = 0
if "clicked" not in st.session_state:
    st.session_state.clicked = False
if "click_pos" not in st.session_state:
    st.session_state.click_pos = None


uploaded_file = st.file_uploader("Upload a map image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)

    # Detect clicks on the image (approximate) using Streamlit's experimental_click_coords
    coords = st.experimental_get_query_params().get("click", None)
    # experimental_click_coords is limited and may require custom JS, so let's do a workaround:

    # Instead, use st_canvas from streamlit_drawable_canvas for click detection:

    from streamlit_drawable_canvas import st_canvas

    canvas_result = st_canvas(
        fill_color="",
        stroke_width=2,
        stroke_color="#000",
        background_image=img,
        height=img.height,
        width=img.width,
        drawing_mode="point",
        key="canvas",
    )

    if canvas_result.json_data is not None:
        points = canvas_result.json_data["objects"]
        if points and not st.session_state.clicked:
            # Get the first click point
            last_point = points[-1]
            x, y = int(last_point["left"]), int(last_point["top"])
            st.session_state.clicked = True
            st.session_state.click_pos = (x, y)

    if st.session_state.clicked:
        x, y = st.session_state.click_pos
        st.markdown(f"**Clicked position:** x={x}, y={y}")

        with st.form("landmark_form"):
            name = st.text_input("Landmark Name")
            lm_type = st.selectbox("Landmark Type", [t.value for t in LandmarkType])
            submitted = st.form_submit_button("Add Landmark")

            if submitted and name:
                lm = Landmark(
                    uuid=st.session_state.uuid_counter,
                    longitude=float(x),
                    latitude=float(y),
                    name=name,
                    type=LandmarkType(lm_type),
                )
                st.session_state.landmarks.append(lm)
                st.session_state.uuid_counter += 1
                st.session_state.clicked = False  # reset for next click
                st.experimental_rerun()

    if st.session_state.landmarks:
        st.markdown("### Landmarks created:")
        for lm in st.session_state.landmarks:
            st.write(lm.dict())
