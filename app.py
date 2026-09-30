import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.decomposition import PCA
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="PCA Studio",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# THEME
# IMPORTANT: Only CSS is HTML here. All visible UI below uses
# native Streamlit components, so raw <div> text cannot appear.
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

    :root {
        --cream:#FAF8F1;
        --cream2:#F3F1E7;
        --white:#FFFFFF;
        --ink:#14263D;
        --muted:#68736E;
        --green:#39776D;
        --dark:#245B54;
        --sage:#DCE8D6;
        --sage2:#EEF5E9;
        --peach:#F7D9C5;
        --peach2:#FCEBE2;
        --yellow:#FFF4D9;
        --line:#E4E1D7;
    }

    html, body, [class*="css"] {
        font-family:"DM Sans", sans-serif;
    }

    .stApp {
        background:var(--cream);
        color:var(--ink);
    }

    .block-container {
        max-width:1500px;
        padding:1.2rem 2rem 3rem;
    }

    #MainMenu, footer {
        visibility:hidden;
    }

    header {
        background:transparent !important;
    }

    /* SIDEBAR */
    section[data-testid="stSidebar"] {
        background:#F3F2E9;
        border-right:1px solid #E4E2D8;
    }

    section[data-testid="stSidebar"] > div {
        padding:1.2rem .95rem;
    }

    section[data-testid="stSidebar"] .stMarkdown h1 {
        font-family:"Playfair Display", serif;
        color:var(--ink);
    }

    div[role="radiogroup"] {
        gap:.25rem;
    }

    div[role="radiogroup"] label {
        border-radius:15px !important;
        padding:.68rem .75rem !important;
        border:1px solid transparent !important;
    }

    div[role="radiogroup"] label:hover {
        background:#E7EDE1 !important;
    }

    /* HEADINGS */
    h1, h2, h3 {
        color:var(--ink) !important;
    }

    .serif {
        font-family:"Playfair Display", serif !important;
    }

    /* NATIVE STREAMLIT CONTAINERS */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background:rgba(255,255,255,.82);
        border:1px solid var(--line);
        border-radius:24px;
        box-shadow:0 9px 25px rgba(64,70,55,.055);
    }

    /* BUTTONS */
    .stButton > button,
    .stDownloadButton > button {
        border:0 !important;
        border-radius:15px !important;
        background:var(--green) !important;
        color:#fff !important;
        font-weight:700 !important;
        min-height:45px;
        box-shadow:0 7px 17px rgba(57,119,109,.18);
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background:var(--dark) !important;
    }

    /* FILE UPLOADER */
    [data-testid="stFileUploader"] {
        background:#FFFDF8;
        border:2px dashed #8BAA9F;
        border-radius:20px;
        padding:.55rem;
    }

    [data-testid="stFileUploaderDropzone"] {
        background:#FFFDF8 !important;
    }

    /* SLIDER */
    div[data-baseweb="slider"] > div > div {
        background:var(--green) !important;
    }

    /* IMAGES */
    [data-testid="stImage"] img {
        border-radius:18px;
        border:1px solid #E7E3D8;
    }

    /* METRICS */
    div[data-testid="stMetric"] {
        border:1px solid var(--line);
        border-radius:18px;
        padding:14px;
        background:#FFFDF9;
    }

    /* ALERTS */
    div[data-testid="stAlert"] {
        border-radius:15px;
    }

    /* DATAFRAME */
    [data-testid="stDataFrame"] {
        border-radius:18px;
        overflow:hidden;
    }

    /* TABS */
    button[data-baseweb="tab"] {
        color:var(--dark) !important;
    }

    /* SMALL SCREEN */
    @media (max-width:900px) {
        .block-container {
            padding-left:1rem;
            padding-right:1rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PCA
# ============================================================

def compress_channel(channel, components):
    channel = channel.astype(float)

    mean = channel.mean(axis=0)
    centered = channel - mean

    components = min(
        components,
        centered.shape[0],
        centered.shape[1],
    )

    pca = PCA(n_components=components)
    transformed = pca.fit_transform(centered)
    reconstructed = pca.inverse_transform(transformed)

    reconstructed = np.clip(
        reconstructed + mean,
        0,
        255,
    )

    variance = float(np.sum(pca.explained_variance_ratio_))
    return reconstructed, variance


def compress_image(image_array, components):
    r, rv = compress_channel(image_array[:, :, 0], components)
    g, gv = compress_channel(image_array[:, :, 1], components)
    b, bv = compress_channel(image_array[:, :, 2], components)

    result = np.stack([r, g, b], axis=2)
    result = np.round(result).astype(np.uint8)

    variance = (rv + gv + bv) / 3
    return result, variance


def prepare_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")

    max_size = 512

    if max(image.size) > max_size:
        scale = max_size / max(image.size)
        new_size = (
            int(image.size[0] * scale),
            int(image.size[1] * scale),
        )
        image = image.resize(new_size)

    return image


def save_png(image_array):
    buffer = io.BytesIO()
    Image.fromarray(image_array).save(buffer, format="PNG")
    return buffer.getvalue()


def clear_result():
    for key in [
        "compressed_image",
        "original_image",
        "components",
        "variance",
        "mse",
        "psnr",
    ]:
        st.session_state.pop(key, None)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("🌿", unsafe_allow_html=False)

    # Native text instead of HTML content.
    st.markdown("### PCA Studio")
    st.caption("Image Compression")

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🖼️ Compress",
            "📊 Analysis",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.info(
        "🌱 Smaller images\n\n"
        "Preserve important visual information with PCA."
    )

# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    # Greeting
    top1, top2 = st.columns([5, 1])
    with top2:
        st.markdown("☀️  **Good Morning!**")

    # HERO
    with st.container(border=True):
        hero_left, hero_right = st.columns([1.45, 1])

        with hero_left:
            st.caption("SMARTER · LIGHTER · CLEARER")
            st.markdown("# PCA Image Compression")
            st.write(
                "Reduce image dimensions while preserving "
                "important visual information using "
                "Principal Component Analysis."
            )

            if st.button(
                "Start Compressing  →",
                use_container_width=False,
            ):
                st.session_state["home_start"] = True
                st.toast("Upload your image below to begin.")

        with hero_right:
            st.markdown("### 🏔️  →  🖼️")
            st.markdown("#### 🧠 PCA")
            st.caption("Same beauty · smaller dimensions")

    st.markdown("## Start Compressing")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is None:
        st.info(
            "☁️ Upload a JPG, JPEG or PNG image to start."
        )

    if uploaded_file is not None:
        image = prepare_image(uploaded_file)
        image_array = np.array(image)

        max_components = min(
            image_array.shape[0],
            image_array.shape[1],
        )

        default_components = min(50, max_components)

        control_left, control_right = st.columns(
            [1.15, 1],
            gap="large",
        )

        with control_left:
            with st.container(border=True):
                st.markdown("### 🖼️ Upload Your Image")
                st.caption("Your image is ready for compression.")
                st.image(image_array, width="stretch")

                a, b, c = st.columns(3)
                a.metric("Width", image.size[0])
                b.metric("Height", image.size[1])
                c.metric("Channels", 3)

        with control_right:
            with st.container(border=True):
                st.markdown("### 🎛️ PCA Components")

                components = st.slider(
                    "Number of components",
                    1,
                    max_components,
                    default_components,
                )

                st.info(
                    f"🧠 Using **{components}** components. "
                    "Higher values generally retain more details."
                )

                if st.button(
                    "Compress Image  →",
                    use_container_width=True,
                ):
                    clear_result()

                    with st.spinner(
                        "Applying PCA to your image..."
                    ):
                        compressed, variance = compress_image(
                            image_array,
                            components,
                        )

                    mse = mean_squared_error(
                        image_array,
                        compressed,
                    )

                    psnr = peak_signal_noise_ratio(
                        image_array,
                        compressed,
                        data_range=255,
                    )

                    st.session_state["compressed_image"] = compressed
                    st.session_state["original_image"] = image_array
                    st.session_state["components"] = components
                    st.session_state["variance"] = variance
                    st.session_state["mse"] = mse
                    st.session_state["psnr"] = psnr

                    st.success("Compression completed!")

        if "compressed_image" in st.session_state:
            compressed = st.session_state["compressed_image"]
            original = st.session_state["original_image"]
            components = st.session_state["components"]
            variance = st.session_state["variance"]
            mse = st.session_state["mse"]
            psnr = st.session_state["psnr"]

            st.markdown("## Compression Results")

            m1, m2, m3, m4 = st.columns(4)

            with m1:
                st.metric(
                    "▱  Components",
                    components,
                )

            with m2:
                st.metric(
                    "◔  Variance Retained",
                    f"{variance * 100:.1f}%",
                )

            with m3:
                st.metric(
                    "⌁  MSE",
                    f"{mse:.4f}",
                )

            with m4:
                st.metric(
                    "★  PSNR",
                    f"{psnr:.2f} dB",
                )

            st.markdown("## Image Comparison")

            c1, c2 = st.columns(2, gap="large")

            with c1:
                st.markdown("**Original Image**")
                st.image(
                    original,
                    width="stretch",
                )

            with c2:
                st.markdown("**Compressed Image (PCA)**")
                st.image(
                    compressed,
                    width="stretch",
                )

            st.download_button(
                "↓  Download Compressed Image",
                data=save_png(compressed),
                file_name="PCA_Compressed_Image.png",
                mime="image/png",
                use_container_width=True,
            )

# ============================================================
# COMPRESS
# ============================================================

elif page == "🖼️ Compress":

    st.markdown("## Compress Your Image")
    st.caption(
        "Upload → choose components → compress → compare → download"
    )

    uploaded_file = st.file_uploader(
        "Upload your image",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        image = prepare_image(uploaded_file)
        image_array = np.array(image)

        max_components = min(
            image_array.shape[0],
            image_array.shape[1],
        )

        with st.container(border=True):
            st.markdown("### 🎛️ Choose PCA Components")

            components = st.slider(
                "Number of principal components",
                1,
                max_components,
                min(50, max_components),
            )

            if st.button(
                "Compress Image  →",
                use_container_width=True,
            ):
                with st.spinner("Applying PCA..."):
                    compressed, variance = compress_image(
                        image_array,
                        components,
                    )

                mse = mean_squared_error(
                    image_array,
                    compressed,
                )

                psnr = peak_signal_noise_ratio(
                    image_array,
                    compressed,
                    data_range=255,
                )

                st.session_state["compressed_image"] = compressed
                st.session_state["original_image"] = image_array
                st.session_state["components"] = components
                st.session_state["variance"] = variance
                st.session_state["mse"] = mse
                st.session_state["psnr"] = psnr

                st.success("Compression completed!")

        if "compressed_image" in st.session_state:
            compressed = st.session_state["compressed_image"]

            st.markdown("## Results")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(
                "Components",
                st.session_state["components"],
            )
            m2.metric(
                "Variance",
                f'{st.session_state["variance"] * 100:.2f}%',
            )
            m3.metric(
                "MSE",
                f'{st.session_state["mse"]:.4f}',
            )
            m4.metric(
                "PSNR",
                f'{st.session_state["psnr"]:.2f} dB',
            )

            left, right = st.columns(2, gap="large")

            with left:
                st.markdown("**Original Image**")
                st.image(
                    st.session_state["original_image"],
                    width="stretch",
                )

            with right:
                st.markdown("**Compressed Image (PCA)**")
                st.image(
                    compressed,
                    width="stretch",
                )

            st.download_button(
                "↓  Download Compressed Image",
                data=save_png(compressed),
                file_name="PCA_Compressed_Image.png",
                mime="image/png",
                use_container_width=True,
            )

# ============================================================
# ANALYSIS
# ============================================================

else:

    st.markdown("## PCA Analysis")
    st.caption(
        "See how the number of components changes "
        "variance retention and reconstruction quality."
    )

    uploaded_file = st.file_uploader(
        "Upload image for analysis",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        image = prepare_image(uploaded_file)
        image_array = np.array(image)

        max_components = min(
            image_array.shape[0],
            image_array.shape[1],
        )

        possible = [5, 10, 20, 50, 100]
        components_list = [
            x for x in possible
            if x <= max_components
        ]

        if not components_list:
            components_list = [max_components]

        results = []
        progress = st.progress(0)

        for i, components in enumerate(components_list):
            compressed, variance = compress_image(
                image_array,
                components,
            )

            mse = mean_squared_error(
                image_array,
                compressed,
            )

            psnr = peak_signal_noise_ratio(
                image_array,
                compressed,
                data_range=255,
            )

            results.append(
                {
                    "Components": components,
                    "Explained Variance (%)": variance * 100,
                    "MSE": mse,
                    "PSNR (dB)": psnr,
                }
            )

            progress.progress(
                (i + 1) / len(components_list)
            )

        results_df = pd.DataFrame(results)

        st.markdown("### Results Table")
        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
        )

        chart1, chart2 = st.columns(2)

        with chart1:
            st.markdown("**Explained Variance (%)**")
            st.line_chart(
                results_df.set_index("Components")[
                    "Explained Variance (%)"
                ]
            )

        with chart2:
            st.markdown("**PSNR (dB)**")
            st.line_chart(
                results_df.set_index("Components")[
                    "PSNR (dB)"
                ]
            )

        st.markdown("**MSE**")
        st.line_chart(
            results_df.set_index("Components")["MSE"]
        )

        st.download_button(
            "↓  Download Analysis CSV",
            data=results_df.to_csv(index=False),
            file_name="PCA_Analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )

    else:
        st.info(
            "📊 Upload an image to generate the PCA analysis."
        )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🌿 PCA Studio · Image Compression using Principal Component Analysis"
)