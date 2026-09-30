import streamlit as st
import numpy as np
import pandas as pd
import io

from PIL import Image
from sklearn.decomposition import PCA
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="PCA Studio",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');


/* ---------------------------------------------------------
   COLORS
--------------------------------------------------------- */

:root {
    --cream: #FAF8F1;
    --cream-dark: #F3F0E6;

    --white: #FFFFFF;

    --ink: #14263D;
    --muted: #68736E;

    --green: #39776D;
    --green-dark: #245B54;

    --sage: #DCE8D6;
    --sage-light: #EEF5E9;

    --peach: #F7D9C5;
    --peach-light: #FCEBE2;

    --yellow: #F6E8B5;
    --yellow-light: #FFF4D9;

    --lavender: #E9E2F4;

    --border: #E4E1D7;
}


/* ---------------------------------------------------------
   GENERAL APP
--------------------------------------------------------- */

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
}

.stApp {
    background: var(--cream);
    color: var(--ink);
}

.block-container {
    max-width: 1500px;
    padding-top: 1.4rem;
    padding-left: 2.2rem;
    padding-right: 2.2rem;
    padding-bottom: 3rem;
}


/* Remove Streamlit default elements */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}


/* ---------------------------------------------------------
   SIDEBAR
--------------------------------------------------------- */

section[data-testid="stSidebar"] {
    background: #F3F2E9;
    border-right: 1px solid #E4E2D8;
}

section[data-testid="stSidebar"] > div {
    padding: 1.5rem 1rem;
}


/* Brand */

.brand {
    text-align: center;
    padding: 0.5rem 0 1.5rem 0;
}

.brand-icon {
    width: 68px;
    height: 68px;

    margin: 0 auto 0.7rem auto;

    border-radius: 22px;

    display: flex;
    align-items: center;
    justify-content: center;

    font-size: 34px;

    background: linear-gradient(
        145deg,
        #DCE8D6,
        #F7D9C5
    );

    box-shadow:
        0 10px 25px rgba(48, 87, 74, 0.10);
}

.brand-title {
    font-family: "Playfair Display", serif;

    font-size: 1.65rem;

    font-weight: 700;

    color: var(--ink);

    margin: 0;
}

.brand-subtitle {
    color: var(--green-dark);

    font-size: 0.9rem;

    margin-top: 0.15rem;
}


/* Navigation */

div[role="radiogroup"] {
    gap: 0.35rem;
}

div[role="radiogroup"] label {
    border-radius: 16px !important;

    padding: 0.72rem 0.85rem !important;

    border: 1px solid transparent !important;

    background: transparent !important;

    transition: 0.2s ease;
}

div[role="radiogroup"] label:hover {
    background: #E7EDE1 !important;
}


/* Sidebar bottom card */

.sidebar-card {
    background:
        linear-gradient(
            145deg,
            #EAF3E6,
            #FFF0E7
        );

    border-radius: 22px;

    padding: 1.1rem;

    text-align: center;

    margin-top: 1rem;
}


/* ---------------------------------------------------------
   TOP BAR
--------------------------------------------------------- */

.topbar {
    display: flex;

    justify-content: flex-end;

    align-items: center;

    gap: 0.6rem;

    color: var(--ink);

    font-size: 0.95rem;

    margin-bottom: 0.8rem;
}

.sun {
    font-size: 1.3rem;
}


/* ---------------------------------------------------------
   HERO
--------------------------------------------------------- */

.hero {
    position: relative;

    overflow: hidden;

    min-height: 285px;

    border: 1px solid #E6E2D6;

    border-radius: 30px;

    padding: 2.7rem 3rem;

    background:
        radial-gradient(
            circle at 82% 20%,
            rgba(247,217,197,.85) 0 8%,
            transparent 8.5%
        ),

        radial-gradient(
            circle at 72% 88%,
            rgba(220,232,214,.9) 0 16%,
            transparent 16.5%
        ),

        linear-gradient(
            120deg,
            #FFF9E9 0%,
            #F2F4E8 48%,
            #F8E6D7 100%
        );

    box-shadow:
        0 14px 35px rgba(56, 67, 52, 0.08);

    margin-bottom: 1.4rem;
}

.hero-small {
    letter-spacing: 0.22rem;

    font-size: 0.78rem;

    color: #617148;

    font-weight: 700;

    text-transform: uppercase;

    margin-bottom: 0.6rem;
}

.hero h1 {
    font-family: "Playfair Display", serif;

    color: var(--ink);

    font-size: clamp(2.2rem, 4vw, 3.7rem);

    line-height: 1.04;

    margin: 0 0 0.8rem 0;

    max-width: 700px;
}

.hero p {
    color: #44545A;

    font-size: 1.05rem;

    line-height: 1.55;

    max-width: 590px;

    margin-bottom: 1rem;
}


/* Hero illustration */

.hero-art {
    position: absolute;

    right: 4%;

    top: 15%;

    width: 310px;

    height: 210px;
}

.photo-card {
    position: absolute;

    width: 115px;

    height: 92px;

    border: 7px solid white;

    border-radius: 13px;

    background:
        linear-gradient(
            140deg,
            #80A98E,
            #D8E7D5
        );

    box-shadow:
        0 12px 24px rgba(30, 61, 53, .15);

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 2.8rem;
}

.photo-one {
    left: 20px;

    top: 52px;

    transform: rotate(-7deg);
}

.photo-two {
    right: 25px;

    bottom: 25px;

    width: 88px;

    height: 72px;

    transform: rotate(7deg);
}

.arrow-art {
    position: absolute;

    left: 135px;

    top: 83px;

    font-size: 2.8rem;

    color: var(--green-dark);

    z-index: 3;
}

.pca-badge {
    position: absolute;

    right: 75px;

    top: 4px;

    background: rgba(255,255,255,.90);

    border: 1px solid #DDE3D5;

    border-radius: 18px;

    padding: 0.55rem 0.85rem;

    color: var(--green-dark);

    font-weight: 700;

    font-size: 0.9rem;
}


/* ---------------------------------------------------------
   SECTION HEADINGS
--------------------------------------------------------- */

.section-heading {
    font-family: "Playfair Display", serif;

    color: var(--ink);

    font-size: 1.4rem;

    font-weight: 700;

    margin-top: 1.2rem;

    margin-bottom: 0.75rem;
}


/* ---------------------------------------------------------
   CARDS
--------------------------------------------------------- */

.soft-card {
    background: rgba(255,255,255,.72);

    border: 1px solid var(--border);

    border-radius: 24px;

    padding: 1.25rem;

    box-shadow:
        0 9px 25px rgba(64, 70, 55, .055);
}

.card-title {
    font-family: "Playfair Display", serif;

    font-size: 1.2rem;

    font-weight: 700;

    color: var(--ink);

    margin-bottom: 0.25rem;
}

.card-subtitle {
    color: var(--muted);

    font-size: 0.84rem;
}


/* ---------------------------------------------------------
   FILE UPLOADER
--------------------------------------------------------- */

[data-testid="stFileUploader"] {
    background: #FFFEFA;

    border: 2px dashed #8BAA9F;

    border-radius: 20px;

    padding: 0.7rem;

    margin-top: 0.3rem;
}

[data-testid="stFileUploader"] section {
    background: transparent !important;
}


/* ---------------------------------------------------------
   BUTTONS
--------------------------------------------------------- */

.stButton > button,
.stDownloadButton > button {

    border: none !important;

    border-radius: 16px !important;

    background: var(--green) !important;

    color: white !important;

    font-weight: 700 !important;

    min-height: 46px;

    box-shadow:
        0 8px 18px rgba(57, 119, 109, .18);

    transition:
        transform .15s ease,
        background .15s ease;
}

.stButton > button:hover,
.stDownloadButton > button:hover {

    background: var(--green-dark) !important;

    transform: translateY(-1px);
}


/* ---------------------------------------------------------
   SLIDER
--------------------------------------------------------- */

div[data-baseweb="slider"] {
    padding-top: 0.7rem;
}

div[data-baseweb="slider"] > div > div {
    background: var(--green) !important;
}


/* ---------------------------------------------------------
   METRIC CARDS
--------------------------------------------------------- */

.metric-card {
    border-radius: 20px;

    padding: 1rem 1.05rem;

    min-height: 125px;

    border: 1px solid rgba(0,0,0,.035);
}

.metric-icon {
    width: 34px;

    height: 34px;

    border-radius: 11px;

    display: flex;

    align-items: center;

    justify-content: center;

    margin-bottom: 0.5rem;

    font-size: 1rem;
}

.metric-value {
    font-size: 1.55rem;

    font-weight: 700;

    color: var(--ink);

    line-height: 1.1;
}

.metric-label {
    color: #64716D;

    font-size: 0.8rem;

    margin-top: 0.3rem;
}

.metric-sage {
    background: #EAF3E6;
}

.metric-sage .metric-icon {
    background: #D8E8D2;
}

.metric-green {
    background: #E2F0E6;
}

.metric-green .metric-icon {
    background: #CFE5D4;
}

.metric-peach {
    background: #FCEBE2;
}

.metric-peach .metric-icon {
    background: #F8D8C8;
}

.metric-yellow {
    background: #FFF4D9;
}

.metric-yellow .metric-icon {
    background: #FBE6B0;
}


/* ---------------------------------------------------------
   IMAGE
--------------------------------------------------------- */

[data-testid="stImage"] img {
    border-radius: 18px;

    border: 1px solid #E7E3D8;
}


/* Image labels */

.image-label {
    text-align: center;

    font-weight: 700;

    color: var(--ink);

    background: #F0F5EB;

    border-radius: 999px;

    padding: 0.4rem 0.8rem;

    margin: 0.3rem auto 0.6rem auto;

    width: fit-content;

    font-size: 0.82rem;
}


/* ---------------------------------------------------------
   INFORMATION BOX
--------------------------------------------------------- */

.mini-note {
    background: #EEF5E9;

    border-radius: 14px;

    padding: 0.7rem 0.85rem;

    color: #46645D;

    font-size: 0.84rem;

    margin-top: 0.6rem;
}


/* ---------------------------------------------------------
   FOOTER
--------------------------------------------------------- */

.footer-note {
    text-align: center;

    color: #7A8179;

    font-size: 0.78rem;

    margin-top: 1.5rem;
}


/* ---------------------------------------------------------
   MOBILE
--------------------------------------------------------- */

@media (max-width: 900px) {

    .hero {
        padding: 2rem;
    }

    .hero-art {
        opacity: 0.20;

        right: -40px;
    }

}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PCA FUNCTIONS
# =========================================================

def compress_channel(channel, components):

    channel = channel.astype(float)

    # Mean of every column
    mean = channel.mean(axis=0)

    # Center the data
    centered = channel - mean

    # Make sure components are valid
    components = min(
        components,
        centered.shape[0],
        centered.shape[1]
    )

    # Apply PCA
    pca = PCA(
        n_components=components
    )

    transformed = pca.fit_transform(
        centered
    )

    # Reconstruct
    reconstructed = pca.inverse_transform(
        transformed
    )

    # Add mean back
    reconstructed = reconstructed + mean

    # Keep valid pixel range
    reconstructed = np.clip(
        reconstructed,
        0,
        255
    )

    # Calculate explained variance
    explained_variance = np.sum(
        pca.explained_variance_ratio_
    )

    return reconstructed, explained_variance


def compress_image(image_array, components):

    # RGB channels
    red = image_array[:, :, 0]
    green = image_array[:, :, 1]
    blue = image_array[:, :, 2]

    # Compress each channel
    red_result, red_variance = compress_channel(
        red,
        components
    )

    green_result, green_variance = compress_channel(
        green,
        components
    )

    blue_result, blue_variance = compress_channel(
        blue,
        components
    )

    # Combine channels
    reconstructed = np.stack(
        [
            red_result,
            green_result,
            blue_result
        ],
        axis=2
    )

    # Convert to image format
    reconstructed = np.round(
        reconstructed
    ).astype(np.uint8)

    # Average explained variance
    explained_variance = (
        red_variance
        + green_variance
        + blue_variance
    ) / 3

    return reconstructed, explained_variance


def prepare_image(uploaded_file):

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Keep processing fast
    max_size = 512

    if max(image.size) > max_size:

        scale = (
            max_size /
            max(image.size)
        )

        new_size = (
            int(image.size[0] * scale),
            int(image.size[1] * scale)
        )

        image = image.resize(
            new_size
        )

    return image


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">

            <div class="brand-icon">
                🌿
            </div>

            <div class="brand-title">
                PCA Studio
            </div>

            <div class="brand-subtitle">
                Image Compression
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📊 Analysis"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="sidebar-card">

            <div style="font-size:2rem;">
                🌱
            </div>

            <div style="
                color:#245B54;
                font-weight:700;
                font-size:1rem;
            ">
                Smaller Images
            </div>

            <div style="
                color:#63706B;
                font-size:.82rem;
                margin-top:.25rem;
            ">
                Same important information
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    # Top greeting
    st.markdown(
        """
        <div class="topbar">

            <span class="sun">
                ☀️
            </span>

            <span>
                Good Morning!
            </span>

        </div>
        """,
        unsafe_allow_html=True
    )


    # Hero
    st.markdown(
        """
        <div class="hero">

            <div class="hero-small">
                SMARTER · LIGHTER · CLEARER
            </div>

            <h1>
                PCA Image Compression
            </h1>

            <p>
                Reduce image dimensions while preserving
                important visual information using
                Principal Component Analysis.
            </p>


            <div class="hero-art">

                <div class="photo-card photo-one">
                    🏔️
                </div>

                <div class="arrow-art">
                    →
                </div>

                <div class="photo-card photo-two">
                    🖼️
                </div>

                <div class="pca-badge">
                    🧠 PCA
                </div>

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # Upload section
    st.markdown(
        """
        <div class="section-heading">
            Start Compressing
        </div>
        """,
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Upload your image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        label_visibility="collapsed"
    )


    # No image
    if uploaded_file is None:

        st.markdown(
            """
            <div class="mini-note">

                ☁️
                <b>Upload a JPG, JPEG or PNG image</b>
                to begin compression.

            </div>
            """,
            unsafe_allow_html=True
        )


    # =====================================================
    # IMAGE UPLOADED
    # =====================================================

    if uploaded_file is not None:

        image = prepare_image(
            uploaded_file
        )

        image_array = np.array(
            image
        )


        max_components = min(
            image_array.shape[0],
            image_array.shape[1]
        )


        default_components = min(
            50,
            max_components
        )


        # Two main cards
        left_col, right_col = st.columns(
            [1.25, 1],
            gap="large"
        )


        # -------------------------------------------------
        # LEFT CARD
        # -------------------------------------------------

        with left_col:

            st.markdown(
                """
                <div class="soft-card">

                    <div class="card-title">
                        🖼️ Your Image
                    </div>

                    <div class="card-subtitle">
                        Ready for PCA compression
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.image(
                image_array,
                width="stretch"
            )


            info1, info2, info3 = st.columns(3)


            with info1:

                st.metric(
                    "Width",
                    image.size[0]
                )


            with info2:

                st.metric(
                    "Height",
                    image.size[1]
                )


            with info3:

                st.metric(
                    "Channels",
                    3
                )


        # -------------------------------------------------
        # RIGHT CARD
        # -------------------------------------------------

        with right_col:

            st.markdown(
                """
                <div class="soft-card">

                    <div class="card-title">
                        🎛️ PCA Components
                    </div>

                    <div class="card-subtitle">
                        Choose how much information
                        to retain.
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


            components = st.slider(
                "Number of principal components",
                min_value=1,
                max_value=max_components,
                value=default_components,
                label_visibility="collapsed"
            )


            st.markdown(
                f"""
                <div class="mini-note">

                    🧠 Currently using
                    <b>{components}</b>
                    principal components.

                    <br>

                    Higher values generally preserve
                    more image details.

                </div>
                """,
                unsafe_allow_html=True
            )


            st.write("")


            compress_button = st.button(
                "Compress Image  →",
                use_container_width=True
            )


            if compress_button:

                with st.spinner(
                    "PCA is processing your image..."
                ):

                    compressed_image, explained_variance = (
                        compress_image(
                            image_array,
                            components
                        )
                    )


                # MSE
                mse = mean_squared_error(
                    image_array,
                    compressed_image
                )


                # PSNR
                psnr = peak_signal_noise_ratio(
                    image_array,
                    compressed_image,
                    data_range=255
                )


                # Save results
                st.session_state[
                    "compressed_image"
                ] = compressed_image

                st.session_state[
                    "original_image"
                ] = image_array

                st.session_state[
                    "components"
                ] = components

                st.session_state[
                    "variance"
                ] = explained_variance

                st.session_state[
                    "mse"
                ] = mse

                st.session_state[
                    "psnr"
                ] = psnr


                st.success(
                    "Compression completed successfully!"
                )


        # =================================================
        # RESULTS
        # =================================================

        if "compressed_image" in st.session_state:

            compressed_image = (
                st.session_state[
                    "compressed_image"
                ]
            )

            original_image = (
                st.session_state[
                    "original_image"
                ]
            )

            components = (
                st.session_state[
                    "components"
                ]
            )

            explained_variance = (
                st.session_state[
                    "variance"
                ]
            )

            mse = (
                st.session_state[
                    "mse"
                ]
            )

            psnr = (
                st.session_state[
                    "psnr"
                ]
            )


            st.markdown(
                """
                <div class="section-heading">
                    Compression Results
                </div>
                """,
                unsafe_allow_html=True
            )


            # -------------------------------------------------
            # METRICS
            # -------------------------------------------------

            m1, m2, m3, m4 = st.columns(
                4,
                gap="medium"
            )


            with m1:

                st.markdown(
                    f"""
                    <div class="metric-card metric-sage">

                        <div class="metric-icon">
                            ▱
                        </div>

                        <div class="metric-value">
                            {components}
                        </div>

                        <div class="metric-label">
                            Components
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with m2:

                st.markdown(
                    f"""
                    <div class="metric-card metric-green">

                        <div class="metric-icon">
                            ◔
                        </div>

                        <div class="metric-value">
                            {explained_variance * 100:.1f}%
                        </div>

                        <div class="metric-label">
                            Variance Retained
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with m3:

                st.markdown(
                    f"""
                    <div class="metric-card metric-peach">

                        <div class="metric-icon">
                            ⌁
                        </div>

                        <div class="metric-value">
                            {mse:.4f}
                        </div>

                        <div class="metric-label">
                            MSE
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            with m4:

                st.markdown(
                    f"""
                    <div class="metric-card metric-yellow">

                        <div class="metric-icon">
                            ★
                        </div>

                        <div class="metric-value">
                            {psnr:.2f}
                        </div>

                        <div class="metric-label">
                            PSNR (dB)
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # -------------------------------------------------
            # IMAGE COMPARISON
            # -------------------------------------------------

            st.markdown(
                """
                <div class="section-heading">
                    Image Comparison
                </div>
                """,
                unsafe_allow_html=True
            )


            comparison1, comparison2 = st.columns(
                2,
                gap="large"
            )


            with comparison1:

                st.markdown(
                    """
                    <div class="image-label">
                        Original Image
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.image(
                    original_image,
                    width="stretch"
                )


            with comparison2:

                st.markdown(
                    """
                    <div class="image-label"
                         style="background:#FCEBE2;">
                        Compressed Image (PCA)
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.image(
                    compressed_image,
                    width="stretch"
                )


            # -------------------------------------------------
            # DOWNLOAD
            # -------------------------------------------------

            output_image = Image.fromarray(
                compressed_image
            )


            buffer = io.BytesIO()


            output_image.save(
                buffer,
                format="PNG"
            )


            st.write("")


            st.download_button(
                "↓  Download Compressed Image",
                data=buffer.getvalue(),
                file_name="PCA_Compressed_Image.png",
                mime="image/png",
                use_container_width=True
            )


            # -------------------------------------------------
            # SMALL EXPLANATION
            # -------------------------------------------------

            st.markdown(
                f"""
                <div class="mini-note">

                    🌿
                    The image was reconstructed using
                    <b>{components}</b> principal components.

                    Approximately
                    <b>{explained_variance * 100:.2f}%</b>
                    of the image variance was retained.

                </div>
                """,
                unsafe_allow_html=True
            )


# =========================================================
# ANALYSIS PAGE
# =========================================================

elif page == "📊 Analysis":

    st.markdown(
        """
        <div class="section-heading"
             style="font-size:2rem;">

            PCA Analysis

        </div>

        <div style="
            color:#63706B;
            margin-bottom:1rem;
        ">

            Compare how different numbers of
            principal components affect image quality.

        </div>
        """,
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Upload image for analysis",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        image = prepare_image(
            uploaded_file
        )

        image_array = np.array(
            image
        )


        max_components = min(
            image_array.shape[0],
            image_array.shape[1]
        )


        possible_components = [
            5,
            10,
            20,
            50,
            100
        ]


        components_list = [
            x
            for x in possible_components
            if x <= max_components
        ]


        if not components_list:

            components_list = [
                max_components
            ]


        results = []


        progress = st.progress(0)


        for i, components in enumerate(
            components_list
        ):

            compressed, variance = (
                compress_image(
                    image_array,
                    components
                )
            )


            mse = mean_squared_error(
                image_array,
                compressed
            )


            psnr = peak_signal_noise_ratio(
                image_array,
                compressed,
                data_range=255
            )


            results.append(
                {
                    "Components": components,

                    "Explained Variance (%)":
                        variance * 100,

                    "MSE":
                        mse,

                    "PSNR (dB)":
                        psnr
                }
            )


            progress.progress(
                (i + 1) /
                len(components_list)
            )


        results_df = pd.DataFrame(
            results
        )


        st.markdown(
            """
            <div class="section-heading">
                Results
            </div>
            """,
            unsafe_allow_html=True
        )


        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )


        chart1, chart2 = st.columns(
            2,
            gap="large"
        )


        with chart1:

            st.markdown(
                "**Explained Variance (%)**"
            )

            st.line_chart(
                results_df.set_index(
                    "Components"
                )[
                    "Explained Variance (%)"
                ]
            )


        with chart2:

            st.markdown(
                "**PSNR (dB)**"
            )

            st.line_chart(
                results_df.set_index(
                    "Components"
                )[
                    "PSNR (dB)"
                ]
            )


        st.markdown(
            "**MSE**"
        )


        st.line_chart(
            results_df.set_index(
                "Components"
            )[
                "MSE"
            ]
        )


        csv_data = results_df.to_csv(
            index=False
        )


        st.download_button(
            "↓  Download Analysis CSV",
            data=csv_data,
            file_name="PCA_Analysis.csv",
            mime="text/csv",
            use_container_width=True
        )


    else:

        st.markdown(
            """
            <div class="mini-note">

                📊
                Upload an image to analyze how
                PCA components affect reconstruction.

            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer-note">

        PCA Studio · Image Compression using
        Principal Component Analysis

    </div>
    """,
    unsafe_allow_html=True
)