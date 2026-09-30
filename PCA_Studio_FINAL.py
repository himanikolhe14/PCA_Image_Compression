import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.decomposition import PCA
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio


# ============================================================
# PAGE SETUP
# ============================================================

st.set_page_config(
    page_title="PCA Studio",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DESIGN SYSTEM
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

:root {
    --cream: #FAF8F1;
    --cream2: #F2EFE5;
    --white: #FFFFFF;
    --ink: #15253B;
    --muted: #68736D;
    --green: #3B786E;
    --green-dark: #245A53;
    --sage: #DCE9D6;
    --sage-light: #EEF5E9;
    --peach: #F7D9C5;
    --peach-light: #FCEBE2;
    --yellow: #F7E8B4;
    --yellow-light: #FFF4D9;
    --lavender: #E9E1F3;
    --border: #E4E1D7;
}

html, body, [class*="css"] {
    font-family: "DM Sans", sans-serif;
}

.stApp {
    background: var(--cream);
    color: var(--ink);
}

.block-container {
    max-width: 1500px;
    padding: 1.2rem 2rem 3rem 2rem;
}

#MainMenu, footer {
    visibility: hidden;
}

header {
    background: transparent !important;
}

/* ---------------- SIDEBAR ---------------- */

section[data-testid="stSidebar"] {
    background: #F3F1E7;
    border-right: 1px solid #E3E0D6;
}

section[data-testid="stSidebar"] > div {
    padding: 1.4rem 1rem;
}

.sidebar-brand {
    text-align: center;
    padding: 0.4rem 0 1.2rem;
}

.sidebar-leaf {
    width: 62px;
    height: 62px;
    margin: 0 auto 0.7rem;
    border-radius: 20px;
    background: linear-gradient(145deg, #DCE9D6, #F7D9C5);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.1rem;
}

.sidebar-title {
    font-family: "Playfair Display", serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--ink);
}

.sidebar-subtitle {
    color: var(--green-dark);
    font-size: 0.85rem;
}

div[role="radiogroup"] {
    gap: 0.35rem;
}

div[role="radiogroup"] label {
    border-radius: 15px !important;
    padding: 0.68rem 0.75rem !important;
    border: 1px solid transparent !important;
}

div[role="radiogroup"] label:hover {
    background: #E6EDE0 !important;
}

.sidebar-bottom {
    margin-top: 1.2rem;
    padding: 1rem;
    border-radius: 20px;
    background: linear-gradient(145deg, #EAF2E6, #FFF0E6);
    text-align: center;
    color: var(--muted);
    font-size: 0.8rem;
}

/* ---------------- TOP BAR ---------------- */

.topbar {
    text-align: right;
    color: var(--ink);
    font-size: 0.92rem;
    margin-bottom: 0.65rem;
}

/* ---------------- HERO CONTAINER ---------------- */

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 25px !important;
    border-color: var(--border) !important;
    background: rgba(255,255,255,0.72) !important;
    box-shadow: 0 9px 25px rgba(60,70,55,.055);
}

/* Hero specifically: the first bordered container */
.hero-wrap {
    border-radius: 26px;
    padding: 0.1rem;
    background:
        radial-gradient(circle at 88% 18%, rgba(247,217,197,.75) 0 7%, transparent 7.5%),
        radial-gradient(circle at 67% 95%, rgba(220,233,214,.9) 0 18%, transparent 18.5%),
        linear-gradient(120deg,#FFF9E9,#F1F3E8 55%,#F9E5D6);
}

/* ---------------- TEXT ---------------- */

.hero-kicker {
    color: #68764E;
    letter-spacing: 0.18rem;
    font-size: 0.76rem;
    font-weight: 700;
}

.hero-title {
    font-family: "Playfair Display", serif;
    color: var(--ink);
    font-size: clamp(2.2rem, 4vw, 3.5rem);
    line-height: 1.03;
    margin: 0.25rem 0 0.7rem;
}

.hero-copy {
    color: #4F5D59;
    font-size: 1.02rem;
    line-height: 1.5;
    max-width: 620px;
}

.section-title {
    font-family: "Playfair Display", serif;
    color: var(--ink);
    font-size: 1.35rem;
    font-weight: 700;
    margin: 1.15rem 0 0.7rem;
}

/* ---------------- HERO ART ---------------- */

.art-box {
    height: 205px;
    position: relative;
    display: flex;
    align-items: center;
    justify-content: center;
}

.art-photo {
    position: absolute;
    width: 118px;
    height: 92px;
    border: 7px solid #FFFFFF;
    border-radius: 13px;
    background: linear-gradient(145deg,#84A58C,#D8E6D3);
    box-shadow: 0 13px 25px rgba(35,70,60,.15);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3rem;
}

.art-photo.one {
    left: 12%;
    top: 45px;
    transform: rotate(-7deg);
}

.art-photo.two {
    right: 9%;
    bottom: 20px;
    width: 92px;
    height: 72px;
    transform: rotate(7deg);
    font-size: 2.3rem;
}

.art-arrow {
    font-size: 2.8rem;
    color: var(--green-dark);
}

.art-pca {
    position: absolute;
    right: 23%;
    top: 6px;
    padding: .5rem .8rem;
    background: rgba(255,255,255,.9);
    border: 1px solid #DCE2D7;
    border-radius: 16px;
    color: var(--green-dark);
    font-weight: 700;
}

/* ---------------- UPLOAD ---------------- */

[data-testid="stFileUploader"] {
    background: #FFFEFA;
    border: 2px dashed #7FA298;
    border-radius: 20px;
    padding: .5rem;
}

[data-testid="stFileUploaderDropzoneInstructions"] {
    color: var(--ink);
}

/* ---------------- BUTTONS ---------------- */

.stButton > button,
.stDownloadButton > button {
    border: none !important;
    border-radius: 15px !important;
    background: var(--green) !important;
    color: white !important;
    font-weight: 700 !important;
    min-height: 45px;
    box-shadow: 0 7px 17px rgba(59,120,110,.17);
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: var(--green-dark) !important;
}

/* ---------------- SLIDER ---------------- */

div[data-baseweb="slider"] {
    padding-top: .3rem;
}

div[data-baseweb="slider"] > div > div {
    background: var(--green) !important;
}

/* ---------------- METRICS ---------------- */

.metric-card {
    border-radius: 19px;
    padding: 1rem;
    min-height: 118px;
}

.metric-card.sage { background: #EAF3E6; }
.metric-card.green { background: #E1F0E5; }
.metric-card.peach { background: #FCEBE2; }
.metric-card.yellow { background: #FFF3D8; }

.metric-icon {
    font-size: 1.15rem;
    margin-bottom: .45rem;
}

.metric-number {
    color: var(--ink);
    font-size: 1.48rem;
    font-weight: 700;
}

.metric-label {
    color: var(--muted);
    font-size: .78rem;
    margin-top: .2rem;
}

/* ---------------- IMAGE LABELS ---------------- */

.image-label {
    text-align: center;
    color: var(--ink);
    font-weight: 700;
    font-size: .82rem;
    padding: .38rem .8rem;
    border-radius: 999px;
    background: #EDF4E9;
    width: fit-content;
    margin: 0 auto .55rem;
}

.image-label.peach {
    background: #FCEBE2;
}

/* ---------------- INFO ---------------- */

.info-pill {
    background: var(--sage-light);
    color: #46635C;
    border-radius: 13px;
    padding: .65rem .8rem;
    font-size: .82rem;
    margin-top: .6rem;
}

.footer-note {
    text-align: center;
    color: #7D837C;
    font-size: .75rem;
    margin-top: 1.4rem;
}

@media (max-width: 900px) {
    .art-box {
        opacity: .35;
    }
}
</style>
""", unsafe_allow_html=True)


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
        centered.shape[1]
    )

    pca = PCA(n_components=components)

    transformed = pca.fit_transform(centered)

    reconstructed = pca.inverse_transform(transformed)

    reconstructed = reconstructed + mean
    reconstructed = np.clip(reconstructed, 0, 255)

    variance = float(np.sum(pca.explained_variance_ratio_))

    return reconstructed, variance


def compress_image(image_array, components):
    results = []
    variances = []

    for channel_index in range(3):
        reconstructed, variance = compress_channel(
            image_array[:, :, channel_index],
            components
        )
        results.append(reconstructed)
        variances.append(variance)

    reconstructed = np.stack(results, axis=2)

    reconstructed = np.round(
        reconstructed
    ).astype(np.uint8)

    return reconstructed, float(np.mean(variances))


def prepare_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")

    max_size = 512

    if max(image.size) > max_size:
        scale = max_size / max(image.size)

        image = image.resize(
            (
                int(image.size[0] * scale),
                int(image.size[1] * scale)
            )
        )

    return image


def calculate_metrics(original, compressed):
    mse = mean_squared_error(
        original,
        compressed
    )

    psnr = peak_signal_noise_ratio(
        original,
        compressed,
        data_range=255
    )

    return float(mse), float(psnr)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <div class="sidebar-leaf">🌿</div>
            <div class="sidebar-title">PCA Studio</div>
            <div class="sidebar-subtitle">Image Compression</div>
        </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Analysis"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("""
        <div class="sidebar-bottom">
            <div style="font-size:1.9rem;">🌱</div>
            <b style="color:#245A53;">Smaller Images</b><br>
            Preserve important visual information with PCA.
        </div>
    """, unsafe_allow_html=True)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="topbar">☀️ &nbsp; Good Morning!</div>',
        unsafe_allow_html=True
    )

    # HERO
    with st.container(border=True):
        hero_left, hero_right = st.columns([1.45, 1])

        with hero_left:
            st.markdown(
                '<div class="hero-kicker">SMARTER · LIGHTER · CLEARER</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="hero-title">PCA Image Compression</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="hero-copy">'
                'Reduce image dimensions while preserving important '
                'visual information using Principal Component Analysis.'
                '</div>',
                unsafe_allow_html=True
            )

            st.write("")

            st.button(
                "Start Compressing  →",
                use_container_width=False
            )

        with hero_right:
            st.markdown("""
                <div class="art-box">
                    <div class="art-photo one">🏔️</div>
                    <div class="art-arrow">→</div>
                    <div class="art-photo two">🖼️</div>
                    <div class="art-pca">🧠 PCA</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Start Compressing</div>',
        unsafe_allow_html=True
    )

    # MAIN WORK AREA
    upload_col, control_col, metric_col = st.columns(
        [1.2, 1.0, .78],
        gap="medium"
    )

    with upload_col:
        with st.container(border=True):
            st.markdown(
                '<div class="section-title" style="margin-top:0;">'
                '🖼️ Upload Your Image</div>',
                unsafe_allow_html=True
            )

            uploaded_file = st.file_uploader(
                "Choose an image",
                type=["jpg", "jpeg", "png"],
                label_visibility="visible"
            )

            if uploaded_file is None:
                st.caption("JPG · JPEG · PNG")

    with control_col:
        with st.container(border=True):
            st.markdown(
                '<div class="section-title" style="margin-top:0;">'
                '🎛️ PCA Components</div>',
                unsafe_allow_html=True
            )

            if uploaded_file is None:
                st.info("Upload an image to choose components.")
            else:
                image = prepare_image(uploaded_file)
                image_array = np.array(image)

                max_components = min(
                    image_array.shape[0],
                    image_array.shape[1]
                )

                components = st.slider(
                    "Components",
                    1,
                    max_components,
                    min(50, max_components)
                )

                st.markdown(
                    f'<div class="info-pill">'
                    f'ℹ️ Higher components retain more details.'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.write("")

                compress_clicked = st.button(
                    "Compress Image  →",
                    use_container_width=True
                )

                if compress_clicked:
                    with st.spinner("Applying PCA..."):
                        compressed, variance = compress_image(
                            image_array,
                            components
                        )

                    mse, psnr = calculate_metrics(
                        image_array,
                        compressed
                    )

                    st.session_state.original = image_array
                    st.session_state.compressed = compressed
                    st.session_state.components = components
                    st.session_state.variance = variance
                    st.session_state.mse = mse
                    st.session_state.psnr = psnr

                    st.success("Compression complete!")

    # METRICS
    with metric_col:
        if "compressed" in st.session_state:
            st.markdown(
                f"""
                <div class="metric-card sage">
                    <div class="metric-icon">▱</div>
                    <div class="metric-number">
                        {st.session_state.components}
                    </div>
                    <div class="metric-label">
                        Components
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            st.markdown(
                f"""
                <div class="metric-card green">
                    <div class="metric-icon">◔</div>
                    <div class="metric-number">
                        {st.session_state.variance * 100:.1f}%
                    </div>
                    <div class="metric-label">
                        Variance Retained
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.markdown(
                """
                <div class="metric-card sage">
                    <div class="metric-icon">▱</div>
                    <div class="metric-number">—</div>
                    <div class="metric-label">Components</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            st.markdown(
                """
                <div class="metric-card green">
                    <div class="metric-icon">◔</div>
                    <div class="metric-number">—</div>
                    <div class="metric-label">Variance Retained</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # RESULTS METRICS
    if "compressed" in st.session_state:

        st.write("")

        r1, r2, r3, r4 = st.columns(4, gap="medium")

        with r1:
            st.markdown(
                f"""
                <div class="metric-card sage">
                    <div class="metric-icon">▱</div>
                    <div class="metric-number">
                        {st.session_state.components}
                    </div>
                    <div class="metric-label">Components</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with r2:
            st.markdown(
                f"""
                <div class="metric-card green">
                    <div class="metric-icon">◔</div>
                    <div class="metric-number">
                        {st.session_state.variance * 100:.1f}%
                    </div>
                    <div class="metric-label">Variance Retained</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with r3:
            st.markdown(
                f"""
                <div class="metric-card peach">
                    <div class="metric-icon">⌁</div>
                    <div class="metric-number">
                        {st.session_state.mse:.4f}
                    </div>
                    <div class="metric-label">MSE</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with r4:
            st.markdown(
                f"""
                <div class="metric-card yellow">
                    <div class="metric-icon">★</div>
                    <div class="metric-number">
                        {st.session_state.psnr:.2f} dB
                    </div>
                    <div class="metric-label">PSNR</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="section-title">Image Comparison</div>',
            unsafe_allow_html=True
        )

        with st.container(border=True):
            c1, c2 = st.columns(2, gap="large")

            with c1:
                st.markdown(
                    '<div class="image-label">Original Image</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    st.session_state.original,
                    width="stretch"
                )

            with c2:
                st.markdown(
                    '<div class="image-label peach">'
                    'Compressed Image (PCA)'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    st.session_state.compressed,
                    width="stretch"
                )

            st.write("")

            output = Image.fromarray(
                st.session_state.compressed
            )

            buffer = io.BytesIO()

            output.save(
                buffer,
                format="PNG"
            )

            st.download_button(
                "↓  Download Compressed Image",
                data=buffer.getvalue(),
                file_name="PCA_Compressed_Image.png",
                mime="image/png",
                use_container_width=True
            )

    else:
        st.markdown(
            '<div class="info-pill">'
            '☁️ Upload an image to begin. '
            'Your image will be processed using PCA.'
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# ANALYSIS
# ============================================================

else:

    st.markdown(
        '<div class="section-title" style="font-size:2rem;">'
        'PCA Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="color:#68736D;">'
        'See how the number of components changes reconstruction quality.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    uploaded_file = st.file_uploader(
        "Upload image for analysis",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = prepare_image(uploaded_file)
        image_array = np.array(image)

        max_components = min(
            image_array.shape[0],
            image_array.shape[1]
        )

        component_options = [
            x for x in [5, 10, 20, 50, 100]
            if x <= max_components
        ]

        if not component_options:
            component_options = [max_components]

        results = []

        progress = st.progress(0)

        for index, components in enumerate(
            component_options
        ):

            compressed, variance = compress_image(
                image_array,
                components
            )

            mse, psnr = calculate_metrics(
                image_array,
                compressed
            )

            results.append({
                "Components": components,
                "Explained Variance (%)": variance * 100,
                "MSE": mse,
                "PSNR (dB)": psnr
            })

            progress.progress(
                (index + 1) / len(component_options)
            )

        df = pd.DataFrame(results)

        st.write("")

        with st.container(border=True):
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

        st.write("")

        chart1, chart2 = st.columns(2)

        with chart1:
            st.markdown("**Explained Variance**")
            st.line_chart(
                df.set_index("Components")[
                    "Explained Variance (%)"
                ]
            )

        with chart2:
            st.markdown("**PSNR**")
            st.line_chart(
                df.set_index("Components")[
                    "PSNR (dB)"
                ]
            )

        st.markdown("**MSE**")

        st.line_chart(
            df.set_index("Components")["MSE"]
        )

        csv_data = df.to_csv(index=False)

        st.download_button(
            "↓  Download Analysis CSV",
            data=csv_data,
            file_name="PCA_Analysis.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:
        st.markdown(
            '<div class="info-pill">'
            '📊 Upload an image to generate the PCA analysis.'
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer-note">'
    'PCA Studio · Image Compression using Principal Component Analysis'
    '</div>',
    unsafe_allow_html=True
)
