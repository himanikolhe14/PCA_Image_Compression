import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
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
# CREATE ILLUSTRATIONS INSIDE THE APP
# ============================================================

def make_landscape(size=(700, 430)):
    """Create the colorful mountain/lake artwork used in the dashboard."""
    w, h = size
    img = Image.new("RGB", size, "#A9DCE8")
    d = ImageDraw.Draw(img)

    # sky
    d.rectangle((0, 0, w, int(h * .52)), fill="#A9DCE8")

    # soft clouds
    for x, y, r in [(90, 80, 28), (125, 72, 38), (165, 84, 25),
                    (520, 92, 30), (560, 82, 40), (605, 96, 24)]:
        d.ellipse((x-r, y-r, x+r, y+r), fill="#F8FBF5")

    # distant mountains
    d.polygon(
        [(0, 260), (120, 130), (205, 245), (330, 80),
         (455, 245), (570, 145), (700, 270)],
        fill="#658C83"
    )

    # snow caps
    d.polygon(
        [(330, 80), (285, 150), (316, 137), (330, 158),
         (350, 128), (375, 151)],
        fill="#F8F4E8"
    )

    d.polygon(
        [(120, 130), (92, 173), (113, 162), (126, 177), (146, 156)],
        fill="#F7F2E6"
    )

    # foreground
    d.polygon(
        [(0, 245), (90, 215), (160, 245), (245, 210),
         (340, 250), (430, 215), (540, 250), (630, 220), (700, 250),
         (700, 315), (0, 315)],
        fill="#477A64"
    )

    # lake
    d.rectangle((0, 300, w, h), fill="#68AEB2")

    # lake highlights
    for y in range(325, h, 22):
        d.rounded_rectangle(
            (60, y, 250, y + 5),
            radius=3,
            fill="#A9D7D4"
        )
        d.rounded_rectangle(
            (410, y + 7, 660, y + 12),
            radius=3,
            fill="#A9D7D4"
        )

    # trees
    for x in [35, 70, 610, 650]:
        d.polygon(
            [(x, 270), (x+28, 165), (x+56, 270)],
            fill="#285B4D"
        )
        d.rectangle(
            (x+24, 260, x+32, 310),
            fill="#5C5941"
        )

    return img


def make_leaf_art(size=(420, 470)):
    """Create a soft botanical illustration for the sidebar."""
    w, h = size
    img = Image.new("RGB", size, "#EAF1E4")
    d = ImageDraw.Draw(img)

    # soft circles
    d.ellipse((-100, 250, 210, 560), fill="#D6E4CF")
    d.ellipse((210, 300, 510, 610), fill="#F3D7C3")
    d.ellipse((250, -100, 500, 180), fill="#DCE8D5")

    # stems
    d.line((210, 420, 170, 120), fill="#527E65", width=7)
    d.line((210, 420, 270, 160), fill="#527E65", width=6)
    d.line((210, 420, 95, 235), fill="#527E65", width=5)

    # leaves
    leaves = [
        (150, 150, 95, 55, "#4E866B"),
        (230, 170, 105, 58, "#8EAC76"),
        (105, 245, 90, 50, "#719B73"),
        (260, 260, 105, 58, "#4A7A63"),
        (175, 90, 80, 45, "#C8B970"),
    ]

    for x, y, ew, eh, color in leaves:
        d.ellipse(
            (x-ew//2, y-eh//2, x+ew//2, y+eh//2),
            fill=color
        )

    return img


# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

/* ===== GLOBAL BACKGROUND ===== */

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
.main,
.main .block-container {
    background: #F7F5EC !important;
}

.stApp {
    color: #15263D !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 8% 5%, rgba(217,234,215,.55) 0 12%, transparent 25%),
        radial-gradient(circle at 96% 80%, rgba(248,216,198,.38) 0 10%, transparent 25%),
        #F7F5EC !important;
}

.block-container {
    max-width: 1500px !important;
    padding: 1rem 2rem 3rem !important;
}

#MainMenu, footer {
    visibility: hidden !important;
}

header {
    background: transparent !important;
}

/* ===== SIDEBAR ===== */

section[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 0 0, #DCEBDA 0 11%, transparent 22%),
        radial-gradient(circle at 100% 92%, #F2D9C6 0 15%, transparent 28%),
        #EFF0E7 !important;
    border-right: 1px solid #DDDCD2 !important;
}

section[data-testid="stSidebar"] > div {
    padding: 1rem .9rem !important;
}

.brand-title {
    text-align: center;
    font-family: "Playfair Display", serif;
    font-size: 1.65rem;
    font-weight: 700;
    color: #14243A;
}

.brand-sub {
    text-align: center;
    color: #52685F;
    font-size: .82rem;
    margin-bottom: 1rem;
}

.brand-leaves {
    text-align: center;
    font-size: 3rem;
    line-height: 1.1;
    margin-top: .1rem;
}

div[role="radiogroup"] {
    gap: .35rem;
}

div[role="radiogroup"] label {
    border-radius: 17px !important;
    padding: .7rem .75rem !important;
    border: 1px solid transparent !important;
}

div[role="radiogroup"] label:hover {
    background: #E1ECDE !important;
}

.side-art {
    margin-top: 1rem;
    border-radius: 24px;
    overflow: hidden;
    border: 1px solid #E0DED4;
    box-shadow: 0 10px 24px rgba(60,70,50,.06);
}

.side-caption {
    text-align: center;
    font-family: "Playfair Display", serif;
    color: #31554B;
    font-size: 1.05rem;
    padding: .7rem .4rem 1rem;
}

/* ===== TOP BAR ===== */

.topbar {
    height: 42px;
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: .5rem;
    color: #14243A;
    font-size: .9rem;
}

/* ===== HERO ===== */

.hero-shell {
    border: 1px solid #E2DFD5;
    border-radius: 30px;
    overflow: hidden;
    background:
        radial-gradient(circle at 78% 0%, #F5D5C0 0 10%, transparent 10.5%),
        radial-gradient(circle at 47% 100%, #D9E9D5 0 20%, transparent 20.5%),
        radial-gradient(circle at 98% 52%, #E2ECD9 0 15%, transparent 15.5%),
        linear-gradient(120deg, #FFF9EA, #EFF3E8 55%, #FBE6D8);
    box-shadow: 0 15px 38px rgba(54,65,52,.08);
}

.hero-title {
    font-family: "Playfair Display", serif;
    font-size: clamp(2.3rem, 4.1vw, 4.05rem);
    line-height: 1.02;
    font-weight: 700;
    color: #13243A;
    margin-top: .35rem;
}

.hero-kicker {
    color: #657448;
    font-size: .73rem;
    font-weight: 700;
    letter-spacing: .22rem;
}

.hero-copy {
    color: #4C5C57;
    font-size: 1.02rem;
    line-height: 1.55;
    max-width: 610px;
}

.hero-note {
    background: rgba(255,255,255,.78);
    border: 1px solid #DCE3D8;
    border-radius: 18px;
    padding: .5rem .8rem;
    color: #2F6659;
    font-weight: 700;
    display: inline-block;
}

/* ===== CARDS ===== */

[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255,255,255,.84) !important;
    border: 1px solid #E3E0D6 !important;
    border-radius: 23px !important;
    box-shadow: 0 10px 28px rgba(59,67,53,.055) !important;
}

/* ===== TITLES ===== */

.section-title {
    font-family: "Playfair Display", serif;
    font-weight: 700;
    color: #14243A;
    font-size: 1.45rem;
    margin: 1rem 0 .65rem;
}

.card-title {
    font-family: "Playfair Display", serif;
    font-size: 1.18rem;
    font-weight: 700;
    color: #14243A;
}

/* ===== UPLOADER ===== */

[data-testid="stFileUploader"] {
    background: #FFFEFA !important;
    border: 2px dashed #6E998C !important;
    border-radius: 20px !important;
    padding: .4rem !important;
}

/* ===== BUTTONS ===== */

.stButton > button,
.stDownloadButton > button {
    background: #397B6D !important;
    color: white !important;
    border: 0 !important;
    border-radius: 16px !important;
    min-height: 46px !important;
    font-weight: 700 !important;
    box-shadow: 0 9px 19px rgba(48,111,96,.18) !important;
    transition: transform .18s ease, box-shadow .18s ease !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #285F55 !important;
    transform: translateY(-2px);
    box-shadow: 0 13px 24px rgba(48,111,96,.25) !important;
}

/* ===== INFO ===== */

.info-box {
    background: linear-gradient(100deg,#ECF4E8,#F8F0E5);
    color: #4B675F;
    border-radius: 14px;
    padding: .65rem .8rem;
    font-size: .82rem;
    margin: .5rem 0;
}

/* ===== METRICS ===== */

.metric {
    border-radius: 20px;
    padding: 1rem;
    min-height: 125px;
}

.metric-green { background:#E4F0E4; }
.metric-mint { background:#E0F0E8; }
.metric-peach { background:#FBE7DE; }
.metric-yellow { background:#FFF0CC; }

.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color:#14243A;
}

.metric-label {
    color:#66736D;
    font-size:.78rem;
    margin-top:.2rem;
}

/* ===== IMAGES ===== */

[data-testid="stImage"] img {
    border-radius: 18px !important;
}

/* ===== SLIDER ===== */

div[data-baseweb="slider"] > div > div {
    background: #397B6D !important;
}

/* ===== FOOTER ===== */

.footer {
    text-align:center;
    color:#7B817A;
    font-size:.75rem;
    margin-top:1.5rem;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# PCA
# ============================================================

def prepare_image(file):
    image = Image.open(file).convert("RGB")

    max_size = 512

    if max(image.size) > max_size:
        scale = max_size / max(image.size)
        image = image.resize((
            int(image.size[0] * scale),
            int(image.size[1] * scale)
        ))

    return image


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
    reconstructed = pca.inverse_transform(transformed) + mean

    reconstructed = np.clip(reconstructed, 0, 255)

    return (
        reconstructed,
        float(np.sum(pca.explained_variance_ratio_))
    )


def compress_image(arr, components):
    output = []
    variance = []

    for channel in range(3):
        result, var = compress_channel(
            arr[:, :, channel],
            components
        )
        output.append(result)
        variance.append(var)

    result = np.stack(output, axis=2)

    return (
        np.round(result).astype(np.uint8),
        float(np.mean(variance))
    )


def get_metrics(original, compressed):
    mse = float(
        mean_squared_error(original, compressed)
    )

    psnr = float(
        peak_signal_noise_ratio(
            original,
            compressed,
            data_range=255
        )
    )

    return mse, psnr


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        '<div class="brand-leaves">🌿</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-title">PCA Studio</div>'
        '<div class="brand-sub">Image Compression</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Analysis"],
        label_visibility="collapsed"
    )

    st.divider()

    # Actual illustration displayed by Streamlit — NOT HTML.
    st.image(
        make_leaf_art(),
        use_container_width=True
    )

    st.markdown(
        '<div class="side-caption">'
        'Smaller Images<br>'
        '<span style="font-family:DM Sans;font-size:.78rem;'
        'color:#63736B;">Brighter Possibilities</span>'
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.markdown(
        '<div class="topbar">'
        '☀️ &nbsp; Good Morning! &nbsp; 👤'
        '</div>',
        unsafe_allow_html=True
    )

    # HERO
    with st.container(border=True):

        left, right = st.columns(
            [1.25, 1],
            gap="medium"
        )

        with left:

            st.markdown(
                '<div class="hero-shell" style="padding:2rem;">'
                '<div class="hero-kicker">'
                'SMARTER · LIGHTER · CLEARER'
                '</div>'
                '<div class="hero-title">'
                'PCA Image Compression'
                '</div>'
                '<div class="hero-copy">'
                'Reduce image size while preserving important '
                'visual information using Principal Component Analysis.'
                '</div>'
                '</div>',
                unsafe_allow_html=True
            )

            st.write("")

            st.button(
                "Start Compressing  →",
                key="hero_button"
            )

        with right:

            # IMPORTANT: actual image, not HTML artwork
            hero_img = make_landscape((720, 430))

            st.image(
                hero_img,
                use_container_width=True
            )

            st.markdown(
                '<div class="hero-note">'
                '🧠 PCA &nbsp; → &nbsp; Same beauty · smaller size'
                '</div>',
                unsafe_allow_html=True
            )

    st.markdown(
        '<div class="section-title">Start Compressing</div>',
        unsafe_allow_html=True
    )

    upload_col, controls_col, metrics_col = st.columns(
        [1.18, 1.0, .72],
        gap="medium"
    )

    # UPLOAD
    with upload_col:

        with st.container(border=True):

            st.markdown(
                '<div class="card-title">'
                '☁️ Upload Your Image'
                '</div>',
                unsafe_allow_html=True
            )

            st.caption(
                "JPG · JPEG · PNG"
            )

            uploaded = st.file_uploader(
                "Choose image",
                type=["jpg", "jpeg", "png"],
                label_visibility="visible"
            )

            if uploaded is None:
                st.markdown(
                    '<div class="info-box">'
                    '🖼️ Drag & drop your image here or click Upload'
                    '</div>',
                    unsafe_allow_html=True
                )

    # CONTROLS
    with controls_col:

        with st.container(border=True):

            st.markdown(
                '<div class="card-title">'
                '🎛️ PCA Components'
                '</div>',
                unsafe_allow_html=True
            )

            if uploaded is None:

                st.markdown(
                    '<div class="info-box">'
                    'Upload an image to choose the number of components.'
                    '</div>',
                    unsafe_allow_html=True
                )

            else:

                image = prepare_image(uploaded)
                arr = np.array(image)

                max_components = min(
                    arr.shape[0],
                    arr.shape[1]
                )

                components = st.slider(
                    "Components",
                    1,
                    max_components,
                    min(50, max_components)
                )

                st.markdown(
                    '<div class="info-box">'
                    '💡 Higher components retain more image details.'
                    '</div>',
                    unsafe_allow_html=True
                )

                if st.button(
                    "Compress Image  →",
                    key="compress",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Applying PCA..."
                    ):

                        compressed, variance = compress_image(
                            arr,
                            components
                        )

                    mse, psnr = get_metrics(
                        arr,
                        compressed
                    )

                    st.session_state.result = {
                        "original": arr,
                        "compressed": compressed,
                        "components": components,
                        "variance": variance,
                        "mse": mse,
                        "psnr": psnr
                    }

                    st.success(
                        "Compression complete!"
                    )

    # METRICS
    with metrics_col:

        if "result" in st.session_state:

            r = st.session_state.result

            st.markdown(
                f'<div class="metric metric-green">'
                f'<div class="metric-value">{r["components"]}</div>'
                f'<div class="metric-label">Components</div>'
                f'</div>',
                unsafe_allow_html=True
            )

            st.write("")

            st.markdown(
                f'<div class="metric metric-mint">'
                f'<div class="metric-value">'
                f'{r["variance"]*100:.1f}%'
                f'</div>'
                f'<div class="metric-label">Variance Retained</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                '<div class="metric metric-green">'
                '<div class="metric-value">50</div>'
                '<div class="metric-label">Components</div>'
                '</div>',
                unsafe_allow_html=True
            )

            st.write("")

            st.markdown(
                '<div class="metric metric-mint">'
                '<div class="metric-value">—</div>'
                '<div class="metric-label">Variance Retained</div>'
                '</div>',
                unsafe_allow_html=True
            )

    # RESULTS
    if "result" in st.session_state:

        r = st.session_state.result

        st.write("")

        c1, c2, c3, c4 = st.columns(4)

        cards = [
            (c1, "metric-green", str(r["components"]), "Components"),
            (c2, "metric-mint", f'{r["variance"]*100:.1f}%', "Variance Retained"),
            (c3, "metric-peach", f'{r["mse"]:.4f}', "MSE"),
            (c4, "metric-yellow", f'{r["psnr"]:.2f} dB', "PSNR"),
        ]

        for col, color, value, label in cards:
            with col:
                st.markdown(
                    f'<div class="metric {color}">'
                    f'<div class="metric-value">{value}</div>'
                    f'<div class="metric-label">{label}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        st.markdown(
            '<div class="section-title">Image Comparison</div>',
            unsafe_allow_html=True
        )

        with st.container(border=True):

            a, b = st.columns(2)

            with a:

                st.markdown(
                    '<div class="image-pill">Original Image</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    r["original"],
                    use_container_width=True
                )

            with b:

                st.markdown(
                    '<div class="image-pill" '
                    'style="background:#FBE7DE;">'
                    'Compressed Image (PCA)'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    r["compressed"],
                    use_container_width=True
                )

            output = Image.fromarray(
                r["compressed"]
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
            '<div class="info-box">'
            '🌱 Upload an image above to activate the PCA compressor.'
            '</div>',
            unsafe_allow_html=True
        )


# ============================================================
# ANALYSIS
# ============================================================

else:

    st.markdown(
        '<div class="section-title" style="font-size:2rem;">'
        'PCA Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="color:#69746E;">'
        'Compare how different component counts affect image quality.'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("")

    uploaded = st.file_uploader(
        "Upload image for analysis",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        image = prepare_image(uploaded)
        arr = np.array(image)

        max_components = min(
            arr.shape[0],
            arr.shape[1]
        )

        component_list = [
            x for x in [5, 10, 20, 50, 100]
            if x <= max_components
        ]

        if not component_list:
            component_list = [max_components]

        rows = []
        progress = st.progress(0)

        for i, components in enumerate(component_list):

            compressed, variance = compress_image(
                arr,
                components
            )

            mse, psnr = get_metrics(
                arr,
                compressed
            )

            rows.append({
                "Components": components,
                "Explained Variance (%)": variance * 100,
                "MSE": mse,
                "PSNR (dB)": psnr
            })

            progress.progress(
                (i + 1) / len(component_list)
            )

        df = pd.DataFrame(rows)

        with st.container(border=True):
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

        x, y = st.columns(2)

        with x:
            st.markdown("**Explained Variance**")
            st.line_chart(
                df.set_index("Components")[
                    "Explained Variance (%)"
                ]
            )

        with y:
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

        st.download_button(
            "↓ Download Analysis CSV",
            data=df.to_csv(index=False),
            file_name="PCA_Analysis.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:

        st.markdown(
            '<div class="info-box">'
            '📊 Upload an image to generate the analysis.'
            '</div>',
            unsafe_allow_html=True
        )


st.markdown(
    '<div class="footer">'
    'PCA Studio · Image Compression using Principal Component Analysis'
    '</div>',
    unsafe_allow_html=True
)
