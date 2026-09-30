import io
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.decomposition import PCA
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio


st.set_page_config(
    page_title="PCA Studio",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# COMPLETE VISUAL DESIGN
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

:root{
    --bg:#F8F6EE;
    --paper:#FFFDF8;
    --ink:#14243A;
    --muted:#69746E;
    --green:#397B6D;
    --green2:#245B52;
    --mint:#DCEBDD;
    --mint2:#EFF6EB;
    --peach:#F6D5C0;
    --peach2:#FCE9DF;
    --yellow:#F6E5A9;
    --yellow2:#FFF3D1;
    --purple:#E9DDF4;
    --purple2:#F4ECF9;
    --line:#E6E2D8;
}

*{box-sizing:border-box}

html,body,[class*="css"]{
    font-family:"DM Sans",sans-serif;
}

.stApp{
    background:
        radial-gradient(circle at 7% 8%,rgba(220,235,221,.45),transparent 19%),
        radial-gradient(circle at 94% 75%,rgba(246,213,192,.35),transparent 20%),
        var(--bg);
    color:var(--ink);
}

.block-container{
    max-width:1540px;
    padding:1.15rem 1.8rem 3rem;
}

#MainMenu,footer{visibility:hidden}
header{background:transparent!important}

/* ---------------- SIDEBAR ---------------- */

section[data-testid="stSidebar"]{
    background:
        radial-gradient(circle at 10% 12%,#DDEBDD 0 7%,transparent 7.5%),
        radial-gradient(circle at 80% 88%,#F4DDCB 0 12%,transparent 12.5%),
        #F2F1E8;
    border-right:1px solid #E1DED4;
}

section[data-testid="stSidebar"]>div{
    padding:1rem .95rem;
}

.brand{
    text-align:center;
    padding:.2rem 0 1.1rem;
}

.brand-art{
    height:90px;
    position:relative;
    display:flex;
    justify-content:center;
    align-items:center;
}

.brand-art .leaf{
    position:absolute;
    width:39px;
    height:62px;
    border-radius:70% 10% 70% 10%;
    transform:rotate(-25deg);
    background:linear-gradient(145deg,#72A888,#2D6B5D);
}

.brand-art .leaf.two{
    transform:rotate(35deg) scale(.86);
    margin-left:42px;
    margin-top:-9px;
    background:linear-gradient(145deg,#B8C98B,#5B8C70);
}

.brand-art .leaf.three{
    transform:rotate(5deg) scale(.7);
    margin-left:-42px;
    margin-top:10px;
    background:#D7C879;
}

.brand-art .stem{
    width:4px;
    height:56px;
    background:#376B5C;
    position:absolute;
    top:30px;
    transform:rotate(-8deg);
    border-radius:5px;
}

.brand-name{
    font-family:"Playfair Display",serif;
    font-size:1.65rem;
    font-weight:700;
    color:var(--ink);
}

.brand-sub{
    color:#526A62;
    font-size:.82rem;
}

div[role="radiogroup"]{
    gap:.35rem;
}

div[role="radiogroup"] label{
    border-radius:16px!important;
    padding:.72rem .8rem!important;
    border:1px solid transparent!important;
    transition:.18s ease;
}

div[role="radiogroup"] label:hover{
    background:#E1ECDF!important;
}

.side-poster{
    margin-top:1.2rem;
    padding:1rem .7rem;
    min-height:270px;
    border-radius:24px;
    background:linear-gradient(155deg,#E5F0E1,#FFF0E4);
    position:relative;
    overflow:hidden;
    text-align:center;
}

.side-photo{
    width:105px;
    height:82px;
    border:7px solid white;
    border-radius:11px;
    background:
        linear-gradient(160deg,#9ED2E2 0 42%,#6F9E72 43% 65%,#BBD7E0 66%);
    box-shadow:0 10px 22px rgba(42,74,61,.18);
    transform:rotate(-8deg);
    margin:1rem auto .8rem;
    position:relative;
}

.side-photo:before{
    content:"";
    position:absolute;
    left:18px;
    top:17px;
    width:0;
    height:0;
    border-left:30px solid transparent;
    border-right:30px solid transparent;
    border-bottom:34px solid #5B7F70;
}

.side-leaf{
    position:absolute;
    width:55px;
    height:90px;
    right:-12px;
    bottom:-20px;
    border-radius:80% 5% 80% 5%;
    background:#C9DDBD;
    transform:rotate(27deg);
}

.side-copy{
    font-family:"Playfair Display",serif;
    font-size:1.08rem;
    line-height:1.25;
    color:#28483F;
}

/* ---------------- TOP ---------------- */

.top{
    display:flex;
    justify-content:flex-end;
    align-items:center;
    gap:.65rem;
    color:var(--ink);
    font-size:.9rem;
    margin-bottom:.6rem;
}

.avatar{
    width:37px;
    height:37px;
    border-radius:50%;
    background:#E7E4DA;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    font-size:1.1rem;
}

/* ---------------- HERO ---------------- */

.hero{
    min-height:300px;
    border:1px solid #E4E0D5;
    border-radius:31px;
    padding:2.2rem 2.7rem;
    background:
        radial-gradient(circle at 79% 0%,rgba(246,213,192,.85) 0 9%,transparent 9.5%),
        radial-gradient(circle at 54% 100%,rgba(204,226,201,.9) 0 17%,transparent 17.5%),
        radial-gradient(circle at 96% 50%,rgba(220,235,214,.7) 0 12%,transparent 12.5%),
        linear-gradient(118deg,#FFF9EA,#F1F4E8 57%,#FBE7D9);
    box-shadow:0 16px 38px rgba(59,72,55,.07);
    overflow:hidden;
    position:relative;
}

.hero-kicker{
    letter-spacing:.22rem;
    font-size:.73rem;
    font-weight:700;
    color:#6B774E;
}

.hero-title{
    font-family:"Playfair Display",serif;
    font-size:clamp(2.5rem,4.5vw,4.3rem);
    line-height:1;
    color:var(--ink);
    margin:.5rem 0 .85rem;
}

.hero-copy{
    max-width:620px;
    font-size:1.05rem;
    line-height:1.55;
    color:#4E5B57;
}

.hero-art{
    height:245px;
    position:relative;
}

.float{
    animation:float 3.6s ease-in-out infinite;
}

.float2{
    animation:float2 4.2s ease-in-out infinite;
}

@keyframes float{
    0%,100%{transform:translateY(0) rotate(-7deg)}
    50%{transform:translateY(-9px) rotate(-4deg)}
}

@keyframes float2{
    0%,100%{transform:translateY(0) rotate(7deg)}
    50%{transform:translateY(8px) rotate(11deg)}
}

.hero-card{
    position:absolute;
    width:155px;
    height:118px;
    border:8px solid white;
    border-radius:14px;
    box-shadow:0 17px 28px rgba(40,69,58,.18);
    overflow:hidden;
}

.hero-card.big{
    left:7%;
    top:58px;
}

.hero-card.small{
    right:6%;
    bottom:22px;
    width:106px;
    height:82px;
}

.landscape{
    width:100%;
    height:100%;
    position:relative;
    background:linear-gradient(#9ED6E7 0 48%,#75A777 49% 65%,#79B5B9 66%);
}

.landscape:before{
    content:"";
    position:absolute;
    left:-12px;
    bottom:34px;
    width:125px;
    height:76px;
    background:#587F6D;
    clip-path:polygon(0 100%,38% 12%,56% 58%,78% 20%,100% 100%);
}

.landscape:after{
    content:"";
    position:absolute;
    right:7px;
    top:8px;
    width:28px;
    height:28px;
    border-radius:50%;
    background:#F6D875;
}

.hero-arrow{
    position:absolute;
    left:49%;
    top:104px;
    color:var(--green2);
    font-size:3rem;
    animation:pulseArrow 2s ease-in-out infinite;
}

@keyframes pulseArrow{
    0%,100%{transform:translateX(0)}
    50%{transform:translateX(7px)}
}

.pca-tag{
    position:absolute;
    right:19%;
    top:20px;
    padding:.58rem .9rem;
    border-radius:18px;
    background:rgba(255,255,255,.9);
    border:1px solid #DDE4D9;
    color:#286255;
    font-weight:700;
}

.hero-note{
    position:absolute;
    right:2%;
    top:105px;
    font-family:"Playfair Display",serif;
    font-size:1.15rem;
    color:#4E685D;
    transform:rotate(-4deg);
}

/* ---------------- BUTTON ---------------- */

.stButton>button,
.stDownloadButton>button{
    border:0!important;
    border-radius:16px!important;
    background:linear-gradient(135deg,#3E8274,#286357)!important;
    color:white!important;
    font-weight:700!important;
    min-height:47px;
    box-shadow:0 9px 20px rgba(44,105,91,.18);
    transition:.2s ease!important;
}

.stButton>button:hover,
.stDownloadButton>button:hover{
    transform:translateY(-2px);
    box-shadow:0 13px 24px rgba(44,105,91,.25);
}

/* ---------------- CARDS ---------------- */

.card-title{
    font-family:"Playfair Display",serif;
    font-size:1.15rem;
    font-weight:700;
    color:var(--ink);
}

.section-title{
    font-family:"Playfair Display",serif;
    font-size:1.45rem;
    font-weight:700;
    color:var(--ink);
    margin:1rem 0 .7rem;
}

/* Make bordered Streamlit containers look like cards */
div[data-testid="stVerticalBlockBorderWrapper"]{
    border-radius:23px!important;
    border-color:var(--line)!important;
    background:rgba(255,255,255,.78)!important;
    box-shadow:0 9px 25px rgba(62,72,55,.055);
}

/* ---------------- UPLOAD ---------------- */

[data-testid="stFileUploader"]{
    background:#FFFDF8;
    border:2px dashed #789D91;
    border-radius:19px;
    padding:.55rem;
}

[data-testid="stFileUploaderDropzone"]{
    background:transparent!important;
}

/* ---------------- INFO ---------------- */

.info{
    background:linear-gradient(100deg,#EDF5EA,#F6F0E4);
    border-radius:14px;
    padding:.7rem .85rem;
    color:#49665E;
    font-size:.82rem;
    margin:.45rem 0;
}

/* ---------------- METRICS ---------------- */

.metric{
    border-radius:21px;
    padding:1.05rem;
    min-height:128px;
    border:1px solid rgba(0,0,0,.025);
}

.metric.mint{background:#E8F2E5}
.metric.green{background:#E0F0E5}
.metric.peach{background:#FBE9E0}
.metric.yellow{background:#FFF2D3}

.metric-icon{
    width:38px;
    height:38px;
    border-radius:12px;
    display:flex;
    align-items:center;
    justify-content:center;
    margin-bottom:.55rem;
    font-size:1.15rem;
}

.mint .metric-icon{background:#D3E6CC}
.green .metric-icon{background:#CCE4D1}
.peach .metric-icon{background:#F6D3C4}
.yellow .metric-icon{background:#F7E3AD}

.metric-value{
    font-size:1.55rem;
    font-weight:700;
    color:var(--ink);
}

.metric-label{
    color:#66726D;
    font-size:.78rem;
    margin-top:.18rem;
}

/* ---------------- IMAGE COMPARISON ---------------- */

[data-testid="stImage"] img{
    border-radius:18px;
    border:1px solid #E7E3D8;
}

.image-pill{
    text-align:center;
    margin:0 auto .55rem;
    width:fit-content;
    padding:.42rem .9rem;
    border-radius:999px;
    background:#EAF3E7;
    color:#26483F;
    font-weight:700;
    font-size:.82rem;
}

.image-pill.peach{background:#FCE8DE}

/* ---------------- RESPONSIVE ---------------- */

@media(max-width:900px){
    .hero-art{opacity:.35}
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# PCA FUNCTIONS
# ============================================================

def prepare_image(uploaded):
    image = Image.open(uploaded).convert("RGB")

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

    variance = float(
        np.sum(pca.explained_variance_ratio_)
    )

    return reconstructed, variance


def compress_image(image_array, components):
    channels = []
    variances = []

    for i in range(3):
        result, variance = compress_channel(
            image_array[:, :, i],
            components
        )
        channels.append(result)
        variances.append(variance)

    result = np.stack(channels, axis=2)

    return (
        np.round(result).astype(np.uint8),
        float(np.mean(variances))
    )


def metrics(original, compressed):
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

    st.markdown("""
    <div class="brand">
        <div class="brand-art">
            <div class="stem"></div>
            <div class="leaf"></div>
            <div class="leaf two"></div>
            <div class="leaf three"></div>
        </div>
        <div class="brand-name">PCA Studio</div>
        <div class="brand-sub">Image Compression</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📊 Analysis"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("""
    <div class="side-poster">
        <div class="side-photo"></div>
        <div class="side-copy">
            Smaller Images<br>
            <span style="font-size:.85rem;color:#5D746B;">
            Brighter Possibilities
            </span>
        </div>
        <div class="side-leaf"></div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.markdown("""
    <div class="top">
        <span>☀️</span>
        <span>Good Morning!</span>
        <span class="avatar">👤</span>
    </div>
    """, unsafe_allow_html=True)

    # HERO
    with st.container(border=True):

        left, right = st.columns([1.35, 1])

        with left:
            st.markdown(
                '<div class="hero-kicker">'
                'SMARTER · LIGHTER · CLEARER'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="hero-title">'
                'PCA Image Compression'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="hero-copy">'
                'Reduce image size while preserving important '
                'visual information using Principal Component Analysis.'
                '</div>',
                unsafe_allow_html=True
            )

            st.write("")

            st.button(
                "Start Compressing  →",
                key="hero_start"
            )

        with right:
            st.markdown("""
            <div class="hero-art">

                <div class="hero-card big float">
                    <div class="landscape"></div>
                </div>

                <div class="hero-arrow">→</div>

                <div class="hero-card small float2">
                    <div class="landscape"></div>
                </div>

                <div class="pca-tag">🧠 PCA</div>

                <div class="hero-note">
                    Same beauty<br>
                    Smaller size ♡
                </div>

            </div>
            """, unsafe_allow_html=True)

    st.markdown(
        '<div class="section-title">Start Compressing</div>',
        unsafe_allow_html=True
    )

    upload_col, control_col, metric_col = st.columns(
        [1.25, 1.0, .75],
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

            st.caption("Choose a JPG, JPEG or PNG image")

            uploaded = st.file_uploader(
                "Upload",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )

            if uploaded is None:
                st.markdown(
                    '<div class="info">'
                    '🖼️ Drag & drop your image here or click Upload'
                    '</div>',
                    unsafe_allow_html=True
                )

    # CONTROL
    with control_col:

        with st.container(border=True):

            st.markdown(
                '<div class="card-title">'
                '🎛️ PCA Components'
                '</div>',
                unsafe_allow_html=True
            )

            if uploaded is None:

                st.markdown(
                    '<div class="info">'
                    'Upload an image to choose components.'
                    '</div>',
                    unsafe_allow_html=True
                )

            else:

                image = prepare_image(uploaded)
                image_array = np.array(image)

                max_components = min(
                    image_array.shape[0],
                    image_array.shape[1]
                )

                components = st.slider(
                    "Components",
                    min_value=1,
                    max_value=max_components,
                    value=min(50, max_components)
                )

                st.markdown(
                    '<div class="info">'
                    '💡 Higher components retain more details.'
                    '</div>',
                    unsafe_allow_html=True
                )

                if st.button(
                    "Compress Image  →",
                    key="compress",
                    use_container_width=True
                ):

                    with st.spinner(
                        "Compressing with PCA..."
                    ):

                        compressed, variance = compress_image(
                            image_array,
                            components
                        )

                    mse, psnr = metrics(
                        image_array,
                        compressed
                    )

                    st.session_state.result = {
                        "original": image_array,
                        "compressed": compressed,
                        "components": components,
                        "variance": variance,
                        "mse": mse,
                        "psnr": psnr,
                    }

                    st.success("Compression complete!")

    # TOP METRIC CARDS
    with metric_col:

        if "result" in st.session_state:

            result = st.session_state.result

            st.markdown(
                f"""
                <div class="metric mint">
                    <div class="metric-icon">▱</div>
                    <div class="metric-value">
                        {result["components"]}
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
                <div class="metric green">
                    <div class="metric-icon">◔</div>
                    <div class="metric-value">
                        {result["variance"] * 100:.1f}%
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
                <div class="metric mint">
                    <div class="metric-icon">▱</div>
                    <div class="metric-value">—</div>
                    <div class="metric-label">Components</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.write("")

            st.markdown(
                """
                <div class="metric green">
                    <div class="metric-icon">◔</div>
                    <div class="metric-value">—</div>
                    <div class="metric-label">
                        Variance Retained
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

    # RESULT AREA
    if "result" in st.session_state:

        result = st.session_state.result

        st.write("")

        m1, m2, m3, m4 = st.columns(4)

        cards = [
            (
                m1,
                "mint",
                "▱",
                str(result["components"]),
                "Components"
            ),
            (
                m2,
                "green",
                "◔",
                f'{result["variance"] * 100:.1f}%',
                "Variance Retained"
            ),
            (
                m3,
                "peach",
                "⌁",
                f'{result["mse"]:.4f}',
                "MSE"
            ),
            (
                m4,
                "yellow",
                "★",
                f'{result["psnr"]:.2f} dB',
                "PSNR"
            ),
        ]

        for col, color, icon, value, label in cards:

            with col:

                st.markdown(
                    f"""
                    <div class="metric {color}">
                        <div class="metric-icon">{icon}</div>
                        <div class="metric-value">
                            {value}
                        </div>
                        <div class="metric-label">
                            {label}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown(
            '<div class="section-title">Image Comparison</div>',
            unsafe_allow_html=True
        )

        with st.container(border=True):

            c1, c2 = st.columns(2)

            with c1:

                st.markdown(
                    '<div class="image-pill">'
                    'Original Image'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    result["original"],
                    width="stretch"
                )

            with c2:

                st.markdown(
                    '<div class="image-pill peach">'
                    'Compressed Image (PCA)'
                    '</div>',
                    unsafe_allow_html=True
                )

                st.image(
                    result["compressed"],
                    width="stretch"
                )

            output = Image.fromarray(
                result["compressed"]
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
            '<div class="info">'
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
        '<div style="color:#68746E;">'
        'Compare image quality at different numbers of components.'
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
        image_array = np.array(image)

        max_components = min(
            image_array.shape[0],
            image_array.shape[1]
        )

        component_list = [
            x for x in [5,10,20,50,100]
            if x <= max_components
        ]

        if not component_list:
            component_list = [max_components]

        rows = []

        progress = st.progress(0)

        for i, components in enumerate(component_list):

            compressed, variance = compress_image(
                image_array,
                components
            )

            mse, psnr = metrics(
                image_array,
                compressed
            )

            rows.append({
                "Components": components,
                "Explained Variance (%)": variance * 100,
                "MSE": mse,
                "PSNR (dB)": psnr,
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

        a, b = st.columns(2)

        with a:
            st.markdown("**Explained Variance**")
            st.line_chart(
                df.set_index("Components")[
                    "Explained Variance (%)"
                ]
            )

        with b:
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
            '<div class="info">'
            '📊 Upload an image to generate the analysis.'
            '</div>',
            unsafe_allow_html=True
        )


st.markdown(
    '<div style="text-align:center;color:#7D837C;font-size:.75rem;'
    'margin-top:1.5rem;">'
    'PCA Studio · Image Compression using Principal Component Analysis'
    '</div>',
    unsafe_allow_html=True
)
