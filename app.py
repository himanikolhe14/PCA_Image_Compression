import streamlit as st
import numpy as np
import pandas as pd
import io

from PIL import Image
from sklearn.decomposition import PCA
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="PCA Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CSS - DASHBOARD STYLE
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #F4F8FD;
}

/* Main area */

.block-container {
    padding-top: 1.5rem;
    padding-left: 2rem;
    padding-right: 2rem;
    max-width: 1400px;
}


/* Sidebar */

section[data-testid="stSidebar"] {
    background-color: #EAF3FC;
    border-right: 1px solid #D8E7F4;
}

section[data-testid="stSidebar"] .block-container {
    padding: 2rem 1rem;
}


/* Buttons */

.stButton > button {
    border-radius: 14px;
    border: none;
    background-color: #2878D4;
    color: white;
    font-weight: 700;
    padding: 0.7rem 1.2rem;
}

.stButton > button:hover {
    background-color: #1765BD;
    color: white;
}


/* Metrics */

div[data-testid="stMetric"] {
    background-color: white;
    border-radius: 18px;
    padding: 18px;
    border: 1px solid #E0EAF3;
    box-shadow: 0 5px 18px rgba(50, 90, 130, 0.07);
}


/* File uploader */

[data-testid="stFileUploader"] {
    background-color: white;
    border: 2px dashed #8BBCE5;
    border-radius: 20px;
    padding: 15px;
}


/* Images */

[data-testid="stImage"] img {
    border-radius: 18px;
}


/* Slider */

div[data-baseweb="slider"] {
    padding-top: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PCA FUNCTIONS
# =========================================================

def compress_channel(channel, components):

    channel = channel.astype(float)

    mean = channel.mean(axis=0)

    centered = channel - mean

    components = min(
        components,
        centered.shape[0],
        centered.shape[1]
    )

    pca = PCA(
        n_components=components
    )

    transformed = pca.fit_transform(centered)

    reconstructed = pca.inverse_transform(
        transformed
    )

    reconstructed = reconstructed + mean

    reconstructed = np.clip(
        reconstructed,
        0,
        255
    )

    explained_variance = np.sum(
        pca.explained_variance_ratio_
    )

    return reconstructed, explained_variance


def compress_image(image_array, components):

    red = image_array[:, :, 0]
    green = image_array[:, :, 1]
    blue = image_array[:, :, 2]

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

    reconstructed = np.stack(
        [
            red_result,
            green_result,
            blue_result
        ],
        axis=2
    )

    reconstructed = np.round(
        reconstructed
    ).astype(np.uint8)

    explained_variance = (
        red_variance +
        green_variance +
        blue_variance
    ) / 3

    return reconstructed, explained_variance


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🧠 PCA Studio")

    st.caption(
        "Smart Image Compression"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🖼️ Compress Image",
            "📊 Analysis",
            "🧠 Learn PCA",
            "ℹ️ About Project"
        ]
    )

    st.divider()

    st.markdown("### ✨ Project")

    st.caption(
        "Image Compression using "
        "Principal Component Analysis"
    )

    st.caption(
        "B.Tech CSE – AI & ML"
    )


# =========================================================
# HOME PAGE
# =========================================================

if page == "🏠 Home":

    # Welcome
    st.markdown(
        "# Good morning! 👋"
    )

    st.markdown(
        "### Make your images lighter while keeping "
        "their important visual information."
    )

    st.write("")


    # Hero card using Streamlit container
    with st.container(border=True):

        col1, col2 = st.columns(
            [1.5, 1]
        )

        with col1:

            st.markdown(
                "## 🖼️ Smart Image Compression"
            )

            st.write(
                "Use Principal Component Analysis "
                "(PCA) to reduce image dimensions "
                "and reconstruct the image using "
                "important components."
            )

            st.write("")

            if st.button(
                "✨ Start Compressing →"
            ):

                st.info(
                    "Select **🖼️ Compress Image** "
                    "from the sidebar."
                )


        with col2:

            st.markdown(
                """
                <div style="
                    background:#DFF1FF;
                    border-radius:25px;
                    padding:45px;
                    text-align:center;
                    font-size:75px;
                ">
                🖼️ 🧠 ✨
                </div>
                """,
                unsafe_allow_html=True
            )


    st.write("")

    # Quick stats
    st.subheader("📌 What can you do?")

    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.markdown("### 🖼️ Upload")

            st.write(
                "Upload your JPG, JPEG or PNG image."
            )


    with col2:

        with st.container(border=True):

            st.markdown("### 🧠 Compress")

            st.write(
                "Select PCA components and reconstruct "
                "your image."
            )


    with col3:

        with st.container(border=True):

            st.markdown("### 📊 Analyze")

            st.write(
                "Measure explained variance, MSE "
                "and PSNR."
            )


    st.write("")

    st.subheader("🌈 PCA at a glance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.info("🧠 **PCA**\n\nFind important patterns.")

    with col2:
        st.success("📉 **Reduction**\n\nReduce dimensions.")

    with col3:
        st.warning("✨ **Reconstruction**\n\nRebuild the image.")

    with col4:
        st.error("📊 **Analysis**\n\nMeasure quality.")


# =========================================================
# COMPRESSION PAGE
# =========================================================

elif page == "🖼️ Compress Image":

    st.title("🖼️ Compress Your Image")

    st.write(
        "Upload an image and select how many "
        "principal components you want to keep."
    )

    st.divider()


    # Upload
    uploaded_file = st.file_uploader(
        "📤 Upload your image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        # Resize
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


        image_array = np.array(image)


        # Image info
        st.subheader("📐 Image Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Width",
                image.size[0]
            )

        with col2:
            st.metric(
                "Height",
                image.size[1]
            )

        with col3:
            st.metric(
                "Channels",
                3
            )


        st.write("")


        # Original
        st.subheader("🌸 Original Image")

        st.image(
            image_array,
            width="stretch"
        )


        st.divider()


        # PCA control
        st.subheader("🎛️ Choose PCA Components")

        max_components = min(
            image_array.shape[0],
            image_array.shape[1]
        )

        default_components = min(
            50,
            max_components
        )

        components = st.slider(
            "Number of components",
            min_value=1,
            max_value=max_components,
            value=default_components
        )


        st.info(
            f"🧠 Currently using "
            f"**{components} principal components**."
        )


        if st.button(
            "✨ Compress Image",
            use_container_width=True
        ):

            with st.spinner(
                "🧠 PCA is processing your image..."
            ):

                compressed_image, explained_variance = (
                    compress_image(
                        image_array,
                        components
                    )
                )


            # Metrics
            mse = mean_squared_error(
                image_array,
                compressed_image
            )

            psnr = peak_signal_noise_ratio(
                image_array,
                compressed_image,
                data_range=255
            )


            st.success(
                "🎉 Compression completed!"
            )


            # Comparison
            st.subheader(
                "🔍 Image Comparison"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.markdown(
                    "### 🌸 Original"
                )

                st.image(
                    image_array,
                    width="stretch"
                )


            with col2:

                st.markdown(
                    "### ✨ PCA Reconstruction"
                )

                st.image(
                    compressed_image,
                    width="stretch"
                )


            st.write("")

            # Results
            st.subheader(
                "📊 Compression Results"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "🧠 Components",
                    components
                )

            with col2:

                st.metric(
                    "📈 Variance",
                    f"{explained_variance * 100:.2f}%"
                )

            with col3:

                st.metric(
                    "📉 MSE",
                    f"{mse:.4f}"
                )

            with col4:

                st.metric(
                    "⭐ PSNR",
                    f"{psnr:.2f} dB"
                )


            st.write("")

            # Explanation
            with st.container(border=True):

                st.subheader(
                    "💡 What happened?"
                )

                st.write(
                    f"PCA reconstructed the image using "
                    f"**{components} components**."
                )

                st.write(
                    f"Approximately "
                    f"**{explained_variance * 100:.2f}%** "
                    f"of the variance was retained."
                )

                st.write(
                    "Increasing the number of components "
                    "generally preserves more image details."
                )


            # Download
            st.subheader(
                "⬇️ Download"
            )

            output_image = Image.fromarray(
                compressed_image
            )

            buffer = io.BytesIO()

            output_image.save(
                buffer,
                format="PNG"
            )

            st.download_button(
                "💾 Download Compressed Image",
                data=buffer.getvalue(),
                file_name="PCA_Compressed_Image.png",
                mime="image/png",
                use_container_width=True
            )


# =========================================================
# ANALYSIS PAGE
# =========================================================

elif page == "📊 Analysis":

    st.title("📊 PCA Analysis")

    st.write(
        "Analyze how the number of principal components "
        "affects image reconstruction."
    )

    st.divider()


    uploaded_file = st.file_uploader(
        "📤 Upload image for analysis",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")


        if max(image.size) > 512:

            scale = 512 / max(image.size)

            image = image.resize(
                (
                    int(image.size[0] * scale),
                    int(image.size[1] * scale)
                )
            )


        image_array = np.array(image)

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
            x for x in possible_components
            if x <= max_components
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
                    "MSE": mse,
                    "PSNR (dB)": psnr
                }
            )

            progress.progress(
                (i + 1) /
                len(components_list)
            )


        results_df = pd.DataFrame(
            results
        )


        st.subheader(
            "📋 Results Table"
        )

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True
        )


        st.subheader(
            "📈 Explained Variance"
        )

        st.line_chart(
            results_df.set_index(
                "Components"
            )["Explained Variance (%)"]
        )


        st.subheader(
            "📉 MSE"
        )

        st.line_chart(
            results_df.set_index(
                "Components"
            )["MSE"]
        )


        st.subheader(
            "⭐ PSNR"
        )

        st.line_chart(
            results_df.set_index(
                "Components"
            )["PSNR (dB)"]
        )


        # Download CSV

        csv = results_df.to_csv(
            index=False
        )

        st.download_button(
            "📥 Download Analysis CSV",
            data=csv,
            file_name="PCA_Analysis.csv",
            mime="text/csv"
        )


    else:

        st.info(
            "👆 Upload an image to generate "
            "the PCA analysis."
        )


# =========================================================
# LEARN PCA PAGE
# =========================================================

elif page == "🧠 Learn PCA":

    st.title("🧠 Learn About PCA")

    st.write(
        "A simple explanation of how PCA is used "
        "for image compression."
    )

    st.divider()


    with st.container(border=True):

        st.subheader(
            "1️⃣ What is PCA?"
        )

        st.write(
            "Principal Component Analysis (PCA) is a "
            "dimensionality reduction technique."
        )

        st.write(
            "It identifies the directions containing "
            "the most important variation in the data."
        )


    with st.container(border=True):

        st.subheader(
            "2️⃣ How is PCA used for images?"
        )

        st.write(
            "An image contains a large number of pixel "
            "values. PCA represents the image using "
            "a smaller number of important components."
        )


    with st.container(border=True):

        st.subheader(
            "3️⃣ What is a component?"
        )

        st.write(
            "A principal component represents an "
            "important direction or pattern in the data."
        )


    with st.container(border=True):

        st.subheader(
            "4️⃣ What happens when K is small?"
        )

        st.write(
            "Fewer components give stronger dimensionality "
            "reduction, but some visual details may be lost."
        )


    with st.container(border=True):

        st.subheader(
            "5️⃣ What happens when K is large?"
        )

        st.write(
            "More components retain more information "
            "and usually produce a reconstruction closer "
            "to the original image."
        )


    st.success(
        "💡 The goal is to find a useful balance "
        "between dimensionality reduction and image quality."
    )


# =========================================================
# ABOUT PAGE
# =========================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About the Project")

    st.write(
        "### Image Compression Using PCA"
    )

    st.write(
        "This project demonstrates how Principal "
        "Component Analysis can be applied to image "
        "data to reduce dimensionality while "
        "preserving important visual information."
    )

    st.divider()


    st.subheader("🎯 Aim")

    st.write(
        "To implement PCA techniques to reduce image "
        "dimensions while preserving essential information."
    )


    st.subheader("🛠️ Technologies Used")

    col1, col2 = st.columns(2)

    with col1:

        st.write("🐍 Python")
        st.write("📊 NumPy")
        st.write("🖼️ Pillow")

    with col2:

        st.write("🤖 Scikit-learn")
        st.write("📈 Scikit-image")
        st.write("🌐 Streamlit")


    st.subheader("🔄 Project Workflow")

    st.info(
        "Image → Preprocessing → RGB Channels → PCA → "
        "Select Components → Reconstruction → Evaluation"
    )


    st.subheader("📌 Evaluation Metrics")

    st.write(
        "📈 Explained Variance — information retained"
    )

    st.write(
        "📉 MSE — reconstruction error"
    )

    st.write(
        "⭐ PSNR — reconstruction quality"
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🧠 PCA Studio • Image Compression using PCA"
)

st.caption(
    "Built with Python • NumPy • Scikit-learn • Streamlit"
)