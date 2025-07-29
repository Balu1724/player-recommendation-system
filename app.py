import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Player Recommendation System", layout="centered")
st.title("🏆 Player Recommendation System")

uploaded_file = st.file_uploader("📂 Upload your player dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset loaded successfully!")

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # Feature selection
    st.subheader("🛠️ Feature Selection")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    selected_features = st.multiselect("Select features for recommendation:", numeric_columns)

    if selected_features:
        # Prepare clean data: Impute and scale
        df_selected = df[selected_features]
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(df_selected)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)

        st.subheader("🤖 Choose Clustering Algorithm")
        cluster_algo = st.selectbox("Select algorithm", ["KMeans", "DBSCAN"])

        if cluster_algo == "KMeans":
            n_clusters = st.slider("🔢 Number of Clusters", 2, 10, 2)
            if st.button("Train KMeans"):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(X_scaled)
                st.session_state.kmeans = kmeans
                st.session_state.scaler = scaler
                st.session_state.imputer = imputer
                st.session_state.features = selected_features
                st.session_state.df = df

                # Recommended cluster = one with highest feature center sum
                cluster_centers = kmeans.cluster_centers_
                cluster_scores = cluster_centers.sum(axis=1)
                recommended_cluster = np.argmax(cluster_scores)
                st.session_state.recommended_cluster = recommended_cluster

                st.success(f"✅ KMeans model trained with {n_clusters} clusters. Cluster {recommended_cluster} is 'Recommended'.")

                # Plot clusters with PCA
                pca = PCA(n_components=2)
                pca_components = pca.fit_transform(X_scaled)
                cluster_labels = kmeans.predict(X_scaled)
                df_vis = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
                df_vis['Cluster'] = cluster_labels

                fig, ax = plt.subplots()
                sns.scatterplot(data=df_vis, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax)
                st.pyplot(fig)

        elif cluster_algo == "DBSCAN":
            eps = st.slider("📏 Epsilon", 0.1, 5.0, 0.5)
            min_samples = st.slider("🔁 Min Samples", 2, 10, 5)
            if st.button("Train DBSCAN"):
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                db_labels = dbscan.fit_predict(X_scaled)

                n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
                n_noise = list(db_labels).count(-1)

                st.success(f"✅ DBSCAN completed with {n_clusters} clusters and {n_noise} noise points.")

                # Visualize
                pca = PCA(n_components=2)
                pca_components = pca.fit_transform(X_scaled)
                df_vis = pd.DataFrame(pca_components, columns=['PC1', 'PC2'])
                df_vis['Cluster'] = db_labels

                fig, ax = plt.subplots()
                sns.scatterplot(data=df_vis, x='PC1', y='PC2', hue='Cluster', palette='tab10', ax=ax)
                st.pyplot(fig)

        # Predict Section for KMeans
        if "kmeans" in st.session_state:
            st.subheader("🧍 Player Stats for Recommendation")

            input_data = {}
            for feature in st.session_state.features:
                input_data[feature] = st.number_input(f"{feature}", min_value=0.0, step=1.0)

            if st.button("Predict Recommendation"):
                input_df = pd.DataFrame([input_data])
                input_imputed = st.session_state.imputer.transform(input_df)
                input_scaled = st.session_state.scaler.transform(input_imputed)

                predicted_cluster = st.session_state.kmeans.predict(input_scaled)[0]
                recommended_cluster = st.session_state.recommended_cluster

                if predicted_cluster == recommended_cluster:
                    st.success("✅ Player is **Recommended**")
                else:
                    st.warning("❌ Player is **Not Recommended**")

    else:
        st.info("ℹ️ Please select at least one feature.")
else:
    st.info("📤 Upload a dataset to begin.")
