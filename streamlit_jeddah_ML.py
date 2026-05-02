#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
    return df


df = load_csv("jeddah_traffic_dataset.csv")

st.title("Jeddah Traffic Data Dashboard")

st.info(
    "**City:** Jeddah, Saudi Arabia — Red Sea coastal hub, population ~4.7 million.  \n"
    "**Data source:** Saudi Arabia Road Transport Statistics 2024 — General Authority for Statistics (GASTAT) "
    "& KAPSARC Traffic Accidents Portal.  \n"
    "**Coverage:** Full year 2024 (366 daily records).  \n"
    "**Calibration:** 17,231 serious accidents nationally; Makkah Region = 20.1 % of driving licenses; "
    "Jeddah ≈ 60 % of region → ~12 % national share; 60.3 % intracity / 39.7 % outside cities.  \n"
    "**Notable 2024 events modelled:** Ramadan (11 Mar – 9 Apr), Hajj surge (14–19 Jun), "
    "Eid Al-Fitr, Eid Al-Adha, National Day (23 Sep)."
)

# ===============================
# Prepare Numeric Columns
# ===============================

numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

if len(numeric_columns) == 0:
    st.error("No numeric columns found in the dataset.")
    st.stop()


# ===============================
# Basic Line Chart
# ===============================

feature = st.selectbox(
    "Choose a numeric column for visualization and severity label",
    numeric_columns
)

fig = px.line(df, x="Date", y=feature, title=f"{feature} over time — Jeddah 2024")
st.plotly_chart(fig, use_container_width=True)

st.dataframe(df)


# ===============================
# Machine Learning Options
# ===============================

st.subheader("Machine Learning Options")

st.write("Numeric columns available in the dataset:")
st.write(numeric_columns)


# ===============================
# Decision Tree Classification
# ===============================

if st.button("Run Decision Tree Classification"):

    if len(numeric_columns) < 2:
        st.error("The dataset must contain at least two numeric columns.")
    else:
        data = df.copy()

        threshold = data[feature].mean()

        data["Situation"] = data[feature].apply(
            lambda x: "Severe" if x > threshold else "Not Severe"
        )

        ml_columns = [col for col in numeric_columns if col != feature]

        X = data[ml_columns]
        y = data["Situation"]

        model = DecisionTreeClassifier(random_state=42, max_depth=5)
        model.fit(X, y)

        data["Predicted Situation"] = model.predict(X)

        st.success("Decision Tree Classification completed.")

        st.write("Classification rule:")
        st.write(
            f"If **{feature}** is above its average value **({threshold:.2f})**, "
            "the situation is classified as **Severe**. Otherwise, it is classified as **Not Severe**."
        )

        st.write("Columns used to train the Decision Tree:")
        st.write(ml_columns)

        st.dataframe(data[["Date", feature, "Situation", "Predicted Situation"]])

        st.subheader("Decision Tree Visualization (depth ≤ 5)")

        fig_tree, ax = plt.subplots(figsize=(24, 10))

        plot_tree(
            model,
            feature_names=ml_columns,
            class_names=list(model.classes_),
            filled=True,
            rounded=True,
            fontsize=10,
            ax=ax
        )

        st.pyplot(fig_tree)

        st.subheader("Decision Tree Interpretation")

        tree_rules = export_text(model, feature_names=ml_columns)

        st.write(
            "The Decision Tree learns simple rules from the Jeddah traffic indicators "
            "to predict whether the situation is **Severe** or **Not Severe**."
        )

        st.write(
            "Each internal node in the tree represents a condition on one traffic variable. "
            "If the condition is true, the tree follows the left branch. "
            "If the condition is false, it follows the right branch."
        )

        st.write(
            "The final nodes, called leaf nodes, give the predicted class: "
            "**Severe** or **Not Severe**."
        )

        st.write("Text version of the learned decision rules:")
        st.code(tree_rules)

        st.write(
            "In simple terms, the model checks the most important traffic values step by step. "
            "When these values indicate higher risk or congestion patterns, the model predicts "
            "**Severe**. Otherwise, it predicts **Not Severe**."
        )


# ===============================
# K-Means Clustering
# ===============================

st.subheader("K-Means Clustering")

number_of_clusters = st.slider(
    "Choose the number of clusters",
    min_value=2,
    max_value=10,
    value=3
)

if st.button("Run K-Means Clustering"):

    if len(numeric_columns) < 2:
        st.error("The dataset must contain at least two numeric columns.")
    else:
        data = df.copy()

        X = data[numeric_columns]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=number_of_clusters, random_state=42)
        data["Cluster"] = kmeans.fit_predict(X_scaled)

        st.success(f"K-Means Clustering completed with {number_of_clusters} clusters.")

        st.dataframe(data)

        x_col = numeric_columns[0]
        y_col = numeric_columns[1]

        fig_cluster = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=data["Cluster"].astype(str),
            title=f"K-Means Clustering — Jeddah 2024 ({number_of_clusters} Clusters)",
            labels={"color": "Cluster"}
        )

        st.plotly_chart(fig_cluster, use_container_width=True)
