import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Customer Churn Prediction",page_icon="📊",layout="wide")

@st.cache_resource
def load_models():
    ml_models=joblib.load("ml_models.joblib")
    ann=tf.keras.models.load_model("keras_ann.h5",compile=False)
    ann_preprocessor=joblib.load("ann_preprocessor.joblib")
    thresholds=joblib.load("model_thresholds.joblib")
    comparison=pd.read_csv("optimized_model_comparison.csv")
    return ml_models,ann,ann_preprocessor,thresholds,comparison

try:
    ml_models,ann,ann_preprocessor,thresholds,comparison=load_models()
except Exception as e:
    st.error("Error loading model files.")
    st.exception(e)
    st.stop()

@st.cache_data
def load_data():
    return pd.read_csv("Churn_Modelling.csv")

df=load_data()

def add_features(data):
    data=data.copy()
    data["BalanceSalaryRatio"]=data["Balance"]/(data["EstimatedSalary"]+1e-8)
    data["TenureByAge"]=data["Tenure"]/(data["Age"]+1e-8)
    data["HasZeroBalance"]=(data["Balance"]==0).astype(int)
    data["BalancePerProduct"]=data["Balance"]/(data["NumOfProducts"]+1e-8)
    data["SalaryPerProduct"]=data["EstimatedSalary"]/(data["NumOfProducts"]+1e-8)
    data["IsSenior"]=(data["Age"]>=50).astype(int)
    return data

def risk_segment(probability):
    if probability<0.30:
        return "Low"
    elif probability<0.60:
        return "Medium"
    return "High"

def action(risk):
    if risk=="High":
        return "Immediate retention action"
    elif risk=="Medium":
        return "Monitor and engage"
    return "Routine monitoring"

st.title("📊 Customer Churn Prediction System")
st.caption("Machine Learning • Deep Learning • EDA • Model Comparison • Risk Analytics")

tab_prediction,tab_eda,tab_analysis=st.tabs([
    "🔮 Predict Churn",
    "📊 EDA & Insights",
    "📈 Model Analysis"
])

# ============================================================
# TAB 1 — PREDICT CHURN
# ============================================================

with tab_prediction:
    st.header("🔮 Customer Churn Prediction")
    st.write("Enter customer information and compare predictions from traditional ML models and the Keras ANN.")

    col1,col2,col3=st.columns(3)

    with col1:
        credit_score=st.number_input("Credit Score",300,900,650)
        geography=st.selectbox("Geography",["France","Germany","Spain"])
        gender=st.selectbox("Gender",["Male","Female"])
        age=st.number_input("Age",18,100,40)

    with col2:
        tenure=st.number_input("Tenure",0,10,5)
        balance=st.number_input("Balance",0.0,1000000.0,100000.0,step=1000.0)
        num_products=st.number_input("Number of Products",1,4,2)

    with col3:
        has_card=st.selectbox("Has Credit Card",[1,0],format_func=lambda x:"Yes" if x==1 else "No")
        active=st.selectbox("Is Active Member",[1,0],format_func=lambda x:"Yes" if x==1 else "No")
        salary=st.number_input("Estimated Salary",0.0,1000000.0,60000.0,step=1000.0)

    customer=pd.DataFrame([{
        "CreditScore":credit_score,
        "Geography":geography,
        "Gender":gender,
        "Age":age,
        "Tenure":tenure,
        "Balance":balance,
        "NumOfProducts":num_products,
        "HasCrCard":has_card,
        "IsActiveMember":active,
        "EstimatedSalary":salary
    }])

    customer=add_features(customer)

    st.divider()

    predict=st.button("🔮 Predict Churn",type="primary",use_container_width=True)

    if predict:
        results=[]

        for name,model in ml_models.items():
            probability=model.predict_proba(customer)[0,1]
            threshold=thresholds.get(name,0.5)
            prediction="Churn" if probability>=threshold else "No Churn"
            risk=risk_segment(probability)

            results.append({
                "Model":name,
                "Type":"Traditional ML",
                "Churn Probability":probability,
                "Threshold":threshold,
                "Prediction":prediction,
                "Risk":risk,
                "Action":action(risk)
            })

        ann_input=ann_preprocessor.transform(customer)
        ann_probability=ann.predict(ann_input,verbose=0).ravel()[0]
        ann_threshold=thresholds.get("Keras ANN",0.5)
        ann_prediction="Churn" if ann_probability>=ann_threshold else "No Churn"
        ann_risk=risk_segment(ann_probability)

        results.append({
            "Model":"Keras ANN",
            "Type":"Deep Learning",
            "Churn Probability":ann_probability,
            "Threshold":ann_threshold,
            "Prediction":ann_prediction,
            "Risk":ann_risk,
            "Action":action(ann_risk)
        })

        result_df=pd.DataFrame(results)

        primary_model=comparison.loc[
            comparison["ROC-AUC"].idxmax(),"Model"
        ]

        primary=result_df[
            result_df["Model"]==primary_model
        ].iloc[0]

        st.subheader("Customer Churn Assessment")

        col1,col2,col3,col4=st.columns(4)

        col1.metric("Primary Model",primary_model)
        col2.metric("Churn Probability",f"{primary['Churn Probability']*100:.2f}%")
        col3.metric("Prediction",primary["Prediction"])
        col4.metric("Risk",primary["Risk"])

        st.info("Recommended Action: "+primary["Action"])

        st.subheader("Predictions Across All Models")

        display=result_df.copy()
        display["Churn Probability"]=(display["Churn Probability"]*100).round(2).astype(str)+"%"
        display["Threshold"]=(display["Threshold"]*100).round(2).astype(str)+"%"

        st.dataframe(display,use_container_width=True,hide_index=True)

        with st.expander("View Engineered Features"):
            st.dataframe(
                customer[
                    [
                        "BalanceSalaryRatio",
                        "TenureByAge",
                        "HasZeroBalance",
                        "BalancePerProduct",
                        "SalaryPerProduct",
                        "IsSenior"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

# ============================================================
# TAB 2 — EDA & INSIGHTS
# ============================================================

with tab_eda:
    st.header("📊 Exploratory Data Analysis")
    st.write("Explore customer characteristics and churn patterns in the dataset.")

    eda_df=df.drop(
        columns=["RowNumber","CustomerId","Surname"],
        errors="ignore"
    ).copy()

    total_customers=len(eda_df)
    churned_customers=eda_df["Exited"].sum()
    churn_rate=eda_df["Exited"].mean()*100
    average_age=eda_df["Age"].mean()

    col1,col2,col3,col4=st.columns(4)

    col1.metric("Total Customers",f"{total_customers:,}")
    col2.metric("Churned Customers",f"{churned_customers:,}")
    col3.metric("Overall Churn Rate",f"{churn_rate:.2f}%")
    col4.metric("Average Age",f"{average_age:.1f}")

    st.divider()

    # Churn Distribution + Geography
    col1,col2=st.columns(2)

    with col1:
        st.subheader("Churn Distribution")

        counts=eda_df["Exited"].value_counts().sort_index()

        fig,ax=plt.subplots(figsize=(6,4))
        ax.bar(
            ["Stayed","Churned"],
            [counts.get(0,0),counts.get(1,0)]
        )
        ax.set_ylabel("Number of Customers")
        ax.set_title("Customer Churn Distribution")
        st.pyplot(fig,clear_figure=True)

    with col2:
        st.subheader("Churn Rate by Geography")

        geo=eda_df.groupby("Geography")["Exited"].mean()*100

        fig,ax=plt.subplots(figsize=(6,4))
        ax.bar(geo.index,geo.values)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Churn Rate by Geography")
        st.pyplot(fig,clear_figure=True)

    # Age + Activity
    col1,col2=st.columns(2)

    with col1:
        st.subheader("Churn Rate by Age Group")

        eda_df["AgeGroup"]=pd.cut(
            eda_df["Age"],
            bins=[17,30,40,50,60,100],
            labels=["18-30","31-40","41-50","51-60","60+"]
        )

        age_data=eda_df.groupby(
            "AgeGroup",
            observed=True
        )["Exited"].mean()*100

        fig,ax=plt.subplots(figsize=(6,4))
        ax.bar(
            age_data.index.astype(str),
            age_data.values
        )
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Churn Rate by Age Group")
        st.pyplot(fig,clear_figure=True)

    with col2:
        st.subheader("Churn Rate by Activity")

        activity=eda_df.groupby("IsActiveMember")["Exited"].mean()*100

        fig,ax=plt.subplots(figsize=(6,4))
        ax.bar(
            ["Inactive","Active"],
            [activity.get(0,0),activity.get(1,0)]
        )
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Churn Rate by Active Membership")
        st.pyplot(fig,clear_figure=True)

    # Products + Gender
    col1,col2=st.columns(2)

    with col1:
        st.subheader("Churn Rate by Number of Products")

        products=eda_df.groupby("NumOfProducts")["Exited"].mean()*100

        fig,ax=plt.subplots(figsize=(6,4))
        ax.bar(
            products.index.astype(str),
            products.values
        )
        ax.set_xlabel("Number of Products")
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Churn Rate by Number of Products")
        st.pyplot(fig,clear_figure=True)

    with col2:
        st.subheader("Churn Rate by Gender")

        gender=eda_df.groupby("Gender")["Exited"].mean()*100

        fig,ax=plt.subplots(figsize=(6,4))
        ax.bar(gender.index,gender.values)
        ax.set_ylabel("Churn Rate (%)")
        ax.set_title("Churn Rate by Gender")
        st.pyplot(fig,clear_figure=True)

    # Balance + Correlation
    col1,col2=st.columns(2)

    with col1:
        st.subheader("Balance Distribution by Churn")

        fig,ax=plt.subplots(figsize=(6,4))

        sns.boxplot(
            data=eda_df,
            x="Exited",
            y="Balance",
            ax=ax
        )

        ax.set_xticklabels(["Stayed","Churned"])
        ax.set_xlabel("Customer Status")
        ax.set_ylabel("Balance")
        st.pyplot(fig,clear_figure=True)

    with col2:
        st.subheader("Correlation Matrix")

        numeric_data=eda_df.select_dtypes(
            include=np.number
        )

        fig,ax=plt.subplots(figsize=(7,5))

        sns.heatmap(
            numeric_data.corr(),
            annot=True,
            fmt=".2f",
            ax=ax
        )

        st.pyplot(fig,clear_figure=True)

# ============================================================
# TAB 3 — MODEL ANALYSIS
# ============================================================

with tab_analysis:
    st.header("📈 Model Analysis")

    st.subheader("Model Performance Comparison")

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    # ROC-AUC + F1
    col1,col2=st.columns(2)

    with col1:
        if "ROC-AUC" in comparison.columns:
            st.subheader("ROC-AUC Comparison")

            roc_data=comparison[
                ["Model","ROC-AUC"]
            ].set_index("Model")

            st.bar_chart(roc_data)

    with col2:
        if "F1" in comparison.columns:
            st.subheader("F1-Score Comparison")

            f1_data=comparison[
                ["Model","F1"]
            ].set_index("Model")

            st.bar_chart(f1_data)

    # Precision + Recall
    col1,col2=st.columns(2)

    with col1:
        if "Precision" in comparison.columns:
            st.subheader("Precision Comparison")

            precision_data=comparison[
                ["Model","Precision"]
            ].set_index("Model")

            st.bar_chart(precision_data)

    with col2:
        if "Recall" in comparison.columns:
            st.subheader("Recall Comparison")

            recall_data=comparison[
                ["Model","Recall"]
            ].set_index("Model")

            st.bar_chart(recall_data)

    st.divider()

    # Feature Importance
    st.subheader("🔍 XGBoost Feature Importance")

    try:
        importance=pd.read_csv(
            "xgboost_feature_importance.csv"
        )

        importance=importance.sort_values(
            "Importance",
            ascending=True
        )

        st.bar_chart(
            importance.set_index("Feature")["Importance"]
        )

    except FileNotFoundError:
        st.warning(
            "xgboost_feature_importance.csv not found."
        )

    # Risk Segmentation
    st.subheader("🎯 Risk Segmentation")

    try:
        risk=pd.read_csv(
            "risk_segmentation.csv"
        )

        st.dataframe(
            risk,
            use_container_width=True,
            hide_index=True
        )

        if "Actual_Churn_Rate" in risk.columns:

            chart=risk[
                ["Risk Segment","Actual_Churn_Rate"]
            ].set_index("Risk Segment")

            st.bar_chart(chart)

    except FileNotFoundError:
        st.warning(
            "risk_segmentation.csv not found."
        )

st.divider()
st.caption(
    "Customer Churn Prediction | "
    "Traditional ML + Deep Learning + Business Analytics"
)