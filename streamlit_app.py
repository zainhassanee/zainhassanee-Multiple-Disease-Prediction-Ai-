import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import base64


model = joblib.load("rf_structured.pkl")
symptom_list = joblib.load("all_symptoms.pkl")
class_names = joblib.load("class_names.pkl")

desc_df = pd.read_csv("symptom_Description.csv")
prec_df = pd.read_csv("symptom_precaution.csv")
main_df = pd.read_csv("dataset.csv")


display_name_map = {sym.replace("_", " ").title(): sym for sym in symptom_list}
display_names = list(display_name_map.keys())

st.set_page_config(page_title="Smart Symptom Classifier", page_icon="⚕", layout="centered")

tab1, tab2, tab3 = st.tabs(["➤ Diagnose", "➤ Dashboard", "➤ About & Notes"])

def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


add_bg_from_local("background.jpg")

with tab1:
    st.title("🩺 Smart Symptom Classifier")
    st.markdown("### ➥ Diagnose common diseases based on your selected symptoms")
    st.markdown("**Disclaimer!** Actual clinical data may vary, and this tool is not intended for real-world medical decision-making.")

    selected_display = st.multiselect("Select your symptoms:", options=display_names)
    selected_symptoms = [display_name_map[name] for name in selected_display]

    if selected_symptoms:
        input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptom_list]
        df_input = pd.DataFrame([input_vector], columns=symptom_list)

        probs = model.predict_proba(df_input)[0]
        top3_indices = probs.argsort()[-3:][::-1]

        st.markdown("### ⚕ Top Predicted Diseases:")
        for rank, idx in enumerate(top3_indices):
            disease = class_names[idx]
            confidence = probs[idx]
            prefix = "✔︎" if rank == 0 else "2." if rank == 1 else "3."
            if rank == 0:
                st.success(f"{prefix} **{disease}** — `{confidence:.2f}` confidence")
            else:
                st.info(f"{prefix} {disease} — `{confidence:.2f}` confidence")

        top_disease = class_names[top3_indices[0]]
        desc_row = desc_df[desc_df["Disease"] == top_disease]
        if not desc_row.empty:
            st.markdown(f"✎ **What is it?** {desc_row['Description'].values[0]}")
        prec_row = prec_df[prec_df["Disease"] == top_disease]
        if not prec_row.empty:
            st.markdown("💊 **Recommended Precautions:**")
            for i in range(1, 5):
                val = prec_row[f"Precaution_{i}"].values[0]
                if pd.notna(val):
                    st.write(f"- {val}")

with tab2:
    st.title("📊 Dataset Dashboard")
    st.markdown("""
    This data comes from the Kaggle: [Disease Symptom Description Dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset).

    - **↳ Rows:** One per disease
    - **↳ Columns:** Symptom_1 to Symptom_17
    - **↳ Includes:** I used other tables provided such as "symptom descriptions" and "precaution tables."
    """)

    disease_counts = main_df["Disease"].value_counts()
    st.subheader("➥ Number of Samples per Disease")
    st.bar_chart(disease_counts)

    st.subheader("➥ Most Common Symptoms")
    all_symptoms_flat = main_df[[col for col in main_df.columns if col.startswith("Symptom")]].values.flatten()
    all_symptoms_flat = pd.Series(all_symptoms_flat).dropna().str.replace(" ", "_").str.lower()
    wordcloud = WordCloud(width=1000, height=400, background_color="white").generate(" ".join(all_symptoms_flat))

    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)

    if st.checkbox("➥ Show raw dataset"):
        st.write(main_df.head(20))

with tab3:
    st.title("📝 About This Project & Learnings")

    st.markdown("""
    ### ➥ Project Summary

    This Smart Symptom Classifier was built using a [public dataset on Kaggle](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset).
    
    The goal was to predict diseases based on user-selected symptoms using a machine learning model, and to present the results in an accessible format for learning and exploration purposes. 

    **⚠ Disclaimer:**  
    > The data in this CSV sheet is for reference and training purposes only. Actual clinical data may vary, and this tool is not intended for real-world medical decision-making.

    ---
    ### ➥ Model Used

    - **Type:** Random Forest Classifier  
    - **Input Format:** Structured binary encoding on 486 features representing symptoms 
    - **Training & Validation:** Performed using an 80/20 train-test split  
    - **Evaluation:** For precision, recall, and f1 score. This kind of made me suspicious on my model, and concluded that it was because of the simplicity of the data, and how clean the data seemed since the beginning. There was 120 entries for each disease, and their symptoms were always quite similar. 

    The symptom input was transformed into a binary vector indicating presence/absence of known symptoms from the dataset. TF-IDF was used in early versions, but was replaced with binary encoding for better structure and interpretability. 

    ---

    ### ➥ Next Steps

    - Find a more complex dataset where I can integrate more factors such as age, sex, region.
    - Apply a more sophisticated model (e.g., XGBoost, or BERT for medical text)
    - Include severity scoring from `Symptom-severity.csv` in some form
    - Build a feedback loop or active learning interface

    ---
    ### ➥ Final Note

    I built this tool to practice ML and to better how I can integrate science/medicine (my passion) into what I know. I would love to improve it with access to better data and expert feedback! 
    """)
    
st.markdown("""<hr style="margin-top: 3em; margin-bottom: 1em;"/>""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center; color: #aaa; font-size: 14px;'>

Made with ❤️ by <b>Sarah Lamond</b><br>

<a href='https://www.linkedin.com/in/sarahlamond/' target='_blank'>LinkedIn</a> &nbsp;|&nbsp;
<a href='https://www.kaggle.com/ctrlsari' target='_blank'>Kaggle</a> &nbsp;|&nbsp;
<a href='https://github.com/sarahlamond' target='_blank'>GitHub</a>

</div>
""", unsafe_allow_html=True)

