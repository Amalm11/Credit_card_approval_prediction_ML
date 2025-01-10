import streamlit as st
from PIL import Image
import base64
import pickle
import pandas as pd


# Function to add a background image
def add_bg_from_local(image_path):
    """
    Adds a background image to the Streamlit app using custom CSS.
    :param image_path: Path to the image file (string)
    """
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Add a semi-transparent grey banner for all text */
        .block-container {{
            background-color: rgba(0, 0, 0, 0.5); /* Grey with 70% opacity */
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Set page config for better layout
st.set_page_config(page_title="Credit Card Approval Prediction", layout="wide")

# Add the background image
# Replace 'your_image_path.png' with the actual path of your image file
add_bg_from_local("credit.jpeg.jpg")  # Provide the correct file path here

# Sidebar Navigation
st.sidebar.header("Navigation")

page = st.sidebar.radio("Go to", ("Home", "Dataset", "Prediction", "Graphs", "About"))

# Content Based on Navigation
if page == "Home":
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("""
        <h1 style='text-align: center; font-size: 50px; font-family: "Arial", sans-serif;'>Credit Card Approval Prediction</h1>
    """, unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.write("")
    st.markdown("""
    <div style="text-align: center;">
    A Machine Learning-based application designed to predict whether a credit card application will be approved or denied based on various applicant details.

    """, unsafe_allow_html=True)

    import streamlit as st

    st.markdown(
        """
        <div style="text-align: center;">
            <a href="https://colab.research.google.com/drive/1Z-5JvZL4qzpGnkk0Y7V9q-btBLRZkYK7?usp=sharing" 
               target="_blank" 
               style="text-decoration: none; color: blue; font-size: 18px;">
                Click here for Google Colab link
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    st.write("")

    # st.title("Credit Card Approval Prediction")

elif page == 'Dataset':
    st.write("")
    st.write("")
    st.subheader("Dataset")
    st.markdown("Click here for the [Dataset link](https://www.kaggle.com/datasets/caesarmario/application-data/data)")
    st.write("")

    st.markdown("""
    The dataset contains 690 rows and 16 columns, representing applicant's personal, financial and employment details.
    The description of the features of the dataset used are given below:
    """)

    # Data for the table
    data = {
        "Column Name": [
            "Applicant_Id", "Applicant_Gender", "Owned_Car", "Owned_Realty",
            "Total_Children", "Total_Income", "Income_type", "Education_Type",
            "Family_Status", "Housing_Type", "Owned_Mobile", "Owned_Work_Phone",
            "Owned_Phone", "Owned_Email", "Job_Title", "Total_Family_Members",
            "Applicant_Age", "Years_Of_Working"
        ],
        "Description": [
            "Unique identifier for each applicant.",
            "Gender of the applicant (M/F).",
            "Indicates if the applicant owns a car (1 = Yes, 0 = No).",
            "Indicates if the applicant owns real estate (1 = Yes, 0 = No).",
            "Total number of children in the applicant's family.",
            "Total income of the applicant.",
            "Employment or income category of the applicant.",
            "Education level of the applicant.",
            "Marital or family status of the applicant.",
            "Type of housing the applicant resides in.",
            "Indicates if the applicant owns a mobile phone (1 = Yes, 0 = No).",
            "Indicates if the applicant owns a work phone (1 = Yes, 0 = No).",
            "Indicates if the applicant owns a telephone (1 = Yes, 0 = No).",
            "Indicates if the applicant owns an email address (1 = Yes, 0 = No).",
            "Occupation of the applicant.",
            "Total number of family members.",
            "Age of the applicant in years.",
            "Total years of work experience of the applicant."
        ]
    }
    # Create a DataFrame, then a table
    df = pd.DataFrame(data)
    st.table(df)

    st.write("")

elif page == "Prediction":
    st.title("Prediction Section")
    st.markdown("""
        In this section, we will predict whether a credit card application will be approved or not based on input features.
        Please provide the necessary details for prediction:

    """)

    le = pickle.load(open('le.sav', 'rb'))
    car = st.radio("Own a car: ", ('Yes', 'No'))
    a = le.fit_transform([car])[0]

    realty = st.radio("Own a property: ", ('Yes', 'No'))
    b = le.fit_transform([realty])[0]

    child = st.select_slider("No of children ", options=list(range(0, 11)))

    inc = st.text_input("Total income")

    incometype = st.selectbox("Income type",
                              ('Working', 'Commercial associate', 'State servant', 'Student', 'Pensioner'))
    c = le.fit_transform([incometype])[0]

    fam = st.selectbox("Family status", ('Married', 'Single / not married', 'Civil marriage', 'Separated', 'Widow'))
    d = le.fit_transform([fam])[0]

    house = st.selectbox("Housing type",
                         ('House / apartment', 'Rented apartment', 'Municipal apartment', 'With parents',
                          'Co-op apartment', 'Office apartment'))
    e = le.fit_transform([house])[0]

    jobt = st.selectbox("Job title", ('Security staff', 'Sales staff', 'Accountants', 'Laborers', 'Managers', 'Drivers',
                                      'Core staff', 'High skill tech staff', 'Cleaning staff', 'Private service staff',
                                      'Cooking staff', 'Low-skill Laborers', 'Medicine staff', 'Secretaries',
                                      'Waiters/barmen staff',
                                      'HR staff', 'Realty agents', 'IT staff'))
    f = le.fit_transform([jobt])[0]

    fam_mem = st.select_slider('No of family members', options=list(range(0, 11)))

    yrs = st.text_input("Years of working")

    bad = st.text_input("No of bad debts")

    good = st.text_input("No of good debts")
    features = [a, b, child, inc, c, d, e, f, fam_mem, yrs, bad, good]

    scaler = pickle.load(open('scaler.sav', 'rb'))
    model = pickle.load(open('model.sav', 'rb'))
    pred = st.button("PREDICT")

    if pred:
        result = model.predict(scaler.transform([features]))
        if result == 0:
            st.error("Your request for credit card will not be approved.")
        else:
            st.success('Your request for credit card will be approved.')

elif page == "Graphs":
    st.title("Graphs Section")
    st.markdown("""
        This section visualizes important statistics regarding credit card approval predictions.
    """)

    st.subheader("Correlation graph")
    img1 = Image.open('corr.png')
    st.image(img1, width=1000)
    st.write("")
    st.subheader("Model accuracy comparison")
    img1 = Image.open('comparison.png')
    st.image(img1, width=800)
    st.write("")
    st.subheader("Confusion matrix display for XGBoost Classifier")
    img1 = Image.open('img.png')
    st.image(img1, width=500)



elif page == "About":
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.markdown("""
            <h1 style='text-align: center; font-size: 50px; font-family: "Arial", sans-serif;'>Credit Card Approval Prediction</h1>
        """, unsafe_allow_html=True)
    st.subheader("Introduction")
    st.markdown("""
        <div style="text-align: justify;">
        The Credit Card Approval Prediction project is a machine learning-based application designed to predict whether a credit card application will be approved or denied based on various applicant details.
        This project automates the decision-making process using predictive models, ensuring efficiency, accuracy and fairness.
        </div>
        """
                , unsafe_allow_html=True)
    st.write("")
    st.markdown("""
        <div style="text-align: justify;">
        The dataset used in this project includes key applicant attributes such as age, income, credit score, employment status, and financial history. By leveraging these features, we trained a supervised machine learning model to identify patterns and relationships between applicant characteristics and approval outcomes. The application not only helps financial institutions streamline their operations but also provides applicants with instant feedback on their eligibility, enhancing user experience.
        </div>
        """, unsafe_allow_html=True)

    st.write('')
    st.markdown("""
        <div style="text-align:justify;">
        This project streamlines operations for financial institutions, reducing manual effort while providing instant feedback to applicants. Built using Python and Streamlit, it showcases the potential of machine learning in enhancing decision-making processes and improving user experience.
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.write("")
    st.write("")
    st.subheader("Preprocessing")
    st.markdown("""
        <div style="text-align: justify;">
        The dataset used for the credit card approval prediction has been thoroughly preprocessed to ensure the data is clean, relevant, and ready for machine learning model training. Here is a step-by-step breakdown of the preprocessing process:<br>
        1. Handling Missing Data:<br>
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The dataset does not contain any missing values, so no imputation or removal of missing data was necessary.<br>
        2. Label Encoding:<br>
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Categorical variables such as Industry, Ethnicity, and Citizen were encoded using Label Encoding. Label Encoding is particularly useful when the categorical variables have an inherent order or ranking, converting them into numerical values for better model performance.<br>
        3. Correlation Check:<br>
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A correlation analysis was conducted to identify and remove highly correlated features. Features such as Gender, Married, Ethnicity, DriversLicense, Citizen, and ZipCode were found to be either redundant or irrelevant for the model and were removed from the dataset to avoid overfitting and improve model efficiency.<br>
        4. Data Splitting:<br>
           &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The dataset was split into features (X) and the target variable (y), where X contains all the input variables (independent features), and y represents the target variable (approval status of the credit card).<br>
        5. Feature Scaling:<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The features in X were scaled using Min-Max Scaling. This transformation normalizes the data to a range between 0 and 1, ensuring that all features contribute equally to the model and improving the convergence of machine learning algorithms.<br>
    </div>
        """, unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.subheader("Model Building")

    st.markdown("""
        <div style="text-align: justify;">
        With the preprocessed data ready, various machine learning models were built and evaluated to predict the likelihood of credit card approval. Here’s an overview of the model-building process:<br>
        1. Train-Test Split:<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The dataset was split into training and testing sets using train_test_split from sklearn. This ensures that we have a separate set of data for training the model and evaluating its performance.<br>
        2. Over Sampling:<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;In the preprocessing phase, SMOTE (Synthetic Minority Over-sampling Technique) was applied to address the class imbalance in the Status feature by oversampling the minority class (0), ensuring a more balanced dataset for effective model training.<br>
        3. Model Selection:<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Several classification algorithms such as K-Nearest Neighbors (KNN), Support Vector Classifier (SVC), Decision Tree, Random Forest, Gaussian Naive Bayes (GaussianNB), Gradient Boosting, AdaBoost, XGBoost were evaluated for predicting credit card approval.
        These models were trained and evaluated to determine which one offered the best performance on the test data.<br>
        4. Model Evaluation:<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The models were assessed based on their accuracy, precision, recall, F1-score, and confusion matrix. After trying different algorithms, it was found that XGBoost outperformed the others in terms of prediction accuracy and reliability.<br>
        5. Final Model (XGBoost):<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;XGBoost was selected as the final model due to its superior performance. This gradient boosting model was trained using the training data and evaluated using the test data. XGBoost's ability to handle complex relationships between features made it an ideal choice for this classification task.<br>
        </div>
            """, unsafe_allow_html=True)
    st.write("")
    st.write("")
    st.subheader('Technologies Used')

    data = {
        "Technology": [
            "Python", "Pandas", "Scikit-learn", "XGBoost",
            "Pickle", "Streamlit", "Matplotlib/Seaborn"
        ],
        "Description": [
            "Primary programming language for data manipulation and machine learning.",
            "Used for data manipulation and cleaning.",
            "Machine learning library for model training and evaluation.",
            "Used as the final model for its superior performance in classification tasks.",
            "Used for saving and loading the trained model.",
            "Framework for building and deploying the interactive web app.",
            "Libraries used for data visualization and model performance evaluation."
        ]
    }

    # Convert to DataFrame
    df1 = pd.DataFrame(data)
    st.table(df1)
