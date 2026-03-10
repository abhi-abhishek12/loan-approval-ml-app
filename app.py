import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# Load dataset
data = pd.read_csv("loan-prediction-dataset.csv")

# Drop Loan_ID
data = data.drop("Loan_ID", axis=1)

# Fix Dependents column
data['Dependents'] = data['Dependents'].replace('3+', 3)
data['Dependents'] = pd.to_numeric(data['Dependents'])

# Fill missing values
data = data.fillna(data.mean(numeric_only=True))

# Encode categorical columns
le = LabelEncoder()

categorical_columns = [
    'Gender','Married','Education',
    'Self_Employed','Property_Area','Loan_Status'
]

for col in categorical_columns:
    data[col] = le.fit_transform(data[col])

# Split data
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# -------------------------
# STREAMLIT UI
# -------------------------

st.title("🏦 Loan Approval Prediction")

st.write("Enter applicant details")

gender = st.selectbox("Gender", ["Female", "Male"])
married = st.selectbox("Married", ["No", "Yes"])
dependents = st.selectbox("Dependents", [0,1,2,3])
education = st.selectbox("Education", ["Not Graduate","Graduate"])
self_employed = st.selectbox("Self Employed", ["No","Yes"])
app_income = st.number_input("Applicant Income")
coapp_income = st.number_input("Coapplicant Income")
loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Amount Term")
credit_history = st.selectbox("Credit History", [0,1])
property_area = st.selectbox("Property Area", ["Rural","Semiurban","Urban"])

# Convert inputs to numbers
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
education = 1 if education == "Graduate" else 0
self_employed = 1 if self_employed == "Yes" else 0

property_map = {"Rural":0,"Semiurban":1,"Urban":2}
property_area = property_map[property_area]

# Prediction button
if st.button("Predict Loan Status"):

    input_data = [[
        gender, married, dependents, education,
        self_employed, app_income, coapp_income,
        loan_amount, loan_term, credit_history,
        property_area
    ]]

    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")