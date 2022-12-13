import gradio as gr
import pandas as pd
import os, pickle
from sklearn.preprocessing import OneHotEncoder
import numpy as np

 
# Defining a function to load exported ML toolkit                    
def load_saved_objects(filepath='ML_items'):
    "Function to load saved objects"

    with open(filepath, 'rb') as file:
        loaded_object = pickle.load(file)
    
    return loaded_object

# Loading the toolkit
loaded_toolkit = load_saved_objects('/Users/Admin/Desktop/gradio app/ML_items')

# Instantiating the elements of the Machine Learning Toolkit
print('Instantiating')
best_grid_rf_model = loaded_toolkit["model"]
oh_encoder = loaded_toolkit["encoder"]   
scaler = loaded_toolkit["scaler"]

# Relevant data inputs
expected_inputs = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity',
'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod','MonthlyCharges','tenure','TotalCharges','SeniorCitizen']

categoricals = ['gender','Partner','Dependents','PhoneService','MultipleLines','InternetService','OnlineSecurity',
'OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','Contract','PaperlessBilling','PaymentMethod']

columns_to_scale = ['tenure','MonthlyCharges','TotalCharges',]

#defining the predict function
def predict(*args, encoder = oh_encoder, scaler = scaler, model = best_grid_rf_model):
    
    # Creating a dataframe of inputs
    Input_data = pd.DataFrame([args], columns= expected_inputs)
    
    # Replacing values for the Senior Citizen column
    Input_data['SeniorCitizen'] = np.where(Input_data['SeniorCitizen']== 'Yes', 0, 1)
    
    #Input_data.to_csv('Input data.csv', index= False)

# Encoding    
    print("Encoding")
    #Input_data[categoricals].to_csv('categorical data.csv', index= False)
    encoded_categoricals = encoder.transform(Input_data[categoricals])
    encoded_categoricals = pd.DataFrame(encoded_categoricals, columns = encoder.get_feature_names_out().tolist())

    Final_input = Input_data.join(encoded_categoricals)
    Final_input.drop(columns= categoricals, inplace= True)

#Scaling
    print("scaling")
    Final_input[columns_to_scale].to_csv('columns to scale data.csv', index= False)
    Final_input[columns_to_scale] = scaler.fit_transform(Final_input[columns_to_scale])

# Modeling
    model_output = model.predict(Final_input)
    return float(model_output[0])
    #output_str = "Hey there.....👋 your customer will"
    #return(output_str,model_output)

# Working on inputs 
with gr.Blocks() as demo:

        gr.Markdown("# Classification Model that Predicts Customer Churn")
        
        gr.Markdown("Tenure and Contract Types")
        with gr.Row():
            tenure = gr.Slider(0, 100,label = 'Tenure')
            Contract = gr.Dropdown(['Month-to-month','One year','Two year'], label="Select Contract Type", value = "Month-to-month")

        gr.Markdown("Payment Info")
        with gr.Column():
            MonthlyCharges = gr.Slider(0, 100,label = 'Monthly Charges')
            PaymentMethod = gr.Dropdown(['Credit card (automatic)','Bank transfer (automatic)','Mailed check','Electronic check' ], label="Payment Method", value = 'Mailed check')
            PaperlessBilling = gr.Dropdown(['Yes','No'], label="Select Billing Type", value = "No" )
            TotalCharges = gr.Slider(0, 100,label = 'Total Charges')

        gr.Markdown("Demography Info")
        with gr.Row():
            Dependents = gr.Radio(["Yes", "No"], label="Dependents?")
            Partner = gr.Radio(["Yes", "No"], label="Partner?")
            gender = gr.Dropdown(['Male','Female'], label="Gender", value = 'Male')
            SeniorCitizen = gr.Radio(["Yes", "No"], label="Senior Citizen")

        gr.Markdown("Internet Service Usage")
        with gr.Column():
            TechSupport = gr.Radio(["Yes", "No"], label="Technical support")
            OnlineSecurity = gr.Radio(["Yes", "No"], label="Online security")
            OnlineBackup = gr.Radio(["Yes", "No"], label="Online Backup Service")
            StreamingTV = gr.Radio(["Yes", "No"], label="Streaming TV Service")
            StreamingMovies = gr.Radio(["Yes", "No"], label="Streaming Movie Service")
            DeviceProtection =  gr.Radio(["Yes", "No"], label="Device Protection")
            InternetService =  gr.Dropdown(["Fiber optic", "DSL", "No"], label="Internet Service Package", value= 'No')

        gr.Markdown("Phone Service Usage")
        with gr.Column():
            MultipleLines =  gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value = "No phone service" )
            PhoneService = gr.Radio(["Yes", "No"], label="Phone Services")

        with gr.Row():
            btn = gr.Button("Predict").style(full_width=True)
            output = gr.Textbox(label="Classification Result") 
               
        btn.click(fn=predict,inputs=[gender,Partner,Dependents,PhoneService,MultipleLines,InternetService,OnlineSecurity,
OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,tenure,TotalCharges,SeniorCitizen],outputs=output)

demo.launch(share= True, debug= True)      








