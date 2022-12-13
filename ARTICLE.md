# Telco Customer Churn Prediction App with Gradio

Customer churn is one of the most important benchmark for a growing business to gauge. While it is not the happiest measure, it is a number that can give your company the naked truth about its customer retention.
It is unrealistic to measure success if you do not measure the inevitable failures too. While you fight for 100% of customers to stick with your company, that's simply unrealistic. That's where customer churn comes in.
In this project, I am am going to demonstrate how I used Gradio to deploy my already built classification model. A link to the source code of the built model will be available in this article.

## Introduction to Gradio

Gradio is the fastest way to demo your machine learning model with a friendly web interface so that anyone can use it, anywhere!
To install Gradio, use the code stated below; 

## To install gradio 
pip install gradio

## Processes taken in building my Gradio App
The processes are similar to that of Streamlit in my previous article. 
Exporting Machine Learning(ML) items
Setting up environment
Importing Machine Learning items from local PC
Building the app Interface
Setting up the backend to process inputs and display outputs
Deployment

## 1. Exporting Machine Learning(ML) items
This is the first step I took in processing my Gradio app. Exports are taken from my initial Jupyter notebook. A link will be provided as stated earlier to view the source codes. The ML items exported include ; the chosen model, encoder, scaler if used in the notebook and also a pipeline if available. These various items can be exported individually but for ease of access, I created a dictionary to export the ML Items at a go. Pickle was used in exporting the ML items. Also, OS is going to be a useful tool in exporting the requirements as well.
Below is an illustration of how to export the ML items using dictionary and pickle.

## Exporting the requirements
requirements = "\n".join(f"{m.__name__}=={m.__version__}" for m in globals().values() if getattr(m, "__version__", None))

with open("requirements.txt", "w") as f:
    f.write(requirements)

#creating a dictionary of exports
to_export = {
    "encoder": oh_encoder,
    "scaler": scaler,
    "model": decision_tree_model,
    "pipeline": None,
}

## exporting ML items
with open('ML_items', 'wb') as file:
    pickle.dump(to_export, file)
    
## 2. Setting up environment
This step involves creating a repository or folder for exported items. A python script or Jupyter notebook is needed to host the backend codes for the success of the app. In my case, I used VS code. 
I created a virtual environment to prevent any disputes with any related variables. 
Below is the code I used in activating my virtual environment and setting up up my streamlit

python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt

To confirm if your virtual environment is successfully activated, you will see (venv) prior to the current path in the active terminal. 

## 3. Importing Machine Learning items from local PC
After the virtual environment is set, we then Import the machine learning toolkit that was exported. Below is the code I used in loading my toolkit into active python script.

## Defining a function to load exported ML toolkit                    
def load_saved_objects(filepath='ML_items'):
    "Function to load saved objects"

    with open(filepath, 'rb') as file:
        loaded_object = pickle.load(file)
    
    return loaded_object

## Loading the toolkit
loaded_toolkit = load_saved_objects('/Users/Admin/Desktop/gradio app/ML_items')
After importing the ML items from local PC, you then instantiate the elements of the toolkit. Below is the code I used in Instantiating my ML tool kit elements

## Instantiating the elements of the Machine Learning Toolkit
print('Instantiating')
best_grid_rf_model = loaded_toolkit["model"]
oh_encoder = loaded_toolkit["encoder"]   
scaler = loaded_toolkit["scaler"]

## 4. Building the app Interface
This step is where I employed the use of Gradio's various components to build my Interface. There are two ways in which the interface can be built ,ie, using gr. Interface() or gr. Blocks(). I used the latter in building my interface. 
The most common components I used are stated below;

gr.Column(): to display a vertical space in your workspace.
gr.Markdown(): to write text within the workspace
gr.Row(): to display a horizontal space in your workspace
gr.Dropdown(): for a dropdown with selected options
gr.Radio(): for a radio
gr.Button(): for a button which will activate a sequence of events when clicked
gr.Slider(): for a slider

Below is an image of how my interface looked after defining inputs.

fig 1

## 5. Setting up the backend to process inputs and display outputs
This section replicates the steps taken in my initial Jupyter notebook. As mentioned earlier, the steps taken in the Jupyter notebook are replicated;
a. Receiving inputs
b. Encoding categorical columns
c. Scaling numerical columns
d. Predicting and returning the output of predictions.
You may refer to the notebook for details on the inputs and the functions.

## 6. Deployment
You have the option to deploy for free by adding "share = True" inside your launch function. The deployed app stays online for just about 72 hours. For an extended period of hosting, you may explore more options on the Gradio site.

Thank you and I hope this article was helpful. ☺️
