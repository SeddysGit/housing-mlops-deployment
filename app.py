import gradio as gr
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("model.pkl")

# Define the prediction function
def predict_price(square_feet, num_rooms):
    input_df = pd.DataFrame([[square_feet, num_rooms]], columns=["square_feet", "num_rooms"])
    prediction = model.predict(input_df)[0]
    return round(prediction, 2)

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Square Feet"),
        gr.Number(label="Number of Rooms")
    ],
    outputs="number",
    title="House Price Predictor",
    description="Enter the size and number of rooms to predict the house price"
)

interface.launch()
