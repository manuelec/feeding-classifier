import streamlit as st
import google.generativeai as genai
import os
from PIL import Image
import io

# Configure Gemini API
def configure_api():
    api_key = st.secrets["GEMINI_API_KEY"]
    if not api_key:
        st.error("GEMINI_API_KEY not found in secrets")
        return False
    genai.configure(api_key=api_key)
    return True

# Model setup
def setup_model():
    """Sets up the Gemini model with generation configuration."""
    generation_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-8b",
        generation_config=generation_config,
    )
    return model

# Classification function
def classify_carb_content(uploaded_file, model):
    # Convert uploaded file to bytes
    if uploaded_file is None:
        raise ValueError("No file uploaded")
    
    # Read the file into bytes
    image_bytes = uploaded_file.getvalue()
    
    # Create the image data dictionary
    image_data = {
        "mime_type": uploaded_file.type,
        "data": image_bytes
    }
    
    # Few-shot examples
    few_shot_examples = [
        "Example 1: Enervit Isotonic Gel 60ml - 20g carbs - With Thiamin, Niacin and Vitamin B6",
        "Example 2: ENERVIT Competition Bar - 28g carbs",
        "Example 3: Cetilar Race Carb Gel - 40g carbs - (1:0.8 ratio maltodextrin/fructose)",
        "Example 4: Enervit Competition Bar Orange, 30g - 22g carbs",
        "Example 5: SiS GO Energy+Electrolyte Gel 60ml - 22g carbs",
        "Example 6: SiS GO Energy + Caffeine Gel - 22g carbs",                
        "Example 7: SiS GO Isotonic Energy Gels - 22g carbs",   
        "Example 8: Carbs Fuel Original 50g Energy Gel - 50g carbs",   
        "Example 9: CETILAR ULTRARACE CARB GEL 60 ML - 39g carbs",
        'Example 10: Nutrition gel SiS Beta Fuel Strawberry & Lime 60ml - 40g carbs',
        "Example 11: Clif Bar Shot Bloks Energy Chews - 24g carbs",
        "Example 12: 226ers HIGH ENERGY GEL - 50g carbs",
        "Example 13: 226ers ISOTONIC GEL - 22g carbs",
        "Example 14: Maurten Gel 160 - 40g carbs",
        "Example 15: Enervit Carbo Gel  C2:1 PRO 60ml - 40g carbs",
        "Example 16: Enervit Carbo Chews C2:1 - 30g carbs",
        "Example 17: ENERVIT SPORT Isocarb 2:1 650g - 90g (per serving, one full bottle)"
    ]
    
    prompt = f"""
    Here are a few examples of sports supplements with their correct carbohydrate content:
    {' '.join(few_shot_examples)}
    
    Now, analyze the given image of a sports supplement. Identify the product and provide its exact carbohydrate content in grams.
    Provide a concise response in this format:
    Product: [Name]
    Carbs: [X]g [Exact/Estimated]
    """
    
    # Generate content using the model
    response = model.generate_content([prompt, image_data])
    return response.text

def main():
    st.title("Sports Supplement Carb Classifier")
    st.write("Upload images of sports supplements to analyze their carbohydrate content")

    # Initialize API and model
    if not configure_api():
        return
    
    model = setup_model()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Process the image when button is clicked
        if st.button("Analyze"):
            with st.spinner("Analyzing image..."):
                try:
                    result = classify_carb_content(uploaded_file, model)
                    st.success(result)
                except Exception as e:
                    st.error(f"Error analyzing image: {str(e)}")

if __name__ == "__main__":
    main()
