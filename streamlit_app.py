import streamlit as st
import google.generativeai as genai
import os
from PIL import Image

# Configure Gemini API
def configure_api():
    api_key = st.secrets["GEMINI_API_KEY"]
    if not api_key:
        st.error("GEMINI_API_KEY not found in secrets")
        return False
    genai.configure(api_key=api_key)
    return True

# Your existing model setup and classification code here
def setup_model():
    generation_config = {
        "temperature": 0.05,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
    
    system_instruction = """
    Analyze the provided sports supplement image and identify its carbohydrate content.
    Provide a concise response in this format:
    Product: [Name]
    Carbs: [X]g [Exact/Estimated]
    """
    
    return genai.GenerativeModel(
        model_name="gemini-1.5-flash-8b",
        generation_config=generation_config,
        system_instruction=system_instruction,
    )

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
                # Your existing classification code here
                # Adapt your classify_carb_content function for Streamlit
                try:
                    result = classify_carb_content(uploaded_file, model)
                    st.success(result)
                except Exception as e:
                    st.error(f"Error analyzing image: {str(e)}")

if __name__ == "__main__":
    main()
