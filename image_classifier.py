import os
import logging
from google.generativeai import upload_file
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- 1. Configuration ---
def configure_api():
    """Configures the Gemini API using an environment variable."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    logging.info("Gemini API configured successfully.")

# --- 2. Model Setup ---
def setup_model():
    """Sets up the Gemini model with generation configuration."""
    generation_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 8192,
    }
    system_instruction = """
    Your task is to analyze the provided images of sports supplements (like bars and gels)
    and infer their carbohydrate content (e.g., 40g, 20g). Your output should include only
    the product name and carbs content in grams formatted in text markdown.
    """
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-8b",
        generation_config=generation_config,
        system_instruction=system_instruction,
    )
    logging.info("Gemini model initialized.")
    return model

# --- 3. Image Upload ---
def upload_image(file_path):
    """Uploads an image file for processing."""
    with open(file_path, "rb") as file:
        image_data = file.read()
    return {"mime_type": "image/jpeg", "data": image_data}

# --- 4. Few-shot Classification ---
def classify_carb_content(image_path, model):
    """Classifies carbohydrate content using few-shot examples."""
    image = upload_image(image_path)
    
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
    
    response = model.generate_content([prompt, image])
    return response.text

# --- 5. Batch Processing ---
def batch_process_images(image_folder, model):
    """Processes all images in a folder."""
    results = []
    
    for filename in os.listdir(image_folder):
        if filename.endswith((".jpg", ".jpeg", ".png", ".webp")):
            image_path = os.path.join(image_folder, filename)
            logging.info(f"Processing image: {filename}")
            result = classify_carb_content(image_path, model)
            results.append({"image": filename, "classification": result})
    
    return results

# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Configure API and Model
        configure_api()
        gemini_model = setup_model()
        
        # Set folder path dynamically
        folder_path = os.getenv("IMAGE_FOLDER_PATH", "./images")  # Default to './images'
        
        # Ensure folder exists
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Image folder not found: {folder_path}")
        
        # Process images
        batch_results = batch_process_images(folder_path, gemini_model)
        
        # Output results
        for result in batch_results:
            logging.info(f"Image: {result['image']}")
            logging.info(f"Classification: {result['classification']}\n")
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")