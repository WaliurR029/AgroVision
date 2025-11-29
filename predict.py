import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 1. SETUP: Load the Model and Class Indices
print("Loading model and resources...")
model = load_model("best_agrovision_model.keras")

# Load the class map (index -> name)
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

# Create a reverse map to look up names by ID
index_to_class = {v: k for k, v in class_indices.items()}


# 2. DICTIONARY: The Brains of the Operation
disease_solutions = {
    "Pepper__bell___Bacterial_spot": "Partial Solution: Remove infected leaves immediately. Apply copper-based fungicides (like fixed copper) every 7-10 days. Avoid overhead watering to prevent spread.",
    "Pepper__bell___healthy": "Healthy: No treatment needed. Maintain good watering practices and monitor for pests.",
    "Potato___Early_blight": "Fungicide: Apply fungicides containing Mancozeb or Chlorothalonil. Remove infected lower leaves. Ensure proper nitrogen levels in soil.",
    "Potato___Late_blight": "URGENT: This is destructive. Remove and destroy infected plants immediately (do not compost). Apply fungicides like metalaxyl or copper sprays to protect nearby plants.",
    "Potato___healthy": "Healthy: No treatment needed. Ensure soil is well-drained to prevent future rot.",
    "Tomato_Bacterial_spot": "Sanitation: Remove infected plant parts. Sterilize tools with 10% bleach solution. Apply copper bactericides early in the season.",
    "Tomato_Early_blight": "Pruning: Remove infected lower leaves to improve airflow. Mulch soil to prevent spore splash. Use fungicides like chlorothalonil if severe.",
    "Tomato_Late_blight": "URGENT: Remove infected plants immediately. Apply copper-based fungicides or those with active ingredient 'chlorothalonil' to protect healthy plants.",
    "Tomato_Leaf_Mold": "Ventilation: High humidity causes this. Prune plants to increase airflow. Water at the base, not on leaves. Fungicides are rarely needed if airflow is good.",
    "Tomato_Septoria_leaf_spot": "Sanitation: Remove fallen leaves and infected lower leaves. Avoid watering in the evening. Apply copper fungicide or mancozeb.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Pest Control: Spray plants with a strong stream of water to dislodge mites. Use insecticidal soap or neem oil for organic control.",
    "Tomato__Target_Spot": "Fungicide: Improve airflow. Apply fungicides containing azoxystrobin or chlorothalonil according to label instructions.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Vector Control: This is spread by whiteflies. Use yellow sticky traps. Remove and destroy infected plants immediately to stop spread.",
    "Tomato__Tomato_mosaic_virus": "No Cure: Remove and destroy infected plants. Wash hands and tools thoroughly (smokers should wear gloves). Do not replant tomatoes in the same soil for 2 years.",
    "Tomato_healthy": "Healthy: No treatment needed. Continue regular care."
}

# 3. THE FUNCTION: Predict and Solve
def diagnose_plant(image_path):
    try:
        # Load the image
        img = image.load_img(image_path, target_size=(256, 256))
        
        # Convert to array and normalize (Just like training!)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Make it a batch of 1
        img_array /= 255.0 

        # Make Prediction
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions) # Get the highest score index
        confidence = np.max(predictions)         # Get the confidence score

        # Get the Name
        disease_name = index_to_class[predicted_index]

        # Get the Solution
        solution = disease_solutions.get(disease_name, "No specific solution found.")

        # Print Report
        print("\n" + "="*40)
        print(f"🌱 DIAGNOSIS REPORT")
        print("="*40)
        print(f"Detected Disease:  {disease_name}")
        print(f"Confidence Level:  {confidence * 100:.2f}%")
        print("-" * 40)
        print(f"💊 RECOMMENDATION:\n{solution}")
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"Error processing image: {e}")

# ==========================================
# 4. RUN IT: Change this path to your image!
# ==========================================
# Example: 
test_image = r"C:\Users\Waliur\OneDrive\Documents\Codes\python\Projects\AgroVision\PlantVillage\Tomato_Early_blight\0abc57ec-7f3b-482a-8579-21f3b2fb780b___RS_Erly.B 7609.JPG"

diagnose_plant(test_image)