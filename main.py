#-----Advanced Face Recognition System-----
#Gadi Yohanan

# ----------------- Installations and Imports -----------------
print("--- Step 1: Installing, Importing, and Connecting to Drive ---")
import os, pickle
from google.colab import drive
from tqdm.notebook import tqdm
import pandas as pd, cv2, numpy as np, insightface
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
print("Libraries are ready.")

# ----------------- Connect to Google Drive -----------------
print("\nConnecting to Google Drive...")
drive.mount('/content/drive')
print("Google Drive connected.")

# ==============================================================================
#                           System Configuration
# ==============================================================================
print("\n--- Configuring System Paths and Parameters ---")

# 1. Full path to the folder in Google Drive
DB_SOURCE_IMAGES_PATH = "/content/drive/MyDrive/LFW"

# 2. Name and path of the file where the vector database will be saved
VECTORS_DB_PATH = os.path.join(DB_SOURCE_IMAGES_PATH, "insightface_database.pkl")

# 3. Similarity Threshold (Higher score = more similar)
RECOGNITION_THRESHOLD = 0.5  # Good starting value, can be calibrated

print("Configuration complete. Ready for Step 2.")

# ==============================================================================
#           Step 2: Building Vector Database
# ==============================================================================
print("--- Step 2: Building Vector Database (will run only if needed) ---")

# --- Check if database already exists ---
if os.path.exists(VECTORS_DB_PATH):
    print(f"Database file already exists at '{VECTORS_DB_PATH}'.")
    db_df_check = pd.read_pickle(VECTORS_DB_PATH)
    print(f"Loaded existing database with {len(db_df_check)} vectors.")
    print("Skipping creation. You can proceed to Step 3.")
else:
    # --- Load the model ---
    print("Database not found. Loading InsightFace model (buffalo_l)...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model loaded.")

    # --- Create the database ---
    print("Creating new vector database...")
    all_faces_data = []
    celebrity_folders = [d for d in os.listdir(DB_SOURCE_IMAGES_PATH) if os.path.isdir(os.path.join(DB_SOURCE_IMAGES_PATH, d))]

    for person_name in tqdm(celebrity_folders, desc="Processing Celebrities"):
        person_folder_path = os.path.join(DB_SOURCE_IMAGES_PATH, person_name)
        for image_file in os.listdir(person_folder_path):
            image_path = os.path.join(person_folder_path, image_file)
            try:
                img = cv2.imread(image_path)
                faces = app.get(img)
                if faces:
                    embedding = faces[0].normed_embedding
                    face_data = {"name": person_name, "embedding": embedding, "image_path": image_path}
                    all_faces_data.append(face_data)
            except Exception as e:
                print(f"\nCould not process {image_path}: {e}")

    if all_faces_data:
        db_df = pd.DataFrame(all_faces_data)
        print(f"\nProcessed a total of {len(db_df)} images.")
        print(f"Saving database to '{VECTORS_DB_PATH}'...")
        db_df.to_pickle(VECTORS_DB_PATH)
        print("Database created successfully!")
    else:
        print("No faces were processed.")

print("\n--- Vector Database is ready. Proceed to Step 3. ---")

# ==============================================================================
#           Final Application
# ==============================================================================
print("--- Final Application: Face Recognition with Core Attributes ---")

# --- 0. Installations and Imports (if needed) ---
import gradio as gr, os, cv2, numpy as np, pandas as pd
from PIL import Image
from insightface.app import FaceAnalysis

# --- 1. Load Resources ---
print("Loading resources for the application...")

if 'app' not in locals() or 'genderage' not in app.models: # Check if the correct model is loaded
    print("Loading InsightFace model with gender/age estimation...")
    app = FaceAnalysis(allowed_modules=['detection', 'recognition', 'genderage'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model loaded.")

try:
    VECTORS_DB_PATH = "/content/drive/MyDrive/LFW/insightface_database.pkl"
    RECOGNITION_THRESHOLD = 0.4

    db_df = pd.read_pickle(VECTORS_DB_PATH)
    all_db_embeddings = np.array(db_df['embedding'].tolist())
    print(f"Successfully loaded vector database.")
except Exception as e:
    db_df = None
    print(f"FATAL ERROR: Could not load vector DB. Please run previous steps first. Error: {e}")

# --- 2. Main Recognition Function ---
def find_identity_insightface(new_embedding, db_df, all_db_embeddings, threshold):
    if db_df is None or all_db_embeddings is None:
        return "Error", 0.0

    # Calculate cosine similarity
    similarities = np.dot(all_db_embeddings, new_embedding)
    best_match_index = np.argmax(similarities)
    best_match_score = similarities[best_match_index]

    if best_match_score < threshold:
        return "Unknown", best_match_score

    best_match_name = db_df.iloc[best_match_index]['name']
    return best_match_name, best_match_score

# --- 3. Wrapper function for Gradio ---
def recognize_face_with_all_features(uploaded_image):
    if uploaded_image is None or db_df is None:
        return None, "Please upload an image.", None

    img_rgb = uploaded_image.copy()
    img_bgr = cv2.cvtColor(uploaded_image, cv2.COLOR_RGB2BGR)

    faces = app.get(img_bgr)

    if not faces:
        return None, "No face found in the image.", None

    face = faces[0]
    new_embedding = face.normed_embedding

    # --- Extract Face Attributes ---
    detected_gender = "Male" if face.sex == 'M' else "Female"
    detected_age = face.age
    if detected_age < 0:
        detected_age = "Unknown"
    elif detected_age > 100:
        detected_age = "Over 100"
    else:
        detected_age = int(detected_age)

    # --- Find Identity ---

    name, score = find_identity_insightface(new_embedding, db_df, all_db_embeddings, RECOGNITION_THRESHOLD)

    # --- Prepare Output ---
    result_text = ""
    gallery_images = None

    if name not in ["Unknown", "Error"]:
        display_name = name.replace('_', ' ').title()
        result_text = f"""--- Recognition Details ---\n**Identity:** {display_name}\n**Similarity Score:** {score:.4f}\n\n--- Image Attributes ---\n**Estimated Gender:** {detected_gender}\n**Estimated Age:** ~{detected_age}"""

        person_images_df = db_df[db_df['name'] == name]
        gallery_images = person_images_df['image_path'].tolist()
    else:
        result_text = f"""--- Recognition Details ---\n**Identity:** {name}\n**Best Match Score:** {score:.4f}\n\n--- Image Attributes ---\n**Estimated Gender:** {detected_gender}\n**Estimated Age:** ~{detected_age}"""

    # Draw keypoints on the image
    keypoints = face.kps.astype(int)
    for k_x, k_y in keypoints:
        cv2.circle(img_rgb, (k_x, k_y), 3, (0, 255, 0), -1)

    return img_rgb, gallery_images, result_text

# --- 4. Build and Launch Gradio Interface ---
print("\n--- Launching Final Gradio Interface ---")

with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown("# 🤖 Advanced Face Analysis System")
    gr.Markdown("Upload an image to identify a person, see their estimated attributes, view their photo gallery, and see the detected facial landmarks.")

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="numpy", label="Upload Your Image Here")
            submit_button = gr.Button("Analyze Face", variant="primary")
        with gr.Column(scale=2):
            # Changed the number of lines to 6 as there is less text
            output_text = gr.Textbox(label="Analysis Results", lines=6)
            output_landmarks_image = gr.Image(label="Input with Detected Landmarks")

    output_gallery = gr.Gallery(label="Image Gallery from Database", height="auto")

    submit_button.click(
        fn=recognize_face_with_all_features,
        inputs=input_image,
        outputs=[output_landmarks_image, output_gallery, output_text]
    )

if db_df is not None:
    iface.launch(debug=True)
# ==============================================================================
print("Application is ready. You can now upload images for face recognition and analysis.")
# ==============================================================================
print("Thank you for using the Advanced Face Recognition System!")
# ==============================================================================
# End of main.py
# ==============================================================================
# Note: This code is designed to run in a Jupyter Notebook or Google Colab environment
# and requires the InsightFace library and a compatible GPU for optimal performance.
# ==============================================================================
# Please ensure you have the necessary permissions to access the Google Drive folder specified.
# The code is structured to handle errors gracefully and provide informative messages to the user.

# If you encounter any issues, please check the paths and ensure the required libraries are installed.
# For any further assistance, feel free to reach out.
# ==============================================================================
# End of main.py
# ==============================================================================

