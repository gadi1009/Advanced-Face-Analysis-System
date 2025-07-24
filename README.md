# Advanced Face Analysis System 🔬

An interactive web application for advanced facial recognition and analysis, developed as a final project for the "Advanced Topics in Data Mining" course. This system identifies individuals from a custom, high-quality database built from the LFW dataset and provides a rich, real-time analysis of the input image.

<img width="953" height="555" alt="Screenshot from 2025-07-24 22-36-22" src="https://github.com/user-attachments/assets/52df007c-1e38-45b7-b604-957eb2105eee" />

---

## **Core Features**

* **High-Accuracy Recognition:** Powered by the state-of-the-art **InsightFace** library (`buffalo_l` model package), which implements the ArcFace methodology for superior performance.
* **Advanced Feature Analysis:** Provides real-time estimations for:
    * **Age & Gender**
    * **Facial Landmarks Visualization**
* **Interactive Web UI:** A polished and user-friendly interface built with **Streamlit**, featuring:
    * An intuitive image uploader.
    * A dynamic image gallery displaying all database photos of a recognized individual.
    * A configurable confidence threshold slider to tune the system's sensitivity.
* **High-Quality Database:** Utilizes a custom-filtered version of the renowned **LFW (Labeled Faces in the Wild)** dataset, ensuring a robust and challenging benchmark based on the "quality over quantity" principle.

---

## **Technology Stack**

* **Language:** Python 3
* **Development Environment:** Google Colab (with A100 GPU)
* **Core Libraries:**
    * `InsightFace`: For state-of-the-art face detection and recognition.
    * `Streamlit`: For building the interactive web application UI.
    * `Pandas` & `NumPy`: For data manipulation and vector calculations.
    * `OpenCV` & `Pillow`: For image processing.

---

## **Project Structure & Files**

* `Final_Report.pdf`: The detailed project report (in Hebrew).
* `Project_Notebook.ipynb`: The complete Google Colab notebook containing all development stages.
* `LIVE_VIDEO.mp4`: A video demonstrating the live application in use.
* `README.md`: This file.

---

## **How to Run (in Google Colab)**

This project is designed to be run entirely within the Google Colab environment.

1.  **Open the Notebook:** Upload and open the `Project_Notebook.ipynb` file in Google Colab.
2.  **Set Runtime:** Ensure the notebook is connected to a **GPU-enabled runtime** (`Runtime` > `Change runtime type` > `T4 GPU` or higher).
3.  **Run Cells:** Execute the code cells in order from top to bottom.
    * The initial cells will handle all library installations and connect to your Google Drive.
    * The vector database (`insightface_database.pkl`) will be generated on the first run and saved to your Drive. Subsequent runs will skip this step.
    * The final cell will launch the Streamlit server.
4.  **Access the Application:** The last cell will output a public **ngrok URL**. Open this link in a new browser tab to access and interact with the live application.
