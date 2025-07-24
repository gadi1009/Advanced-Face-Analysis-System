# Advanced Face Analysis System 🔬

An interactive web application for advanced facial recognition and analysis, developed as a final project for the "Advanced Topics in Data Mining" course. This system identifies individuals from a custom, high-quality database built from the LFW dataset and provides a rich, real-time analysis of the input image.

<img width="1001" height="274" alt="Screenshot from 2025-07-24 22-39-38" src="https://github.com/user-attachments/assets/aae93c6f-8b2b-4117-add2-5c0b6888f046" />

<img width="942" height="549" alt="Screenshot from 2025-07-24 22-38-13" src="https://github.com/user-attachments/assets/3e56f4de-8648-481a-ae93-c9b06bfab8d6" />

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
