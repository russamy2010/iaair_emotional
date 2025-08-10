README for Supplementary Material - Emotion Annotation and Fleiss' Kappa App and Emotional AI Backend Code

Description:
This supplementary material provides both the Python code to train the Emotional Intelligence AI model and the source code for a Streamlit web application designed for audio emotion annotation and inter-rater agreement analysis using Fleiss' Kappa. In particular, the application allows users to:
1.  Select and play pre-loaded audio soundbites (MP3, WAV, OGG) from the repository.
2.  Optionally upload new audio files, which are then saved to the repository for future selection.
3.  Assign up to three primary emotion labels (Happy, Sad, Angry, Fear, Disgust, Surprise, Neutral) to each soundbite.
4.  Specify an intensity level (Low, Medium, High) for their emotion rating.
5.  Provide a confidence score (Likert scale 1-5) for their rating.
6.  Submit their ratings, which are stored in a CSV file alongside a unique rater ID and timestamp.
7.  View all collected ratings in a tabular format.
8.  Automatically calculate and display Fleiss' Kappa coefficient for inter-rater agreement on the primary emotion, for each soundbite that has received at least two unique ratings.
The app is designed to be easily deployable on Streamlit Cloud directly from a GitHub repository, enabling collaborative annotation efforts.

Size:
The total size of the supplementary material (source code for both the training file and application, initial data files, and this README) is approximately 5.43MB with sample audio (a total of 31 files). This does not include any audio files that users might upload during live application use, as those are stored ephemerally. The sample audio files are from the CREMA-D corpus located here: https://github.com/CheyneyComputerScience/CREMA-D/tree/master (full citation in paper: "The Empathetic Machine: The Introduction of Emotional Intelligence Recognition in Conversational AI"). 

Streamlit App Player Information:
The primary component is a web application built with Streamlit.
-   **Software Name:** Streamlit
-   **Minimum Version:** 1.x (tested with 1.35.0+)
-   **Platform(s):** Runs in a standard web browser (e.g., Google Chrome, Mozilla Firefox, Microsoft Edge, Safari) on any operating system (Windows, macOS, Linux, Android, iOS).
-   **Special Requirements:** No specific browser plugins are required. Internet access is needed to access the deployed application on Streamlit Cloud. Python 3.9+ is required to run the application locally.

Detailed Information about Interacting with the Streamlit Objects:
The core "object" is the Streamlit web application, accessible via a URL once deployed.
1.  **Deployment:** The application is designed for deployment via Streamlit Cloud directly from a GitHub repository. Users can also run it locally by installing Python and Streamlit (`pip install streamlit pandas statsmodels`) and executing `streamlit run audio_app.py` in the project directory.
2.  **Audio Files:** Pre-loaded audio files (MP3, WAV, OGG) are located in the `data/audio_files/` directory within the repository. The app allows selecting these files or uploading new ones.
3.  **Data Storage:** Annotation data (emotions, intensity, confidence, rater ID) is stored in `data/emotion_ratings.csv`. In a Streamlit Cloud deployment, this file is ephemeral and data will reset upon app restarts/redeployment. For persistent storage in a production environment, integration with external cloud storage (e.g., AWS S3, Google Cloud Storage) and databases would be necessary.
4.  **Fleiss' Kappa:** The coefficient is calculated on the 'emotion_1' column (the first selected emotion) for each audio file with at least two unique ratings. Its interpretation (e.g., "Slight agreement", "Moderate agreement") is provided based on Landis & Koch (1977).

Packing List:
-   `audio_app.py`: The main Streamlit application Python source code.
-   `requirements.txt`: Specifies Python dependencies (streamlit, pandas, statsmodels).
-   `data/`: Directory containing:
    -   `audio_files/`: Subdirectory for audio clips (e.g., `sample_audio.mp3`, `another_clip.wav`).
    -   `emotion_ratings.csv`: CSV file for storing annotation data. This file may be initially empty or contain sample data.
- `emotional_ai_backend.py`: The Python code for the emotional intelligence AI model(i.e., the backend).
-  `README.txt`: This file.

Contact Information:
Amy Russ
Doctorate of Engineering Candidate, University of Tennessee-Knoxville
Institute of Applied Artificial Intelligence and Robotics (IAAIR) Summer 2025 Fellow 
aruss5@vols.utk.com
GitHub Profile: https://github.com/russamy2010 
ORCID:0009-0009-8817-6791


