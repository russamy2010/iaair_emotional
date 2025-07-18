import streamlit as st
import pandas as pd
import os
import uuid
import datetime
from collections import defaultdict
from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters # Import necessary functions

# --- Configuration ---
DATA_DIR = "data"
AUDIO_DIR = os.path.join(DATA_DIR, "audio_files")
RATINGS_FILE = os.path.join(DATA_DIR, "emotion_ratings.csv")

# Ensure directories exist
os.makedirs(AUDIO_DIR, exist_ok=True)

# Define emotion labels
EMOTION_LABELS = ["Joy", "Sadness", "Acceptance", "Disgust", "Fear", "Anger", "Surprise", "Anticipation"]

# Initialize ratings CSV if it doesn't exist
if not os.path.exists(RATINGS_FILE):
    initial_ratings_data = pd.DataFrame(columns=["timestamp", "audio_filename", "rater_id", "selected_emotion"])
    initial_ratings_data.to_csv(RATINGS_FILE, index=False)

st.set_page_config(
    page_title="Emotion Annotation App",
    page_icon="🎧",
    layout="centered"
)

st.title("🎧 Emotion Annotation and Fleiss' Kappa Calculator")
st.write("Select an audio file from the repository or upload a new one, then select the emotion and submit your rating. Fleiss' Kappa will be calculated when enough ratings are gathered.")

# --- Rater ID Input ---
rater_id = st.text_input("Enter your Rater ID (e.g., Rater_A, Rater_B):", value=os.getenv("STREAMLIT_USER", "Anonymous_Rater"))
st.info(f"Current Rater ID: **{rater_id}**")

st.markdown("---")

# --- Audio Selection & Player ---
st.header("Audio Selection")

# Get list of existing audio files
existing_audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(('.mp3', '.wav', '.ogg'))]
existing_audio_files.sort() # Sort alphabetically for easier navigation

selected_preloaded_audio = None
uploaded_file_object = None # This will hold the BytesIO object from st.file_uploader

# Offer choice between pre-loaded and upload
selection_method = st.radio("Choose audio source:", ("Select from existing files", "Upload new audio file"))

if selection_method == "Select from existing files":
    if existing_audio_files:
        selected_preloaded_audio = st.selectbox("Select an audio file from the repository:", ["-- Select an audio file --"] + existing_audio_files)
        if selected_preloaded_audio != "-- Select an audio file --":
            audio_path = os.path.join(AUDIO_DIR, selected_preloaded_audio)
            try:
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format=f'audio/{os.path.splitext(selected_preloaded_audio)[1].lstrip(".")}')
                st.success(f"Playing '{selected_preloaded_audio}' from repository.")
            except Exception as e:
                st.error(f"Error loading audio file '{selected_preloaded_audio}': {e}")
                selected_preloaded_audio = None # Reset if error
        else:
            st.info("Please select an audio file to play.")
    else:
        st.warning("No audio files found in the 'data/audio_files' directory. Please upload one.")
        selection_method = "Upload new audio file" # Force upload if no existing files

if selection_method == "Upload new audio file":
    uploaded_file_object = st.file_uploader("Upload a new audio file...", type=["mp3", "wav", "ogg"])
    if uploaded_file_object is not None:
        st.audio(uploaded_file_object.read(), format=f'audio/{os.path.splitext(uploaded_file_object.name)[1].lstrip(".")}')
        st.success(f"New audio file '{uploaded_file_object.name}' uploaded.")
    else:
        st.info("Upload a new audio file or select from existing to proceed.")


# Determine which audio file is currently active for annotation
active_audio_identifier = None
if selected_preloaded_audio and selected_preloaded_audio != "-- Select an audio file --":
    active_audio_identifier = selected_preloaded_audio # Use the actual filename from repo
elif uploaded_file_object is not None:
    # For uploaded files, generate a unique ID for storage (as before)
    unique_id = uuid.uuid4().hex
    original_filename, file_extension = os.path.splitext(uploaded_file_object.name)
    active_audio_identifier = f"{unique_id}_{original_filename}{file_extension}"
    # Save the uploaded file to the repo's audio_files directory
    audio_save_path = os.path.join(AUDIO_DIR, active_audio_identifier)
    # Reset file pointer before saving, as st.audio might have read it
    uploaded_file_object.seek(0)
    with open(audio_save_path, "wb") as f:
        f.write(uploaded_file_object.getbuffer())
    st.success(f"New audio file saved to repository as '{active_audio_identifier}' for future selection.")
else:
    st.warning("Please select or upload an audio file to enable rating.")


st.markdown("---")

# --- Emotion Selection ---
st.header("Emotion Selection")
selected_emotion = st.selectbox("Select the primary emotion for the audio:", EMOTION_LABELS)
st.write(f"You have selected: **{selected_emotion}**")

st.markdown("---")

# --- Save Rating ---
if active_audio_identifier:
    if st.button("Submit Rating"):
        # Load existing ratings
        try:
            ratings_df = pd.read_csv(RATINGS_FILE)
        except pd.errors.EmptyDataError:
            ratings_df = pd.DataFrame(columns=["timestamp", "audio_filename", "rater_id", "selected_emotion"])

        new_rating_data = pd.DataFrame([{
            "timestamp": datetime.datetime.now().isoformat(),
            "audio_filename": active_audio_identifier, # Store the identifier of the active audio
            "rater_id": rater_id,
            "selected_emotion": selected_emotion
        }])

        updated_ratings_df = pd.concat([ratings_df, new_rating_data], ignore_index=True)
        updated_ratings_df.to_csv(RATINGS_FILE, index=False)
        st.success(f"Your rating for '{active_audio_identifier}' (Emotion: '{selected_emotion}') has been saved!")
else:
    st.warning("Please select or upload an audio file before submitting your rating.")

st.markdown("---")

# --- Fleiss' Kappa Calculation and Display ---
st.header("Fleiss' Kappa for Inter-Rater Agreement")

# Load all stored ratings
try:
    all_ratings = pd.read_csv(RATINGS_FILE)
except pd.errors.EmptyDataError:
    all_ratings = pd.DataFrame()

if not all_ratings.empty:
    st.subheader("All Collected Ratings")
    st.dataframe(all_ratings)

    st.subheader("Fleiss' Kappa Results per Soundbite (Min. 2 Raters)")

    grouped_by_audio = all_ratings.groupby("audio_filename")

    fleiss_kappa_results_df = pd.DataFrame() # To store results for display

    for audio_id, group in grouped_by_audio:
        num_raters_for_audio = group["rater_id"].nunique() # Count unique raters for this audio
        if num_raters_for_audio >= 2:
            st.markdown(f"**Soundbite ID: `{audio_id}` (Raters: {num_raters_for_audio})**")

            # Count occurrences of each emotion for this specific audio_id
            emotion_counts = group['selected_emotion'].value_counts().reindex(EMOTION_LABELS, fill_value=0)

            ratings_matrix_for_audio = emotion_counts.values.reshape(1, -1) # Reshape to 1 row, N_categories columns

            if ratings_matrix_for_audio.shape[1] == len(EMOTION_LABELS): # Ensure all categories are present
                try:
                    kappa_value = fleiss_kappa(ratings_matrix_for_audio)
                    fleiss_kappa_results_df = pd.concat([fleiss_kappa_results_df, pd.DataFrame([{
                        "Audio ID": audio_id,
                        "Num Raters": num_raters_for_audio,
                        "Fleiss' Kappa": f"{kappa_value:.3f}"
                    }])], ignore_index=True)

                    st.write(f"  - Calculated Fleiss' Kappa: `{kappa_value:.3f}`")

                    # Interpretation of Fleiss' Kappa (Landis & Koch, 1977)
                    if kappa_value < 0:
                        st.write("  - **Interpretation: Poor agreement** (< 0)")
                    elif 0 <= kappa_value <= 0.20:
                        st.write("  - **Interpretation: Slight agreement** (0.01 - 0.20)")
                    elif 0.21 <= kappa_value <= 0.40:
                        st.write("  - **Interpretation: Fair agreement** (0.21 - 0.40)")
                    elif 0.41 <= kappa_value <= 0.60:
                        st.write("  - **Interpretation: Moderate agreement** (0.41 - 0.60)")
                    elif 0.61 <= kappa_value <= 0.80:
                        st.write("  - **Interpretation: Substantial agreement** (0.61 - 0.80)")
                    else: # 0.81 - 1.00
                        st.write("  - **Interpretation: Almost perfect agreement** (0.81 - 1.00)")
                except Exception as e:
                    st.error(f"Error calculating Fleiss' Kappa for {audio_id}: {e}")
            else:
                 st.warning(f"Not enough categories rated for `{audio_id}` to calculate Fleiss' Kappa properly.")
        else:
            st.info(f"Soundbite ID `{audio_id}` needs at least 2 raters for Fleiss' Kappa calculation (currently {num_raters_for_audio}).")

    if not fleiss_kappa_results_df.empty:
        st.subheader("Summary of Fleiss' Kappa Results")
        st.dataframe(fleiss_kappa_results_df)
    else:
        st.info("No soundbites have enough ratings (at least 2 unique raters) to calculate Fleiss' Kappa yet.")
else:
    st.info("No ratings available yet. Submit some ratings to see the Fleiss' Kappa calculation.")

st.markdown("---")
st.caption("Developed for IAAIR Summer 2025 Research Fellowship. Created by Amy Russ: Email: aruss5@vols.utk.edu")
