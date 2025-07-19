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
EMOTION_LABELS = ["Joy", "Sadness", "Acceptance", "Disgust", "Fear", "Anger", "Surprise", "Anticipation", "Neutral"]
INTENSITY_LEVELS = {1: "Low", 2: "Medium", 3: "High"}
CONFIDENCE_SCORES = {1: "Not at all Confident", 2: "Slightly Confident", 3: "Moderately Confident", 4: "Very Confident", 5: "Extremely Confident"}

# Initialize ratings CSV if it doesn't exist
if not os.path.exists(RATINGS_FILE):
    # Updated columns to include multiple emotions, intensity, and confidence
    initial_ratings_data = pd.DataFrame(columns=[
        "timestamp",
        "audio_filename",
        "rater_id",
        "emotion_1",
        "emotion_2",
        "emotion_3",
        "intensity_level",
        "confidence_score"
    ])
    initial_ratings_data.to_csv(RATINGS_FILE, index=False)

st.set_page_config(
    page_title="Emotion Annotation App",
    page_icon="🎧",
    layout="centered"
)

st.title("🎧 Emotion Annotation and Fleiss' Kappa Calculator")
st.write("Select an audio file, choose up to three emotions, an intensity level, and a confidence score. Fleiss' Kappa will be calculated for the **primary emotion** when enough ratings are gathered.")

# --- Initialize session state for widgets if not already set ---
# This ensures that when the app first runs, or after a submission,
# the default values for the widgets are correctly set *before* they are rendered.
if 'rater_id' not in st.session_state:
    st.session_state['rater_id'] = os.getenv("STREAMLIT_USER", "Anonymous_Rater")
if 'selection_method' not in st.session_state:
    st.session_state['selection_method'] = "Select from existing files"
if 'preloaded_audio_select' not in st.session_state:
    st.session_state['preloaded_audio_select'] = "-- Select an audio file --"
if 'emotion_multiselect' not in st.session_state:
    st.session_state['emotion_multiselect'] = []
if 'intensity_select' not in st.session_state:
    st.session_state['intensity_select'] = list(INTENSITY_LEVELS.keys())[0]
if 'confidence_select' not in st.session_state:
    st.session_state['confidence_select'] = list(CONFIDENCE_SCORES.keys())[0]

# --- Rater ID Input ---
rater_id = st.text_input("Enter your Rater ID (e.g., Rater_A, Rater_B):", value=st.session_state['rater_id'], key='rater_id_input')
st.session_state['rater_id'] = rater_id # Update session state if user changes input
st.info(f"Current Rater ID: **{rater_id}**")

st.markdown("---")

# --- Audio Selection & Player ---
st.header("Audio Selection")

# Get list of existing audio files
existing_audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(('.mp3', '.wav', '.ogg'))]
existing_audio_files.sort()

selected_preloaded_audio = None
uploaded_file_object = None

selection_method = st.radio("Choose audio source:", ("Select from existing files", "Upload new audio file"), key='selection_method')

if selection_method == "Select from existing files":
    if existing_audio_files:
        selected_preloaded_audio = st.selectbox("Select an audio file from the repository:", ["-- Select an audio file --"] + existing_audio_files, key='preloaded_audio_select')
        if selected_preloaded_audio != "-- Select an audio file --":
            audio_path = os.path.join(AUDIO_DIR, selected_preloaded_audio)
            try:
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                file_extension = os.path.splitext(selected_preloaded_audio)[1].lower()
                audio_format = f"audio/{file_extension.lstrip('.')}"
                st.audio(audio_bytes, format=audio_format)
                st.success(f"Playing '{selected_preloaded_audio}' from repository. Format: {audio_format}")
            except Exception as e:
                st.error(f"Error loading audio file '{selected_preloaded_audio}': {e}")
                selected_preloaded_audio = None
        else:
            st.info("Please select an audio file to play.")
    else:
        st.warning("No audio files found in the 'data/audio_files' directory. Please upload one.")
        st.session_state['selection_method'] = "Upload new audio file"
        st.experimental_rerun()

if selection_method == "Upload new audio file":
    uploaded_file_object = st.file_uploader("Upload a new audio file...", type=["mp3", "wav", "ogg"], key='file_uploader')
    if uploaded_file_object is not None:
        file_extension = os.path.splitext(uploaded_file_object.name)[1].lower()
        audio_format = f"audio/{file_extension.lstrip('.')}"
        uploaded_file_object.seek(0)
        st.audio(uploaded_file_object.read(), format=audio_format)
        st.success(f"New audio file '{uploaded_file_object.name}' uploaded and playing. Format: {audio_format}")
    else:
        st.info("Upload a new audio file or select from existing to proceed.")

active_audio_identifier = None
if selected_preloaded_audio and selected_preloaded_audio != "-- Select an audio file --":
    active_audio_identifier = selected_preloaded_audio
elif uploaded_file_object is not None:
    unique_id = uuid.uuid4().hex
    original_filename, file_extension = os.path.splitext(uploaded_file_object.name)
    if not file_extension.startswith('.'):
        file_extension = '.' + file_extension
    active_audio_identifier = f"{unique_id}_{original_filename}{file_extension}"

    audio_save_path = os.path.join(AUDIO_DIR, active_audio_identifier)
    uploaded_file_object.seek(0)
    with open(audio_save_path, "wb") as f:
        f.write(uploaded_file_object.getbuffer())
    st.success(f"New audio file saved to repository as '{active_audio_identifier}' for future selection.")
else:
    st.warning("Please select or upload an audio file to enable rating.")

st.markdown("---")

# --- Emotion Selection (Multi-select) ---
st.header("Emotion Selection")
selected_emotions = st.multiselect(
    "Select up to three primary emotions for the audio:",
    EMOTION_LABELS,
    max_selections=3,
    key='emotion_multiselect',
    default=st.session_state['emotion_multiselect'] # Use session state for default value
)
if len(selected_emotions) > 0:
    st.write(f"You have selected: **{', '.join(selected_emotions)}**")
else:
    st.info("Please select at least one emotion.")

# --- Intensity Level ---
st.header("Intensity Level")
selected_intensity_value = st.selectbox(
    "Select the intensity level:",
    options=list(INTENSITY_LEVELS.keys()),
    format_func=lambda x: f"{x} ({INTENSITY_LEVELS[x]})",
    key='intensity_select',
    index=list(INTENSITY_LEVELS.keys()).index(st.session_state['intensity_select']) # Use session state for index
)
st.write(f"Selected Intensity: **{selected_intensity_value} ({INTENSITY_LEVELS[selected_intensity_value]})**")

# --- Confidence Score (Likert Scale) ---
st.header("Confidence Score")
selected_confidence_value = st.selectbox(
    "How confident are you in your rating?",
    options=list(CONFIDENCE_SCORES.keys()),
    format_func=lambda x: f"{x} ({CONFIDENCE_SCORES[x]})",
    key='confidence_select',
    index=list(CONFIDENCE_SCORES.keys()).index(st.session_state['confidence_select']) # Use session state for index
)
st.write(f"Selected Confidence: **{selected_confidence_value} ({CONFIDENCE_SCORES[selected_confidence_value]})**")

st.markdown("---")

# --- Save Rating ---
if active_audio_identifier and len(selected_emotions) > 0:
    # Define a callback function to handle submission and clear inputs
    def submit_rating():
        try:
            ratings_df = pd.read_csv(RATINGS_FILE)
        except pd.errors.EmptyDataError:
            ratings_df = pd.DataFrame(columns=[
                "timestamp",
                "audio_filename",
                "rater_id",
                "emotion_1",
                "emotion_2",
                "emotion_3",
                "intensity_level",
                "confidence_score"
            ])

        # Prepare emotions for CSV (fill empty with None/NaN)
        emotions_to_save = selected_emotions + [None] * (3 - len(selected_emotions))

        new_rating_data = pd.DataFrame([{
            "timestamp": datetime.datetime.now().isoformat(),
            "audio_filename": active_audio_identifier,
            "rater_id": st.session_state['rater_id'], # Use session_state for current rater_id
            "emotion_1": emotions_to_save[0],
            "emotion_2": emotions_to_save[1],
            "emotion_3": emotions_to_save[2],
            "intensity_level": st.session_state['intensity_select'], # Use session_state values at time of click
            "confidence_score": st.session_state['confidence_select'] # Use session_state values at time of click
        }])

        updated_ratings_df = pd.concat([ratings_df, new_rating_data], ignore_index=True)
        updated_ratings_df.to_csv(RATINGS_FILE, index=False)
        st.success(f"Your rating for '{active_audio_identifier}' (Emotions: {', '.join(selected_emotions)}, Intensity: {st.session_state['intensity_select']}, Confidence: {st.session_state['confidence_select']}) has been saved!")

        # Clear inputs for next rating by updating session_state
        st.session_state['emotion_multiselect'] = []
        st.session_state['intensity_select'] = list(INTENSITY_LEVELS.keys())[0]
        st.session_state['confidence_select'] = list(CONFIDENCE_SCORES.keys())[0]
        st.session_state['preloaded_audio_select'] = "-- Select an audio file --" # Clear selected audio

    st.button("Submit Rating", on_click=submit_rating) # Call the callback when button is clicked
else:
    st.warning("Please select or upload an audio file AND select at least one emotion before submitting your rating.")

st.markdown("---")

# --- Fleiss' Kappa Calculation and Display ---
st.header("Fleiss' Kappa for Inter-Rater Agreement (Primary Emotion Only)")
st.info("Note: Fleiss' Kappa is calculated only for the 'Emotion 1' column, as it is designed for single-category ratings. For multi-label agreement, more advanced methods would be needed.")

try:
    all_ratings = pd.read_csv(RATINGS_FILE)
except pd.errors.EmptyDataError:
    all_ratings = pd.DataFrame()

if not all_ratings.empty:
    st.subheader("All Collected Ratings")
    st.dataframe(all_ratings)

    st.subheader("Fleiss' Kappa Results per Soundbite (Min. 2 Raters for Emotion 1)")

    # Filter for valid emotion_1 entries to ensure categories are proper
    kappa_eligible_ratings = all_ratings.dropna(subset=['emotion_1'])

    # Group ratings by audio file for Fleiss' Kappa on 'emotion_1'
    grouped_by_audio_for_kappa = kappa_eligible_ratings.groupby("audio_filename")

    fleiss_kappa_results_df = pd.DataFrame()

    kappa_emotion_labels = sorted(list(set(EMOTION_LABELS)))

    for audio_id, group in grouped_by_audio_for_kappa:
        num_raters_for_audio = group["rater_id"].nunique()
        
        if num_raters_for_audio >= 2:
            st.markdown(f"**Soundbite ID: `{audio_id}` (Raters for Emotion 1: {num_raters_for_audio})**")

            emotion_1_counts = group['emotion_1'].value_counts().reindex(kappa_emotion_labels, fill_value=0)

            ratings_matrix_for_audio = emotion_1_counts.values.reshape(1, -1)

            if ratings_matrix_for_audio.shape[1] == len(kappa_emotion_labels):
                try:
                    kappa_value = fleiss_kappa(ratings_matrix_for_audio)
                    fleiss_kappa_results_df = pd.concat([fleiss_kappa_results_df, pd.DataFrame([{
                        "Audio ID": audio_id,
                        "Num Raters (Emotion 1)": num_raters_for_audio,
                        "Fleiss' Kappa (Emotion 1)": f"{kappa_value:.3f}"
                    }])], ignore_index=True)

                    st.write(f"  - Calculated Fleiss' Kappa (Emotion 1): `{kappa_value:.3f}`")

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
                    else:
                        st.write("  - **Interpretation: Almost perfect agreement** (0.81 - 1.00)")
                except Exception as e:
                    st.error(f"Error calculating Fleiss' Kappa for {audio_id}: {e}")
            else:
                 st.warning(f"Not enough categories rated for `{audio_id}` for Emotion 1 Kappa calculation.")
        else:
            st.info(f"Soundbite ID `{audio_id}` needs at least 2 unique raters with 'Emotion 1' for Fleiss' Kappa calculation (currently {num_raters_for_audio}).")

    if not fleiss_kappa_results_df.empty:
        st.subheader("Summary of Fleiss' Kappa Results (Primary Emotion)")
        st.dataframe(fleiss_kappa_results_df)
    else:
        st.info("No soundbites have enough ratings for 'Emotion 1' to calculate Fleiss' Kappa yet.")
else:
    st.info("No ratings available yet. Submit some ratings to see the Fleiss' Kappa calculation.")


st.markdown("---")
st.caption("Developed for IAAIR Summer 2025 Research Fellowship. Created by Amy Russ: Email: aruss5@vols.utk.edu")
