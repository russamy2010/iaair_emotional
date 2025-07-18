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
    # Columns: timestamp, audio_filename, rater_id (optional), selected_emotion
    initial_ratings_data = pd.DataFrame(columns=["timestamp", "audio_filename", "rater_id", "selected_emotion"])
    initial_ratings_data.to_csv(RATINGS_FILE, index=False)

st.set_page_config(
    page_title="Emotion Annotation App",
    page_icon="🎧",
    layout="centered"
)

st.title("🎧 Emotion Annotation and Fleiss' Kappa Calculator")
st.write("Upload an audio file, select the emotion you hear, and optionally provide a rater ID. When enough ratings are gathered for a soundbite, Fleiss' Kappa will be calculated.")

# --- Rater ID Input (For tracking) ---
rater_id = st.text_input("Enter your Rater ID (e.g., Rater_A, Rater_B):", value=os.getenv("STREAMLIT_USER", "Anonymous_Rater"))
st.info(f"Current Rater ID: **{rater_id}**")

st.markdown("---")

# --- Audio Player & Uploader ---
st.header("Audio Player")
uploaded_file = st.file_uploader("Choose an audio file...", type=["mp3", "wav", "ogg"])

audio_unique_id = None # Unique ID for the audio file, used as its internal identifier
audio_display_name = None # Original name for display

if uploaded_file is not None:
    # Use a persistent unique ID for the audio file across sessions
    # For a real system, check if this file was already uploaded to avoid duplicates
    # For simplicity here, assume each upload is a 'new' soundbite instance
    audio_unique_id = uuid.uuid4().hex
    original_filename, file_extension = os.path.splitext(uploaded_file.name)
    audio_display_name = uploaded_file.name # Keep original name for display
    audio_save_path = os.path.join(AUDIO_DIR, f"{audio_unique_id}{file_extension}")

    # Save the uploaded file
    with open(audio_save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Audio file '{uploaded_file.name}' saved internally as '{audio_unique_id}{file_extension}'")

    # Display the audio player for the uploaded file
    st.audio(uploaded_file.read(), format=f'audio/{file_extension.lstrip(".")}')
else:
    st.info("Please upload an audio file to proceed with annotation.")

st.markdown("---")

# --- Emotion Selection ---
st.header("Emotion Selection")
selected_emotion = st.selectbox("Select the primary emotion for the audio:", EMOTION_LABELS)
st.write(f"You have selected: **{selected_emotion}**")

st.markdown("---")

# --- Save Rating ---
if uploaded_file is not None:
    if st.button("Submit Rating"):
        # Load existing ratings
        try:
            ratings_df = pd.read_csv(RATINGS_FILE)
        except pd.errors.EmptyDataError:
            ratings_df = pd.DataFrame(columns=["timestamp", "audio_filename", "rater_id", "selected_emotion"])

        # Check if this rater has already rated this specific audio file (optional logic)
        # Prevent re-rating or allow updating a rating
        # For simplicity, allow multiple ratings per audio_unique_id by different raters.
        # If same rater_id and audio_unique_id, may update the existing entry.

        new_rating_data = pd.DataFrame([{
            "timestamp": datetime.datetime.now().isoformat(),
            "audio_filename": audio_unique_id, # Store the unique ID, not original name
            "rater_id": rater_id,
            "selected_emotion": selected_emotion
        }])

        updated_ratings_df = pd.concat([ratings_df, new_rating_data], ignore_index=True)
        updated_ratings_df.to_csv(RATINGS_FILE, index=False)
        st.success(f"Your rating for '{audio_display_name}' (Emotion: '{selected_emotion}') has been saved!")
else:
    st.warning("Upload an audio file to submit your rating.")

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

    # Calculate Fleiss' Kappa for each audio file that has multiple ratings
    st.subheader("Fleiss' Kappa Results per Soundbite (Min. 2 Raters)")
    
    # Group ratings by audio file
    grouped_by_audio = all_ratings.groupby("audio_filename")
    
    fleiss_kappa_results = []

    for audio_id, group in grouped_by_audio:
        num_raters_for_audio = group["rater_id"].nunique() # Count unique raters for this audio
        if num_raters_for_audio >= 2:
            st.markdown(f"**Soundbite ID: `{audio_id}` (Raters: {num_raters_for_audio})**")
            
            # Prepare data for Fleiss' Kappa:
            # Need a matrix where rows are subjects (here, implicitly the soundbite, but Fleiss' Kappa
            # expects a matrix of subjects vs. categories with counts of raters for each category).
            # If calculating for *each* soundbite as a "subject" with multiple raters,
            # need to transform the data.

            # Calculate Kappa for *all* soundbites at once.
            # This requires converting the long format data into a "counts" matrix
            # where rows are audio_filename, columns are categories, and values are counts.
            
            # Map selected_emotion to a numeric index for statsmodels (important!)
            category_mapping = {label: i for i, label in enumerate(EMOTION_LABELS)}
            group['category_idx'] = group['selected_emotion'].map(category_mapping)
            
            # Construct a contingency table for this specific audio_id
            # Each row would be an 'item' (if multiple items within an audio_id to rate)
            # but for Fleiss' Kappa over a single item (this audio_id) and its raters,
            # count how many raters chose each category.
            
            # Count occurrences of each emotion for this specific audio_id
            emotion_counts = group['selected_emotion'].value_counts().reindex(EMOTION_LABELS, fill_value=0)
            
            # The 'table' for fleiss_kappa needs to be [subjects, categories]
            # where subjects are implicitly the items being rated.
            # For a single item (this audio_id), the 'table' would have one row.
            # However, statsmodels.fleiss_kappa expects a table where rows are items,
            # and columns are categories, and entries are counts of raters.
            # If have only one "item" (the audio_id itself), we can just pass
            # the counts for this audio_id across categories as a single row.
            
            # Convert counts to a numpy array, ordered by EMOTION_LABELS
            ratings_matrix_for_audio = emotion_counts.values.reshape(1, -1) # Reshape to 1 row, N_categories columns

            if ratings_matrix_for_audio.shape[1] == len(EMOTION_LABELS): # Ensure all categories are present
                try:
                    kappa_value = fleiss_kappa(ratings_matrix_for_audio)
                    fleiss_kappa_results.append({
                        "Audio ID": audio_id,
                        "Num Raters": num_raters_for_audio,
                        "Fleiss' Kappa": kappa_value
                    })
                    st.write(f"  - Calculated Fleiss' Kappa: `{kappa_value:.3f}`")
                    
                    # Interpretation of Fleiss' Kappa (Landis & Koch, 1977, Hartling and Milne, 2012)
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
    
    if not fleiss_kappa_results:
        st.info("No soundbites have enough ratings (at least 2 unique raters) to calculate Fleiss' Kappa yet.")
    # else:
    #     st.dataframe(pd.DataFrame(fleiss_kappa_results)) # Could display all kappas in a table too
else:
    st.info("No ratings available yet. Submit some ratings to see the Fleiss' Kappa calculation.")

st.markdown("---")
st.caption("Developed for IAAIR Summer 2025 Research Fellowship. Created by Amy Russ: Email: aruss5@vols.utk.edu")
