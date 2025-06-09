"""
Audiobook Creator
Copyright (C) 2025 Prakhar Sharma

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import re
import os
import time
import json
import random
import asyncio
import traceback
from collections import Counter
from openai import OpenAI, AsyncOpenAI
from tqdm import tqdm
import torch
from gliner import GLiNER
import warnings
from config.constants import TEMP_DIR, get_current_llm_client
from utils.file_utils import write_jsons_to_jsonl_file, empty_file, write_json_to_file
from utils.find_book_protagonist import find_book_protagonist
from utils.llm_utils import check_if_have_to_include_no_think_token, check_if_llm_is_up
from utils.embedding_utils import CharacterEmbedding, CharacterRelationshipGraph
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from utils.config_manager import config_manager


def download_with_progress(model_name):
    print(f"Starting download of {model_name}")

    # Define cache directory
    cache_dir = os.path.join(os.getcwd(), "model_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Log progress manually since tqdm might not work in Docker
    print("Download in progress - this may take several minutes...")

    # Download without tqdm
    snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        local_files_only=False,
        local_dir=cache_dir,
    )

    print(f"Download complete for {model_name}")

    # Load the model from cache
    return GLiNER.from_pretrained(model_name, cache_dir=cache_dir)


def cluster_characters_by_embedding(character_list, character_contexts, embedding_system, similarity_threshold=0.8):
    """Cluster similar character names using embedding similarity with context."""
    if not character_list or not embedding_system:
        return character_list, {char: char for char in character_list}
    
    # Create character clusters
    clusters = {}
    processed = set()
    
    for char in character_list:
        if char in processed or char.lower() in ["narrator", "unknown"]:
            continue
            
        # Create a new cluster with this character as the representative
        cluster_key = char
        clusters[cluster_key] = [char]
        processed.add(char)
        
        char_contexts = character_contexts.get(char, [])
        # Use first 200 chars from first 4 contexts + character frequency info
        char_context_sample = " ".join(char_contexts[:4])[:200] if char_contexts else ""
        char_frequency = len(char_contexts)
        char_text = f"Character named '{char}' appears {char_frequency} times. Dialogue context: {char_context_sample}"
        char_embedding = embedding_system.get_embedding(char_text)
        
        for other_char in character_list:
            if other_char in processed or other_char.lower() in ["narrator", "unknown"]:
                continue
                
            other_contexts = character_contexts.get(other_char, [])
            other_context_sample = " ".join(other_contexts[:3])[:200] if other_contexts else ""
            other_frequency = len(other_contexts)
            other_text = f"Character named '{other_char}' appears {other_frequency} times. Dialogue context: {other_context_sample}"
            other_embedding = embedding_system.get_embedding(other_text)
            
            similarity = embedding_system.calculate_similarity(char_embedding, other_embedding)
            
            # Calculate name similarity using simple string matching
            def calculate_name_similarity(name1, name2):
                """Calculate similarity between two character names."""
                name1_lower = name1.lower().strip()
                name2_lower = name2.lower().strip()
                
                # Exact match
                if name1_lower == name2_lower:
                    return 1.0
                
                # Check if one name is contained in the other
                if name1_lower in name2_lower or name2_lower in name1_lower:
                    return 0.8
                
                # Check for shared words
                words1 = set(name1_lower.split())
                words2 = set(name2_lower.split())
                
                if words1 & words2:  # If there are common words
                    return 0.6
                
                return 0.0
            
            name_similarity = calculate_name_similarity(char, other_char)
            
            # Use higher of context similarity or name similarity
            final_similarity = max(similarity, name_similarity)
            
            # Debug output for similarity scores
            if final_similarity > 0.5:  # Log potential matches
                print(f"DEBUG: '{char}' vs '{other_char}' - Context: {similarity:.3f}, Name: {name_similarity:.3f}, Final: {final_similarity:.3f}")
            
            if final_similarity >= similarity_threshold:
                clusters[cluster_key].append(other_char)
                processed.add(other_char)
                print(f"CLUSTERED: '{other_char}' merged into '{char}' (similarity: {final_similarity:.3f})")
    
    # Return the representative (first) character from each cluster
    clustered_characters = list(clusters.keys())
    
    # Create mapping from all variations to representative
    character_mapping = {}
    for representative, variations in clusters.items():
        for variation in variations:
            character_mapping[variation] = representative
    
    return clustered_characters, character_mapping


def select_top_characters(character_frequencies, protagonist, max_characters=8):
    """Select top characters based on frequency, limited to max_characters."""
    # Always include narrator
    main_characters = ["narrator"]
    
    # Sort characters by frequency (excluding narrator)
    sorted_chars = sorted(
        [(char, freq) for char, freq in character_frequencies.items() 
         if char != "narrator" and is_valid_character_name(char)], 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # Ensure protagonist is included if valid
    protagonist_lower = protagonist.lower()
    protagonist_found = False
    
    for char_name, frequency in sorted_chars:
        if char_name.lower() == protagonist_lower:
            protagonist_found = True
            break
    
    # Add protagonist first if found
    if protagonist_found:
        main_characters.append(protagonist_lower)
    
    # Add other top characters up to max_characters
    for char_name, frequency in sorted_chars:
        if len(main_characters) >= max_characters:
            break
            
        if char_name.lower() != protagonist_lower:  # Don't add protagonist twice
            main_characters.append(char_name.lower())
    
    return set(main_characters)


def is_valid_character_name(name):
    """Check if a name represents an actual proper name rather than a generic descriptor."""
    if not name or not name.strip():
        return False
    
    # Simple normalization - lowercase and clean whitespace
    normalized = name.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    original_words = name.strip().split()  # Keep original capitalization
    words = normalized.split()
    
    # Titles that can prefix proper names
    valid_titles = {
        'detective', 'lieutenant', 'sergeant', 'chief', 'agent', 'investigator',
        'doctor', 'dr', 'professor', 'prof', 'captain', 'officer',
        'mr', 'mrs', 'ms', 'miss', 'sir', 'lady', 'lord',
        'judge', 'attorney', 'lawyer', 'councilman', 'mayor', 'senator'
    }
    
    # Pure generic descriptors (no proper name component)
    pure_generics = {
        'man', 'woman', 'person', 'people', 'child', 'boy', 'girl', 'adult',
        'stranger', 'voice', 'someone', 'anyone', 'everyone', 'nobody',
        'unknown', 'narrator', 'character', 'speaker', 'individual',
        'male', 'female', 'human', 'figure', 'shadow', 'silhouette',
        'guard', 'soldier', 'worker', 'employee', 'customer',
        'waiter', 'waitress', 'driver', 'pilot', 'nurse', 'teacher', 
        'student', 'manager', 'boss', 'receptionist', 'clerk', 'cashier', 
        'bartender', 'chef', 'cook', 'partner', 'colleague', 'friend', 
        'neighbor', 'landlord', 'tenant', 'guy', 'dude', 'buddy', 'pal', 
        'somebody', 'shooter', 'victim', 'suspect', 'witness', 'caller', 
        'client', 'husband', 'wife', 'mom', 'dad', 'mother', 'father', 
        'parent', 'son', 'daughter', 'kid', 'baby', 'elder'
    }
    
    # Check if it's a pure generic descriptor
    if normalized in pure_generics:
        return False
    
    # Check if it starts with possessives or articles + generics
    bad_prefixes = ['the ', 'a ', 'an ', 'some ', 'another ', 'my ', 'your ', 'his ', 'her ', 'their ', 'our ']
    for prefix in bad_prefixes:
        if normalized.startswith(prefix):
            remaining = normalized[len(prefix):].strip()
            if remaining in pure_generics:
                return False
    
    
    # Exclude pure numbers
    if normalized.replace(' ', '').isdigit():
        return False
    
   
    
    # Special handling for multi-word names
    if len(words) > 1:
        # Check if it's Title + Proper Name (e.g., "Detective Bosch")
        if len(words) == 2 and words[0] in valid_titles:
            # Second word should look like a proper name (capitalized in original)
            if len(original_words) >= 2 and original_words[1][0].isupper():
                return True
        
        # Check if it has at least one word that's not a generic descriptor
        has_proper_name = False
        for i, word in enumerate(words):
            if word not in pure_generics and word not in valid_titles:
                # Check if this word was capitalized in original (likely proper name)
                if i < len(original_words) and original_words[i][0].isupper():
                    has_proper_name = True
                    break
        
        if not has_proper_name:
            return False
    
    return True


load_dotenv()

# Get LLM configuration from config manager
llm_config = config_manager.get_llm_config()
async_openai_client = AsyncOpenAI(
    base_url=llm_config["base_url"], 
    api_key=llm_config["api_key"]
)
model_name = llm_config["model_name"]

# warnings.simplefilter("ignore")

print("\n🚀 **Downloading the GLiNER Model ...**")

gliner_model = download_with_progress("urchade/gliner_large-v2.1")

print("\n🚀 **GLiNER Model Backend Selection**")

if torch.cuda.is_available():
    print("🟢 Using **CUDA** backend (NVIDIA GPU detected)")
    gliner_model = gliner_model.cuda()  # For Nvidia CUDA Accelerated GPUs
elif torch.backends.mps.is_available():
    print("🍏 Using **MPS** backend (Apple Silicon GPU detected)")
    gliner_model = gliner_model.to("mps")  # For Apple Silicon GPUs
else:
    print("⚪ Using **CPU** backend (No compatible GPU found)")

print("✅ Model is ready!\n")


def extract_dialogues(text):
    """Extract dialogue lines enclosed in quotes."""
    return re.findall(r'("[^"]+")', text)


def identify_speaker_using_named_entity_recognition(
    line_map: list[dict],
    index: int,
    line: str,
    prev_speaker: str,
    protagonist: str,
    character_gender_map: dict,
    known_characters: set = None,
    embedding_system: CharacterEmbedding = None,
    relationship_graph: CharacterRelationshipGraph = None,
) -> str:
    """
    Identifies the speaker of a given line in a text using Named Entity Recognition (NER).

    This function analyzes the provided line and its context to determine the speaker. It uses
    a pre-trained NER model to detect entities and matches them with known characters or pronouns.
    If no entity is found, it falls back to the previous speaker or assigns a default value.

    Args:
        line_map (list[dict]): A list of dictionaries representing lines of text, where each dictionary
                              contains information about a line (e.g., the text itself).
        index (int): The index of the current line in the `line_map`.
        line (str): The current line of text to analyze.
        prev_speaker (str): The speaker identified in the previous line.
        protagonist (str): The name of the protagonist, used to resolve first-person references.
        character_gender_map (dict): A dictionary mapping character names to their genders, used to
                                     resolve third-person references.

    Returns:
        str: The identified speaker, normalized to lowercase.
    """

    current_line = line
    text = f"{current_line}"
    speaker: str = "narrator"  # Default speaker is the narrator

    # Labels for the NER model to detect
    labels = ["character", "person"]

    # Lists of pronouns for different person and gender references
    first_person_person_single_references = [
        "i",
        "me",
        "my",
        "mine",
        "myself",
    ]  # First person singular
    first_person_person_collective_references = [
        "we",
        "us",
        "our",
        "ours",
        "ourselves",
    ]  # First person collective
    second_person_person_references = [
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    ]  # Second person
    third_person_male_references = ["he", "him", "his", "himself"]  # Third person male
    third_person_female_references = [
        "she",
        "her",
        "hers",
        "herself",
    ]  # Third person female
    third_person_others_references = [
        "they",
        "them",
        "their",
        "theirs",
        "themself",
        "themselves",
        "it",
        "its",
        "itself",
    ]  # Third person neutral/unknown

    # Extract character names based on gender from the character_gender_map
    gender_scores = list(character_gender_map["scores"].values())
    male_characters = [x["name"] for x in gender_scores if x["gender"] == "male"]
    female_characters = [x["name"] for x in gender_scores if x["gender"] == "female"]
    other_characters = [x["name"] for x in gender_scores if x["gender"] == "unknown"]

    # Check if GLiNER model is loaded, reload if needed
    global gliner_model
    if 'gliner_model' not in globals() or gliner_model is None:
        gliner_model = download_with_progress("urchade/gliner_large-v2.1")
        if torch.cuda.is_available():
            gliner_model = gliner_model.cuda()
        elif torch.backends.mps.is_available():
            gliner_model = gliner_model.to("mps")

    # Use the NER model to detect entities in the current line
    entities = gliner_model.predict_entities(text, labels)
    entity = entities[0] if len(entities) > 0 else None

    # If no entity is found, check previous lines (up to 5 lines back) for context
    loop_index = index - 1
    while (entity is None) and loop_index >= max(0, index - 5):
        prev_lines = "\n".join(x["line"] for x in line_map[loop_index:index])
        text = f"{prev_lines}\n{current_line}"
        entities = gliner_model.predict_entities(text, labels)
        entity = entities[0] if len(entities) > 0 else None
        loop_index -= 1

    # Determine the speaker based on the detected entity or fallback logic
    if entity is None:
        # If no entity is found, try to use embedding similarity with context
        if embedding_system and known_characters and current_line:
            similar_speaker = embedding_system.find_similar_character(
                "unknown", current_line, list(known_characters), threshold=0.6
            )
            if similar_speaker:
                speaker = similar_speaker
            elif prev_speaker == "narrator":
                speaker = "unknown"
            else:
                speaker = prev_speaker
        else:
            # Use the previous speaker or mark as unknown
            if prev_speaker == "narrator":
                speaker = "unknown"
            else:
                speaker = prev_speaker
    elif entity["text"].lower() in first_person_person_single_references:
        # First-person singular pronouns refer to the protagonist
        speaker = protagonist
    elif entity["text"].lower() in first_person_person_collective_references:
        # First-person collective pronouns refer to the previous speaker
        speaker = prev_speaker
    elif entity["text"].lower() in second_person_person_references:
        # Second-person pronouns refer to the previous speaker
        speaker = prev_speaker
    elif entity["text"].lower() in third_person_male_references:
        # Third-person male pronouns refer to the last mentioned male character
        last_male_character = (
            male_characters[-1] if len(male_characters) > 0 else "unknown"
        )
        speaker = last_male_character
    elif entity["text"].lower() in third_person_female_references:
        # Third-person female pronouns refer to the last mentioned female character
        last_female_character = (
            female_characters[-1] if len(female_characters) > 0 else "unknown"
        )
        speaker = last_female_character
    elif entity["text"].lower() in third_person_others_references:
        # Third-person neutral/unknown pronouns refer to the last mentioned neutral/unknown character
        last_other_character = (
            other_characters[-1] if len(other_characters) > 0 else "unknown"
        )
        speaker = last_other_character
    else:
        # If the entity is not a pronoun, use the entity text as the speaker
        speaker = entity["text"]
        

    # Update embedding system with context if available
    if embedding_system and speaker and speaker != "unknown":
        embedding_system.update_context(speaker, current_line, index)
        
        # Track character interactions if we have a relationship graph
        if relationship_graph and prev_speaker and prev_speaker != "narrator" and prev_speaker != speaker:
            relationship_graph.add_interaction(speaker, prev_speaker, current_line, index)

    return speaker.lower()


async def identify_character_gender_and_age_using_llm_and_assign_score(
    character_name, index, lines
):
    """
    Identifies a character's gender and age using a Language Model (LLM) and assigns a gender score.

    Args:
        character_name (str): The name or description of the character.
        index (int): The index of the character's dialogue in the `lines` list.
        lines (list): A list of strings representing the text lines (dialogues or descriptions).

    Returns:
        dict: A dictionary containing the character's name, inferred age, inferred gender, and gender score.
              Example: {"name": "John", "age": "adult", "gender": "male", "gender_score": 2}
    """

    try:
        # Extract a window of dialogues around the character's line for context
        character_dialogues = lines[max(0, index - 2) : index + 5]
        text_character_dialogues = "\n".join(character_dialogues)

        no_think_token = check_if_have_to_include_no_think_token()

        # System prompt to guide the LLM in inferring age and gender
        system_prompt = """
        {no_think_token}
        You are an expert in analyzing character names and inferring their gender and age based on the character's name and the text excerpt. Take into consideration the character name and the text excerpt and then assign the age and gender accordingly. 
        For a masculine character return the gender as 'male', for a feminine character return the gender as 'female' and for a character whose gender is neutral/ unknown return gender as 'unknown'. 
        For assigning the age, if the character is a child return the age as 'child', if the character is an adult return the age as 'adult' and if the character is an elderly return the age as 'elderly'.
        Return only the gender and age as the output. Dont give any explanation or doubt. 
        Give the output as a string in the following format:
        Age: <age>
        Gender: <gender>""".format(
            no_think_token=no_think_token
        )

        # User prompt containing the character name and dialogue context
        user_prompt = f"""
        Character Name/ Character Description: {character_name}

        Text Excerpt: {text_character_dialogues}
        """

        # Query the LLM to infer age and gender
        response = await get_current_llm_client().chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        # Extract and clean the LLM's response
        age_and_gender = response.choices[0].message.content
        age_and_gender = age_and_gender.lower().strip()
        split_text = age_and_gender.split("\n")
        age_text = split_text[0]
        gender_text = split_text[1]

        # Parse age and gender from the response
        age = age_text.split(":")[1].strip()
        gender = gender_text.split(":")[1].strip()

        # Default to "adult" if age is unknown or neutral
        if age not in ["child", "adult", "elderly"]:
            age = "adult"

        # Default to "unknown" if gender is unknown or neutral
        if gender not in ["male", "female", "unknown"]:
            gender = "unknown"

        # Assign a gender score based on inferred gender and age
        gender_score = 5  # Default to neutral/unknown

        if gender == "male":
            if age == "child":
                gender_score = 4  # Slightly masculine for male children
            elif age == "adult":
                gender_score = random.choice(
                    [1, 2, 3]
                )  # Mostly to completely masculine for male adults
            elif age == "elderly":
                gender_score = random.choice(
                    [1, 2]
                )  # Mostly to completely masculine for elderly males
        elif gender == "female":
            if age == "child":
                gender_score = 10  # Completely feminine for female children
            elif age == "adult":
                gender_score = random.choice(
                    [7, 8, 9]
                )  # Mostly to completely feminine for female adults
            elif age == "elderly":
                gender_score = random.choice(
                    [6, 7]
                )  # Slightly to moderately feminine for elderly females

        # Compile character information into a dictionary
        character_info = {
            "name": character_name,
            "age": age,
            "gender": gender,
            "gender_score": gender_score,
        }
        return character_info
    except Exception as e:
        print(
            f"Error: {e}. Defaulting to 'adult' age and 'unknown' gender in response."
        )
        traceback.print_exc()
        character_info = {
            "name": character_name,
            "age": "adult",
            "gender": "unknown",
            "gender_score": 5,
        }
        return character_info


async def identify_characters_and_output_book_to_jsonl(
    text: str, protagonist: str, book_title: str
):
    """
    Processes a given text to identify characters, assign gender scores, and output the results to JSONL files.

    This function performs the following steps:
    1. Clears an existing JSONL file for storing speaker-attributed lines.
    2. Identifies characters in the text using Named Entity Recognition (NER) and vector embeddings.
    3. Consolidates duplicate character names using fuzzy matching and embedding similarity.
    4. Identifies main characters based on appearance frequency.
    5. Assigns gender and age scores only to main characters using a Language Model (LLM).
    6. Tracks character relationships using embedding-based context analysis.
    7. Outputs the processed text with speaker attributions to a JSONL file.
    8. Saves the character gender and age scores and relationship data to separate JSON files.

    Args:
        text (str): The input text to be processed, typically a book or script.
        protagonist: The main character of the text, used as a reference for speaker identification.
        book_title (str): The title of the book, used for constructing file paths.

    Outputs:
        - speaker_attributed_book.jsonl: A JSONL file where each line contains a speaker and their corresponding dialogue or narration.
        - character_gender_map.json: A JSON file containing gender and age scores for each character.
        - character_embeddings.json: A JSON file containing character embeddings and relationship data.
    """
    # Ensure the book-specific temporary directory exists
    book_temp_dir = os.path.join(TEMP_DIR, book_title)
    os.makedirs(book_temp_dir, exist_ok=True)

    speaker_file_path = os.path.join(book_temp_dir, "speaker_attributed_book.jsonl")
    character_map_file_path = os.path.join(book_temp_dir, "character_gender_map.json")
    embeddings_file_path = os.path.join(book_temp_dir, "character_embeddings.json")

    # Clear the output JSONL file
    empty_file(speaker_file_path)
    
    # Initialize embedding system for character relationship tracking
    yield "Initializing character embedding system..."
    try:
        embedding_system = CharacterEmbedding()
        relationship_graph = CharacterRelationshipGraph(embedding_system)
        yield "✅ Character embedding system initialized successfully"
    except Exception as e:
        yield f"⚠️ Warning: Could not initialize embedding system ({e}). Falling back to traditional methods."
        embedding_system = None
        relationship_graph = None

    yield ("Identifying Characters. Progress 0%")

    # Initialize tracking variables
    all_detected_characters = []
    character_contexts = {}
    
    # Define a mapping for character gender scores and initialize with the narrator
    character_gender_map = {
        "legend": {
            "1": "completely masculine",
            "2": "mostly masculine",
            "3": "moderately masculine",
            "4": "slightly masculine",
            "5": "neutral/unknown",
            "6": "slightly feminine",
            "7": "moderately feminine",
            "8": "mostly feminine",
            "9": "almost completely feminine",
            "10": "completely feminine",
        },
        "scores": {
            "narrator": {
                "name": "narrator",
                "age": "adult",
                "gender": "female",  # or male based on the user's selection in audiobook generation step
                "gender_score": -1,  # Special score for narrator (not in voice map)
            }
        },
    }

    # Split the text into lines and extract dialogues
    lines = text.split("\n")
    dialogues = extract_dialogues(text)
    prev_speaker = "narrator"  # Track the previous speaker
    line_map: list[dict] = []  # Store speaker-attributed lines
    dialogue_last_index = 0  # Track the last processed dialogue index

    # Pass 1: Extract all named entities using GLiNER
    yield "Pass 1: Extracting named entities with GLiNER..."
    
    with tqdm(
        total=len(lines),
        unit="line",
        desc="Extracting named entities: ",
    ) as pbar1:
        for index, line in enumerate(lines):
            try:
                # Skip empty lines
                if not line:
                    continue

                # Check if the line contains a dialogue
                dialogue = None
                for dialogue_index in range(dialogue_last_index, len(dialogues)):
                    dialogue_inner = dialogues[dialogue_index]
                    if dialogue_inner in line:
                        dialogue_last_index = dialogue_index
                        dialogue = dialogue_inner
                        break

                # If the line contains a dialogue, identify the speaker
                if dialogue:
                    speaker = identify_speaker_using_named_entity_recognition(
                        line_map,
                        index,
                        line,
                        prev_speaker,
                        protagonist,
                        character_gender_map,
                        None,  # No known characters yet
                        embedding_system,
                        relationship_graph,
                    )
                    
                    # Filter out invalid character names (generic descriptors, etc.)
                    if is_valid_character_name(speaker):
                        all_detected_characters.append(speaker)
                        if speaker not in character_contexts:
                            character_contexts[speaker] = []
                        character_contexts[speaker].append(line)
                        
                        # Add to line map as-is for now
                        line_map.append({"speaker": speaker, "line": line})
                        prev_speaker = speaker
                    else:
                        line_map.append({"speaker": "narrator", "line": line})
                        prev_speaker = "narrator"
                else:
                    # If no dialogue, attribute the line to the narrator
                    line_map.append({"speaker": "narrator", "line": line})
                    prev_speaker = "narrator"

                # Update the progress bar
                pbar1.update(1)

            except Exception as e:
                # Handle errors and log them
                print(f"!!! Error !!! Index: {index}, Error: ", e)
                traceback.print_exc()

    yield f"Found {len(set(all_detected_characters))} unique character entities"

    # Pass 2: Cluster similar characters using embeddings
    yield "Pass 2: Clustering similar characters using embeddings..."
    
    if embedding_system and all_detected_characters:
        # Pre-filter: Only cluster characters that appear multiple times
        char_counts = Counter(all_detected_characters)
        frequent_characters = [char for char, count in char_counts.items() if count >= 2]
        unique_characters = list(set(frequent_characters))
        
        yield f"Clustering {len(unique_characters)} frequent characters (filtered from {len(set(all_detected_characters))} total)"
        
        if len(unique_characters) > 0:
            clustered_characters, character_mapping = cluster_characters_by_embedding(
                unique_characters, character_contexts, embedding_system, similarity_threshold=0.7
            )
        else:
            clustered_characters = []
            character_mapping = {}
        
        yield f"Clustered {len(unique_characters)} entities into {len(clustered_characters)} unique characters"
        
        # Show clustering results
        if len(character_mapping) > len(clustered_characters):
            yield "🔄 Character Clustering Results:"
            clusters_shown = {}
            for original, representative in character_mapping.items():
                if representative not in clusters_shown:
                    clusters_shown[representative] = []
                if original != representative:
                    clusters_shown[representative].append(original)
            
            for representative, variations in clusters_shown.items():
                if variations:
                    yield f"  '{representative}' ← {', '.join(variations)}"
    else:
        clustered_characters = list(set(all_detected_characters))
        character_mapping = {char: char for char in clustered_characters}
        yield "⚠️ Embedding clustering unavailable, using original entities"

    # Pass 3: Select top 8 characters and count frequencies with clustering
    yield "Pass 3: Selecting top 8 characters by frequency..."
    
    # Apply character mapping to line_map and count frequencies
    character_frequencies = Counter()
    character_frequencies["narrator"] = 0
    
    for line_data in line_map:
        original_speaker = line_data["speaker"]
        if original_speaker in character_mapping:
            mapped_speaker = character_mapping[original_speaker]
            line_data["speaker"] = mapped_speaker
            character_frequencies[mapped_speaker] += 1
        elif original_speaker == "narrator":
            character_frequencies["narrator"] += 1

    # Select top characters (limited to 8 total including narrator)
    main_characters = select_top_characters(character_frequencies, protagonist, max_characters=8)
    
    # Debug: Send character frequencies to UI
    yield "📊 Character Frequencies After Clustering:"
    for char, freq in character_frequencies.most_common(10):  # Limit to top 10 for UI
        if char.lower() in main_characters:
            status = "✅ Main Character"
        else:
            status = "❌ Minor Character"
        yield f"  {char}: {freq} appearances - {status}"
    
    yield f"Selected {len(main_characters)} main characters (including narrator)"

    # Unload GLiNER model to free memory (no longer needed)
    yield "Unloading GLiNER model to free memory..."
    global gliner_model
    try:
        del gliner_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (NameError, UnboundLocalError):
        pass  # Model already unloaded

    # Pass 4: Assign voices to main characters
    yield "Pass 4: Assigning voices to main characters..."
    
    with tqdm(
        total=len(main_characters) - 1,  # -1 for narrator
        unit="character",
        desc="Assigning voices: ",
    ) as pbar2:
        for char_name in main_characters:
            if char_name != "narrator":
                # Find a sample line for this character
                sample_index = 0
                for i, line_data in enumerate(line_map):
                    if line_data["speaker"].lower() == char_name.lower():
                        sample_index = i
                        break
                
                # Get character info using LLM
                character_gender_map["scores"][char_name] = (
                    await identify_character_gender_and_age_using_llm_and_assign_score(
                        char_name, sample_index, lines
                    )
                )
                
                pbar2.update(1)
                yield f"Processed character: {char_name}"

    # Pass 5: Consolidate minor characters to narrator
    yield "Pass 5: Consolidating minor characters to narrator..."
    
    consolidated_count = 0
    for line_data in line_map:
        speaker = line_data["speaker"]
        if speaker.lower() not in main_characters and speaker != "narrator":
            line_data["speaker"] = "narrator"
            consolidated_count += 1
    
    yield f"Consolidated {consolidated_count} minor character lines to narrator"

    # Write the processed lines to a JSONL file
    write_jsons_to_jsonl_file(line_map, speaker_file_path)

    # Write the character gender and age scores to a JSON file
    write_json_to_file(character_gender_map, character_map_file_path)

    # Save embedding data if available
    if embedding_system:
        try:
            embedding_system.save_embeddings(embeddings_file_path)
            yield "✅ Character embeddings and relationship data saved"
            
            # Generate relationship summary
            yield "📊 Character Relationship Analysis:"
            for char in main_characters:
                if char != "narrator":
                    relationships = embedding_system.get_character_relationships(char, min_strength=0.3)
                    if relationships:
                        relationship_summary = ", ".join([f"{other}({strength:.2f})" for other, strength in relationships[:3]])
                        yield f"  {char} → {relationship_summary}"
        except Exception as e:
            yield f"⚠️ Warning: Could not save embedding data ({e})"

    yield f"✅ Character Identification Completed!\n📊 Summary:\n  • {len(main_characters)} main characters selected (max 8)\n  • {consolidated_count} minor character lines consolidated to narrator\n  • Character clustering and relationship analysis completed\n  • Ready for audiobook generation!"


async def process_book_and_identify_characters(book_title):
    converted_book_path = f"{TEMP_DIR}/{book_title}/converted_book.txt"
    is_llm_up, message = await check_if_llm_is_up(get_current_llm_client(), model_name)

    if not is_llm_up:
        raise Exception(message)

    yield "Finding protagonist. Please wait..."
    protagonist = await find_book_protagonist(
        book_title, get_current_llm_client(), model_name
    )
    if not os.path.exists(converted_book_path):
        raise Exception(f"Converted book not found at {converted_book_path}")
    f = open(converted_book_path, "r", encoding="utf-8")
    book_text = f.read()
    yield f"Found protagonist: {protagonist}"
    await asyncio.sleep(1)

    async for update in identify_characters_and_output_book_to_jsonl(
        book_text, protagonist, book_title
    ):
        yield update


async def main(book_title: str):
    converted_book_path = f"{TEMP_DIR}/{book_title}/converted_book.txt"
    if not os.path.exists(converted_book_path):
        raise Exception(f"Converted book not found at {converted_book_path}")
    f = open(converted_book_path, "r", encoding="utf-8")
    book_text = f.read()
    f.close()

    # Ask for the protagonist's name
    print("\n📖 **Character Identification Setup**")
    protagonist = input(
        "🔹 Enter the name of the **protagonist** (Check from Wikipedia if needed): "
    ).strip()

    # Start processing
    start_time = time.time()
    print("\n🔍 Identifying characters and processing the book...")
    async for update in identify_characters_and_output_book_to_jsonl(
        book_text, protagonist, book_title
    ):
        print(update)
    end_time = time.time()

    # Calculate execution time
    execution_time = end_time - start_time
    print(f"\n⏱️ **Execution Time:** {execution_time:.6f} seconds")

    # Completion message
    print("\n✅ **Character identification complete!**")
    print("🎧 Next, run the following script to generate the audiobook:")
    print("   ➜ `python generate_audiobook.py`")
    print("\n🚀 Happy audiobook creation!\n")


if __name__ == "__main__":
    asyncio.run(main())
