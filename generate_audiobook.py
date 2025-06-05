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

import shutil
from openai import AsyncOpenAI
from tqdm import tqdm
import json
import os
import asyncio
import re
from word2number import w2n
import time
import sys
from config.constants import (
    TTS_API_KEY,
    API_OUTPUT_FORMAT,
    TTS_BASE_URL,
    TTS_MODEL,
    MAX_PARALLEL_REQUESTS_BATCH_SIZE,
    TEMP_DIR,
    get_current_tts_client,
)
from utils.check_tts_api import check_tts_api
from utils.run_shell_commands import (
    check_if_ffmpeg_is_installed,
    check_if_calibre_is_installed,
)
from utils.file_utils import  concatenate_wav_files,  empty_directory
from utils.audiobook_utils import (
    add_silence_to_audio_file_by_appending_silence_file,
    merge_chapters_to_m4b,
    convert_audio_file_formats,
    merge_chapters_to_standard_audio_file,
)
from utils.check_tts_api import check_tts_api
from dotenv import load_dotenv
import subprocess
import random
from utils.task_utils import (
    update_task_status,
    is_task_cancelled,
    get_task_progress_index,
    set_task_progress_index,
)
from utils.tts_api import generate_tts_with_retry, select_tts_voice

load_dotenv()


os.makedirs("audio_samples", exist_ok=True)



def sanitize_filename(text):
    # Remove or replace problematic characters
    text = text.replace("'", "").replace('"', "").replace("/", " ").replace(".", " ")
    text = text.replace(":", "").replace("?", "").replace("\\", "").replace("|", "")
    text = text.replace("*", "").replace("<", "").replace(">", "").replace("&", "and")

    # Normalize whitespace and trim
    text = " ".join(text.split())

    return text


def sanitize_book_title_for_filename(book_title):
    """Sanitize book title to be safe for filesystem use"""
    safe_title = "".join(
        c for c in book_title if c.isalnum() or c in (" ", "-", "_")
    ).rstrip()
    return safe_title or "audiobook"  # fallback if title becomes empty


def split_and_annotate_text(text):
    """Splits text into dialogue and narration while annotating each segment."""
    parts = re.split(r'("[^"]+")', text)  # Keep dialogues in the split result
    annotated_parts = []

    for part in parts:
        if part:  # Ignore empty strings
            annotated_parts.append(
                {
                    "text": part,
                    "type": (
                        "dialogue"
                        if part.startswith('"') and part.endswith('"')
                        else "narration"
                    ),
                }
            )

    return annotated_parts


def check_if_chapter_heading(text):
    """
    Checks if a given text line represents a chapter heading.

    A chapter heading is considered a string that starts with either "Chapter",
    "Part", or "PART" (case-insensitive) followed by a number (either a digit
    or a word that can be converted to an integer).

    :param text: The text to check
    :return: True if the text is a chapter heading, False otherwise
    """
    pattern = r"^(Chapter|Part|PART)\s+([\w-]+|\d+)"
    regex = re.compile(pattern, re.IGNORECASE)
    match = regex.match(text)

    if match:
        label, number = match.groups()
        try:
            # Try converting the number (either digit or word) to an integer
            w2n.word_to_num(number) if not number.isdigit() else int(number)
            return True
        except ValueError:
            return False  # Invalid number format
    return False  # No match


def concatenate_chapters(
    chapter_files, book_title, chapter_line_map, temp_line_audio_dir
):
    """
    Concatenates the chapters into a single audiobook file.
    Returns a list of full paths to the assembled chapter audio files.
    """
    # Third pass: Concatenate audio files for each chapter in order
    chapter_assembly_bar = tqdm(
        total=len(chapter_files), unit="chapter", desc="Assembling Chapters"
    )

    def assemble_single_chapter(chapter_filename_simple):  # Renamed for clarity
        # Create a temporary file list for this chapter's lines
        chapter_lines_list = os.path.join(
            f"{TEMP_DIR}/{book_title}",
            f"chapter_lines_list_{chapter_filename_simple.replace('/', '_').replace('.', '_')}.txt",
        )

        # Delete the chapter_lines_list file if it exists
        if os.path.exists(chapter_lines_list):
            os.remove(chapter_lines_list)

        with open(chapter_lines_list, "w", encoding="utf-8") as f:
            for line_index in sorted(chapter_line_map[chapter_filename_simple]):
                line_audio_path = os.path.join(
                    temp_line_audio_dir, f"line_{line_index:06d}.{API_OUTPUT_FORMAT}"
                )
                # Use absolute path to prevent path duplication issues
                f.write(f"file '{os.path.abspath(line_audio_path)}'\n")

        output_chapter_full_path = os.path.join(
            TEMP_DIR, book_title, chapter_filename_simple
        )

        # Convert WAV input files to M4A output using AAC codec
        ffmpeg_cmd = f'ffmpeg -y -f concat -safe 0 -i "{chapter_lines_list}" -c:a aac -b:a 256k -avoid_negative_ts make_zero -fflags +genpts -threads 0 "{output_chapter_full_path}"'

        try:
            result = subprocess.run(
                ffmpeg_cmd, shell=True, check=True, capture_output=True, text=True
            )
            # print(f"[DEBUG] FFmpeg stdout: {result.stdout}") # Usually too verbose
            # print(f"[DEBUG] FFmpeg stderr: {result.stderr}") # Usually too verbose unless debugging
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] FFmpeg failed for {chapter_filename_simple}:")
            print(f"[ERROR] Command: {ffmpeg_cmd}")
            print(f"[ERROR] Stdout: {e.stdout}")
            print(f"[ERROR] Stderr: {e.stderr}")
            raise e

        print(f"Assembled chapter: {output_chapter_full_path}") 

        os.remove(chapter_lines_list)
        return output_chapter_full_path  # Return the full path

   
    import concurrent.futures

    max_workers = min(
        4, os.cpu_count() or 1, len(chapter_files)
    )  
    if (
        max_workers == 0 and len(chapter_files) > 0
    ):  
        max_workers = 1

    assembled_chapter_full_paths_ordered = [None] * len(chapter_files)

    if not chapter_files:  
        chapter_assembly_bar.close()
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map futures to their original index to ensure order is preserved
        future_to_index = {
            executor.submit(assemble_single_chapter, chapter_files[i]): i
            for i in range(len(chapter_files))
        }

        for future in concurrent.futures.as_completed(future_to_index):
            original_index = future_to_index[future]
            try:
                processed_full_path = future.result()
                assembled_chapter_full_paths_ordered[original_index] = (
                    processed_full_path
                )
                chapter_assembly_bar.update(1)
            except Exception as e:
                failed_chapter_name = "unknown"
                if original_index < len(chapter_files):
                    failed_chapter_name = chapter_files[original_index]
                print(f"Error assembling chapter {failed_chapter_name}: {e}")
                chapter_assembly_bar.close()  
                raise e

    chapter_assembly_bar.close()
    return assembled_chapter_full_paths_ordered 


async def parallel_post_processing(chapter_full_paths, book_title, output_format):
    """
    Parallel post-processing of chapter files (given as full paths)
    to add silence and convert formats.
    Returns a list of simple filenames of the processed M4A files,
    which reside in TEMP_DIR/book_title/.
    """

    

    import concurrent.futures

    max_workers = min(4, os.cpu_count() or 1, len(chapter_full_paths))
    if max_workers == 0 and len(chapter_full_paths) > 0:
        max_workers = 1

    # Initialize list to store results (simple filenames) in the correct order
    processed_m4a_simple_filenames_ordered = [None] * len(chapter_full_paths)

    post_processing_bar = tqdm(
        total=len(chapter_full_paths), unit="chapter", desc="Post Processing (Parallel)"
    )

    if not chapter_full_paths:  # Handle empty list gracefully
        post_processing_bar.close()
        return []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map futures to their original index to ensure order is preserved
        future_to_index = {
            executor.submit(add_silence_to_audio_file_by_appending_silence_file, chapter_full_paths[i]): i
            for i in range(len(chapter_full_paths))
        }

        for future in concurrent.futures.as_completed(future_to_index):
            original_index = future_to_index[future]
            try:
                processed_full_path = future.result()
                # Extract just the filename from the full path
                processed_simple_filename = os.path.basename(processed_full_path)
                processed_m4a_simple_filenames_ordered[original_index] = (
                    processed_simple_filename
                )
                post_processing_bar.update(1)
            except Exception as e:
                failed_chapter_name = "unknown"
                if original_index < len(chapter_full_paths):
                    failed_chapter_name = os.path.basename(
                        chapter_full_paths[original_index]
                    )
                print(
                    f"Error in post-processing for chapter {failed_chapter_name}: {e}"
                )
                post_processing_bar.close()  # Ensure bar is closed on error
                raise e

    post_processing_bar.close()

    return processed_m4a_simple_filenames_ordered


def find_voice_for_gender_score(character: str, character_gender_map, voice_map):
    """
    Finds the appropriate voice for a character based on their gender score.

    This function takes in the name of a character, a dictionary mapping character names to their gender scores,
    and a dictionary mapping voice identifiers to gender scores. It returns the voice identifier that matches the
    character's gender score.

    Args:
        character (str): The name of the character for whom the voice is being determined.
        character_gender_map (dict): A dictionary mapping character names to their gender scores.
        voice_map (dict): A dictionary mapping voice identifiers to gender scores.

    Returns:
        str: The voice identifier that matches the character's gender score.
    """

    try:
        # Get the character's gender score
        character_lower = character.lower()

        # Check if character exists in the character_gender_map
        if character_lower not in character_gender_map["scores"]:
            print(
                f"WARNING: Character '{character}' not found in character_gender_map. Using narrator voice as fallback."
            )
            # Use narrator's voice as fallback
            if "narrator" in character_gender_map["scores"]:
                character_gender_score_doc = character_gender_map["scores"]["narrator"]
                character_gender_score = character_gender_score_doc["gender_score"]
            else:
                print(
                    f"ERROR: Even narrator not found in character_gender_map. Using score 5 (neutral)."
                )
                character_gender_score = 5
        else:
            character_gender_score_doc = character_gender_map["scores"][character_lower]
            character_gender_score = character_gender_score_doc["gender_score"]

        # Iterate over the voice identifiers and their scores
        for voice, score in voice_map.items():
            # Find the voice identifier that matches the character's gender score
            if score == character_gender_score:
                return voice

        # If no exact match found, find the closest gender score
        print(
            f"WARNING: No exact voice match for character '{character}' with gender score {character_gender_score}. Finding closest match."
        )

        closest_voice = None
        closest_diff = float("inf")

        for voice, score in voice_map.items():
            diff = abs(score - character_gender_score)
            if diff < closest_diff:
                closest_diff = diff
                closest_voice = voice

        if closest_voice:
            print(
                f"Using voice '{closest_voice}' (score {voice_map[closest_voice]}) for character '{character}' (score {character_gender_score})"
            )
            return closest_voice

        # Final fallback: use the first available voice
        if voice_map:
            fallback_voice = list(voice_map.keys())[0]
            print(
                f"ERROR: Could not find suitable voice for character '{character}'. Using fallback voice '{fallback_voice}'."
            )
            return fallback_voice

        # Absolute fallback for empty voice_map
        print(f"CRITICAL ERROR: voice_map is empty. Using hardcoded fallback voice.")
        return "af_heart"  # Default fallback voice

    except Exception as e:
        print(f"ERROR in find_voice_for_gender_score for character '{character}': {e}")
        print("Using hardcoded fallback voice.")
        return "af_heart"  # Default fallback voice


def preprocess_text_for_orpheus(text):
    """
    Preprocess text for Orpheus TTS to prevent repetition issues.
    Adds full stops where necessary while handling edge cases.
    """
    if not text or len(text.strip()) == 0:
        return text

    text = text.strip()

    # Don't modify very short text (single words or very short phrases)
    if len(text) <= 3:
        return text

    # Check if text already ends with proper punctuation
    punctuation_marks = {".", "!", "?", ":", ";", ",", '"', "'", ")", "]", "}"}
    if text[-1] in punctuation_marks:
        return text

    # Handle dialogue - don't add period inside quotes
    if text.startswith('"') and text.endswith('"'):
        # For dialogue, check if there's already punctuation before the closing quote
        if len(text) > 2 and text[-2] in {".", "!", "?", ",", ";", ":"}:
            return text
        else:
            # Add period before closing quote
            return text[:-1] + '."'

    # Handle text that ends with quotes but doesn't start with them
    if text.endswith('"') and not text.startswith('"'):
        # Check if there's punctuation before the quote
        if len(text) > 1 and text[-2] in {".", "!", "?", ",", ";", ":"}:
            return text
        else:
            # Add period before the quote
            return text[:-1] + '."'

    # For regular narration text, add a period
    return text + "."


async def generate_audio_files(
    output_format,
    narrator_gender,
    book_path="",
    book_title="audiobook",
    type="single_voice",
    task_id=None,
):
    # Read the text from the file
    """
    Generate an audiobook using a single voice for narration and dialogues or multiple voices for multi-voice lines.

    This asynchronous function reads text from a file, processes each line to determine
    if it is narration or dialogue, and generates corresponding audio using specified
    voices. The generated audio is organized by chapters, with options to create
    an M4B audiobook file or a standard audio file in the specified output format.

    Args:
        output_format (str): The desired output format for the final audiobook (e.g., "mp3", "wav").
        narrator_gender (str): The gender of the narrator ("male" or "female") to select appropriate voices.
        book_path (str, optional): The file path for the book to be used in M4B creation. Defaults to an empty string.

    Yields:
        str: Progress updates as the audiobook generation progresses through loading text, generating audio,
             organizing by chapters, assembling chapters, and post-processing steps.
    """

    # Check if converted book exists, if not, process the book first
    converted_book_path = f"{TEMP_DIR}/{book_title}/converted_book.txt"
    if not os.path.exists(converted_book_path):
        yield "Converting book to text format..."
        # Import and use the book processing function
        from book_to_txt import process_book_and_extract_text

        # Create the temp directory structure
        os.makedirs(f"{TEMP_DIR}/{book_title}", exist_ok=True)

        # Process the book and extract text
        for text in process_book_and_extract_text(book_path, "textract", book_title):
            pass 
        yield "Book conversion completed"

    lines_to_process = []

    # Import the improved voice selection utilities
    from utils.select_voice import select_voice

    # Setup for multi-voice lines
    character_gender_map = None
    voice_map = None
    narrator_voice = ""
    dialogue_voice = ""
    
    if type.lower() == "multi_voice":
        print(f"Processing multi-voice lines")

        try:
            voice_config = select_voice(narrator_gender, TTS_MODEL, "multi_voice", book_title)
            speaker_file_path = voice_config["speaker_file_path"]
            narrator_voice = voice_config["narrator_voice"]
            
            # Load character gender map and voice map once for multi-voice mode
            with open(voice_config["character_map_path"], "r", encoding="utf-8") as f:
                character_gender_map = json.load(f)
            with open(voice_config["voice_map_path"], "r", encoding="utf-8") as f:
                voice_map = json.load(f)
            
            with open(speaker_file_path, "r", encoding="utf-8") as file:
                for line in file:
                    # Parse each line as a JSON object
                    json_object = json.loads(line.strip())
                    # Append the parsed JSON object to the array
                    lines_to_process.append(json_object)

            yield "Loaded speaker-attributed lines from JSONL file"

        except (FileNotFoundError, ValueError) as e:
            yield f"Error: {str(e)}"
            return

        total_lines = len(
            lines_to_process
        )  

        yield "Loaded voice mappings and selected narrator voice"

    else:
        # handle single voice
        try:
            voice_config = select_voice(narrator_gender, TTS_MODEL, "single_voice", book_title)
            narrator_voice = voice_config["narrator_voice"]
            dialogue_voice = voice_config["dialogue_voice"]
            
            with open(converted_book_path, "r", encoding="utf-8") as f:
                text = f.read()
                lines_to_process = text.split("\n")
                lines_to_process = [line.strip() for line in lines_to_process if line.strip()]

        except (FileNotFoundError, ValueError) as e:
            yield f"Error: {str(e)}"
            return

        total_lines = len(
            lines_to_process
        )  

    # Setup directories
    temp_line_audio_dir = os.path.join(TEMP_DIR, book_title, "line_segments")

    # restart from the last line if the directory exists or create a new directory
    if os.path.exists(temp_line_audio_dir):
        resume_index, _ = get_task_progress_index(task_id)
        print(f"Resuming from line {resume_index}")
    else:
        os.makedirs(TEMP_DIR, exist_ok=True)
        empty_directory(os.path.join(temp_line_audio_dir, book_title))
        os.makedirs(temp_line_audio_dir, exist_ok=True)

    # Batch processing parameters
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS_BATCH_SIZE)
    print(f"Starting task {task_id}")

    # Initial setup for chapters
    chapter_index = 1

    current_chapter_audio = "Introduction.m4a"
    chapter_files = []
    resume_index = 0
    if task_id:
        resume_index, _ = get_task_progress_index(task_id)
        print(f"Resuming from line {resume_index}")

    progress_counter = 0
    progress_lock = asyncio.Lock()  

    progress_bar = tqdm(
        total=total_lines, unit="line", desc="Audio Generation Progress"
    )

    # Maps chapters to their line indices
    chapter_line_map = {}

    async def update_progress_and_task_status(
        line_index,
        actual_text_content,
    ):
        nonlocal progress_counter
        async with progress_lock:
            progress_bar.update(1)
            progress_counter = progress_counter + 1
            update_task_status(
                task_id,
                "generating",
                f"Generating audiobook. Progress: {progress_counter}/{total_lines}",
            )
            set_task_progress_index(task_id, progress_counter, total_lines)

            return {
                "index": line_index,
                "is_chapter_heading": check_if_chapter_heading(actual_text_content),
                "line": actual_text_content,  # Return the processed text content
            }

    async def process_single_line(
        line_index,
        line, 
        type="single_voice",
    ):
        async with semaphore:
            nonlocal progress_counter

            actual_text_content = ""
            speaker_name = ""

            if type.lower() == "multi_voice":
                actual_text_content = line.get("line", "").strip()
                speaker_name = line.get("speaker", "").strip()
            else:  
                actual_text_content = str(line).strip()

            annotated_parts = split_and_annotate_text(actual_text_content)
            audio_parts = []
            line_audio_path = os.path.join(
                temp_line_audio_dir, f"line_{line_index:06d}.{API_OUTPUT_FORMAT}"
            )
            # if the line audio file exists and is not empty, we can skip the line
            if (
                os.path.exists(line_audio_path)
                and os.path.getsize(line_audio_path) > 1024
            ):
                return await update_progress_and_task_status(
                    line_index, actual_text_content
                )

            try:
                for i, part in enumerate(annotated_parts):
                    text_to_speak = part["text"].strip()

                    if task_id and is_task_cancelled(task_id):
                        print(
                            f"[DEBUG] Task {task_id} cancelled before processing line {line_index}, part {i}"
                        )
                        raise asyncio.CancelledError("Task was cancelled by user")

                    if TTS_MODEL == "orpheus":
                        # add full stops where necessary
                        text_to_speak = preprocess_text_for_orpheus(text_to_speak)

                    voice = ""
                    if type.lower() == "multi_voice":
                        # For multi-voice mode, use speaker-based voice selection
                        if speaker_name and speaker_name.lower() == "narrator":
                            voice = narrator_voice
                        elif speaker_name and character_gender_map and voice_map:
                            voice = find_voice_for_gender_score(speaker_name, character_gender_map, voice_map)
                        else:
                            # Fallback to narrator voice if speaker not found
                            voice = narrator_voice
                            print(f"[DEBUG] Using narrator voice fallback for speaker: {speaker_name}")
                    else:
                        # For single-voice mode, use narration/dialogue logic
                        if part["type"] == "narration":
                            voice = narrator_voice
                        else:
                            voice = dialogue_voice

                    try:

                        current_part_audio_buffer = await generate_tts_with_retry(
                            TTS_MODEL,
                            voice,  
                            text_to_speak,
                            API_OUTPUT_FORMAT,
                            speed=0.85,
                            max_retries=5,
                            task_id=task_id,
                        )

                        part_file_path = os.path.join(
                            temp_line_audio_dir,
                            f"line_{line_index:06d}_part_{i}.{API_OUTPUT_FORMAT}",
                        )

                        with open(part_file_path, "wb") as part_file:
                            part_file.write(current_part_audio_buffer)
                        audio_parts.append(part_file_path)
                        print(
                            f"[DEBUG] Created part file: {part_file_path} ({len(current_part_audio_buffer)} bytes)"
                        )
                    except asyncio.CancelledError:
                        # Clean up any created files and remove final line file before re-raising
                        for part_file in audio_parts:
                            if os.path.exists(part_file):
                                os.remove(part_file)
                        if os.path.exists(line_audio_path):
                            os.remove(line_audio_path)
                        raise
                    except Exception as e:

                        # Clean up any created files and remove final line file before re-raising
                        for part_file in audio_parts:
                            if os.path.exists(part_file):
                                os.remove(part_file)
                        if os.path.exists(line_audio_path):
                            os.remove(line_audio_path)
                        raise e

            except Exception as e:
                # Clean up any created files and remove final line file before re-raising
                for part_file in audio_parts:
                    if os.path.exists(part_file):
                        os.remove(part_file)
                if os.path.exists(line_audio_path):
                    os.remove(line_audio_path)
                print(f"ERROR processing line {line_index}: {e}")
                raise e

            if audio_parts:
                concatenate_wav_files(audio_parts, line_audio_path)
                # Clean up individual part files after successful concatenation
                for part_file in audio_parts:
                    if os.path.exists(part_file):
                        os.remove(part_file)
                        print(f"[DEBUG] Cleaned up part file: {part_file}")
            else:
                print(f"WARNING: Line {line_index} resulted in no valid audio parts.")
                # Create an empty file to mark this line as processed
                with open(line_audio_path, "wb") as f:
                    f.write(b"")

            return await update_progress_and_task_status(
                line_index, actual_text_content
            )

    # Create tasks and store them with their index for result collection
    tasks = []
    task_to_index = {}
    for i, line_content in enumerate(lines_to_process):  # Iterate over lines_to_process
        if i < resume_index:
            # Already processed, skip
            continue

        task = asyncio.create_task(
            process_single_line(i, line_content, type)
        )  # Pass line_content
        tasks.append(task)
        task_to_index[task] = i

    # Initialize results_all list
    results_all = [None] * total_lines  

    # Create a cancellation monitor task
    async def cancellation_monitor():
        while tasks:
            await asyncio.sleep(0.5)  # Check every 500ms
            if task_id and is_task_cancelled(task_id):
                print(
                    f"[DEBUG] Cancellation monitor detected task {task_id} is cancelled"
                )
                # Cancel all remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                        print(f"[DEBUG] Cancellation monitor cancelled a task")
                # clean up the temp directory
                temp_book_dir = f"{TEMP_DIR}/{book_title}"
                if os.path.exists(temp_book_dir):
                    try:
                        shutil.rmtree(temp_book_dir)
                        print(f"[DEBUG] Cleaned up temp directory: {temp_book_dir}")
                    except Exception as e:
                        print(f"[DEBUG] Error cleaning up temp directory: {e}")
                break

    # Start the cancellation monitor
    monitor_task = asyncio.create_task(cancellation_monitor())

    # Process tasks with progress updates and retry logic
    last_reported = -1

    while tasks:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        # Store results as tasks complete
        for completed_task in done:
            idx = task_to_index[completed_task]
            results_all[idx] = completed_task.result()

        tasks = list(pending)

        # Only yield if the counter has changed
        if progress_counter > last_reported:
            last_reported = progress_counter
            percent = (progress_counter / total_lines) * 100

            # Check if task has been cancelled
            if task_id and is_task_cancelled(task_id):
                print(
                    f"[DEBUG] Task {task_id} was cancelled, cancelling all pending tasks"
                )
                # Cancel all pending tasks immediately
                for task in tasks:
                    if not task.done():
                        task.cancel()
                        print(f"[DEBUG] Cancelled pending task")
                raise asyncio.CancelledError("Task was cancelled by user")

            # update the task status
            update_task_status(
                task_id,
                "generating",
                f"Generating audiobook. Progress: {percent:.1f}%",
            )
            yield f"Generating audiobook. Progress: {percent:.1f}%"

    # All tasks have completed at this point and results_all is populated
    results = [r for r in results_all if r is not None]  # Filter out empty lines

    # Clean up the monitor task
    if not monitor_task.done():
        monitor_task.cancel()

    progress_bar.update(total_lines)
    progress_bar.close()

    results = [r for r in results_all if r is not None]

    yield f"Completed generating audio for {len(results)}/{total_lines} lines"

    # Validate all audio files exist before proceeding to concatenation
    print("Validating audio files before concatenation...")
    missing_files = []
    for result in results:
        line_idx = result["index"]
        final_line_path = os.path.join(
            temp_line_audio_dir,
            f"line_{line_idx:06d}.{API_OUTPUT_FORMAT}",
        )
        if not os.path.exists(final_line_path) or os.path.getsize(final_line_path) == 0:
            missing_files.append(line_idx)

    if missing_files:
        print(
            f"ERROR: {len(missing_files)} audio files are missing or empty: {missing_files[:10]}..."
        )
        raise Exception(
            f"Cannot proceed with concatenation - {len(missing_files)} audio files are missing"
        )

    print(f"✅ All {len(results)} audio files validated successfully")

    # Second pass: Organize by chapters
    chapter_organization_bar = tqdm(
        total=len(results), unit="result", desc="Organizing Chapters"
    )

    for result in sorted(results, key=lambda x: x["index"]):
        # Check if this is a chapter heading
        if result["is_chapter_heading"]:
            chapter_index += 1

            # Always assemble chapters as M4A files first, regardless of final output format
            current_chapter_audio = f"{sanitize_filename(result['line'])}.m4a"

        if current_chapter_audio not in chapter_files:
            chapter_files.append(current_chapter_audio)
            chapter_line_map[current_chapter_audio] = []

        # Add this line index to the chapter
        chapter_line_map[current_chapter_audio].append(result["index"])
        chapter_organization_bar.update(1)

    chapter_organization_bar.close()
    yield "Organizing audio by chapters complete"
    # concatenate chapters into m4a files
    chapter_files = concatenate_chapters(
        chapter_files, book_title, chapter_line_map, temp_line_audio_dir
    )
    yield f"Completed concatenating {len(chapter_files)} chapters"

    # Optimized parallel post-processing
    yield "Starting parallel post-processing..."
    chapter_files = await parallel_post_processing(
        chapter_files, book_title, output_format
    )
    yield f"Completed parallel post-processing of {len(chapter_files)} chapters"

    

    # create audiobook directory if it does not exist
    os.makedirs(f"generated_audiobooks", exist_ok=True)

    yield "Creating final audiobook..."
    if output_format == "m4b":
        merge_chapters_to_m4b(book_path, chapter_files, book_title)
    else:
        merge_chapters_to_standard_audio_file(chapter_files, book_title)
    # convert to final output format
        convert_audio_file_formats(
        "m4a", output_format, "generated_audiobooks", book_title
        )

    yield f"Audiobook in {output_format} format created successfully"


async def process_audiobook_generation(
    voice_option,
    narrator_gender,
    output_format,
    book_path,
    book_title="audiobook",
    task_id=None,
):
    # Select narrator voice string based on narrator_gender and TTS_MODEL
    narrator_voice = select_tts_voice(TTS_MODEL, narrator_gender)

    is_tts_api_up, message = await check_tts_api(
        get_current_tts_client(), TTS_MODEL, narrator_voice
    )

    if not is_tts_api_up:
        raise Exception(message)


    

    if voice_option == "Single Voice":
        yield "\n🎧 Generating audiobook with a **single voice**..."
        await asyncio.sleep(1)
        async for line in generate_audio_files(
            output_format.lower(),
            narrator_gender,
            book_path,
            book_title,
            "single_voice",
            task_id,
        ):
            yield line
    elif voice_option == "Multi-Voice":
        yield "\n🎭 Generating audiobook with **multiple voices**..."
        await asyncio.sleep(1)
        async for line in generate_audio_files(
            output_format.lower(),
            narrator_gender,
            book_path,
            book_title,
            "multi_voice",
            task_id,
        ):
            yield line

    yield f"\n🎧 Audiobook is generated ! You can now download it in the Download section below. Click on the blue download link next to the file name."


async def main(
    book_title="audiobook",
    book_path="./sample_book_and_audio/The Adventure of the Lost Treasure - Prakhar Sharma.epub",
    output_format="wav",
    voice_option="single",
    narrator_gender="male",
):
    os.makedirs(f"{TEMP_DIR}/{book_title}/generated_audiobooks", exist_ok=True)

    # Check if we're running with command line arguments (non-interactive mode)
    running_with_args = (
        len(sys.argv) > 1 
    )

    if not running_with_args:
        # Prompt user for voice selection
        print("\n🎙️ **Audiobook Voice Selection**")
        voice_option = input(
            "🔹 Enter **1** for **Single Voice** or **2** for **Multiple Voices**: "
        ).strip()

        # Prompt user for audiobook type selection
        print("\n🎙️ **Audiobook Type Selection**")
        print(
            "🔹 Do you want the audiobook in M4B format (the standard format for audiobooks) with chapter timestamps and embedded book cover ? (Needs calibre and ffmpeg installed)"
        )
        print(
            "🔹 OR do you want a standard audio file in either of ['aac', 'm4a', 'mp3', 'wav', 'opus', 'flac', 'pcm'] formats without any of the above features ?"
        )
        audiobook_type_option = input(
            "🔹 Enter **1** for **M4B audiobook format** or **2** for **Standard Audio File**: "
        ).strip()
    else:
        # Running with arguments, set defaults based on parameters
        if voice_option == "single" or voice_option == "1":
            voice_option = "1"
        else:
            voice_option = "2"
        audiobook_type_option = "1" if output_format == "m4b" else "2"

    if audiobook_type_option == "1":
        is_calibre_installed = check_if_calibre_is_installed()

        if not is_calibre_installed:
            print(
                "⚠️ Calibre is not installed. Please install it first and make sure **calibre** and **ebook-meta** commands are available in your PATH. Defaulting to standard audio file format."
            )

        is_ffmpeg_installed = check_if_ffmpeg_is_installed()

        if not is_ffmpeg_installed:
            print(
                "⚠️ FFMpeg is not installed. Please install it first and make sure **ffmpeg** and **ffprobe** commands are available in your PATH."
            )
            return

        # Check if a path is provided via command-line arguments
        if len(sys.argv) > 1:
            book_path = sys.argv[1]
            print(f"📂 Using book file from command-line argument: **{book_path}**")
        elif not running_with_args:
            # Ask user for book file path if not provided and in interactive mode
            if (
                book_path is None
                or book_path
                == "./sample_book_and_audio/The Adventure of the Lost Treasure - Prakhar Sharma.epub"
            ):
                input_path = input(
                    "\n📖 Enter the **path to the book file**, needed for metadata and cover extraction. (Press Enter to use default): "
                ).strip()
                if input_path:
                    book_path = input_path

        print(f"📂 Using book file: **{book_path}**")

        print("✅ Book path set. Proceeding...\n")

    else:
        # Prompt user for audio format selection
        print("\n🎙️ **Audiobook Output Format Selection**")
        output_format = input(
            "🔹 Choose between ['m4b', 'aac', 'm4a', 'mp3', 'wav', 'opus', 'flac', 'pcm']. "
        ).strip()

        if output_format not in ["m4b", "aac", "m4a", "mp3", "wav", "opus", "flac", "pcm"]:
            print("\n⚠️ Invalid output format! Please choose from the give options")
            return

    if not running_with_args:
        # Prompt user for narrator's gender selection
        print("\n🎙️ **Audiobook Narrator Voice Selection**")
        narrator_gender = input(
            "🔹 Enter **male** if you want the book to be read in a male voice or **female** if you want the book to be read in a female voice: "
        ).strip()

    if narrator_gender not in ["male", "female"]:
        print("\n⚠️ Using default narrator gender: male")
        narrator_gender = "male"

    start_time = time.time()

    if voice_option == "1":
        print("\n🎧 Generating audiobook with a **single voice**...")
        async for line in generate_audio_files(
            output_format,
            narrator_gender,
            book_path,
            book_title,
            "single_voice",
            f"non_interactive_{voice_option}_{narrator_gender}_{output_format}",
        ):
            print(line)
    elif voice_option == "2":
        print("\n🎭 Generating audiobook with **multiple voices**...")
        async for line in generate_audio_files(
            output_format,
            narrator_gender,
            book_path,
            book_title,
            "multi_voice",
            f"non_interactive_{voice_option}_{narrator_gender}_{output_format}",
        ):
            print(line)
    else:
        print("\n⚠️ Invalid option! Please restart and enter either **1** or **2**.")
        return

    print(
        f"\n🎧 Audiobook is generated ! The audiobook is saved as **audiobook.{output_format}** in the **generated_audiobooks** directory in the current folder."
    )

    end_time = time.time()

    execution_time = end_time - start_time
    print(
        f"\n⏱️ **Execution Time:** {execution_time:.6f} seconds\n✅ Audiobook generation complete!"
    )


if __name__ == "__main__":
    asyncio.run(main())
