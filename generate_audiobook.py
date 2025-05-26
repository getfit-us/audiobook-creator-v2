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
    API_KEY,
    BASE_URL,
    MODEL,
    MAX_PARALLEL_REQUESTS_BATCH_SIZE,
    TEMP_DIR,
)
from utils.check_tts_api import check_tts_api
from utils.run_shell_commands import (
    check_if_ffmpeg_is_installed,
    check_if_calibre_is_installed,
)
from utils.file_utils import read_json, empty_directory
from utils.audiobook_utils import (
    merge_chapters_to_m4b,
    convert_audio_file_formats,
    merge_chapters_to_standard_audio_file,
    add_silence_to_audio_file_by_appending_pre_generated_silence,
)
from utils.check_tts_api import check_tts_api
from dotenv import load_dotenv
import subprocess
import random
from utils.task_utils import update_task_status, is_task_cancelled

load_dotenv()


API_OUTPUT_FORMAT = "wav" if MODEL == "orpheus" else "aac"

os.makedirs("audio_samples", exist_ok=True)

async_openai_client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

print(BASE_URL)


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
    """
    # Third pass: Concatenate audio files for each chapter in order
    chapter_assembly_bar = tqdm(
        total=len(chapter_files), unit="chapter", desc="Assembling Chapters"
    )

    def assemble_single_chapter(chapter_file):
        # Create a temporary file list for this chapter's lines
        chapter_lines_list = os.path.join(
            f"{TEMP_DIR}/{book_title}",
            f"chapter_lines_list_{chapter_file.replace('/', '_').replace('.', '_')}.txt",
        )

        # Delete the chapter_lines_list file if it exists
        if os.path.exists(chapter_lines_list):
            os.remove(chapter_lines_list)

        with open(chapter_lines_list, "w", encoding="utf-8") as f:
            for line_index in sorted(chapter_line_map[chapter_file]):
                line_audio_path = os.path.join(
                    temp_line_audio_dir, f"line_{line_index:06d}.{API_OUTPUT_FORMAT}"
                )
                # Use absolute path to prevent path duplication issues
                f.write(f"file '{os.path.abspath(line_audio_path)}'\n")

        # Use FFmpeg to concatenate the lines with optimized parameters
        if MODEL == "orpheus":
            # For Orpheus, convert WAV segments to M4A chapters directly with timestamp filtering
            ffmpeg_cmd = (
                f'ffmpeg -y -f concat -safe 0 -i "{chapter_lines_list}" '
                f'-c:a aac -b:a 256k -ar 44100 -ac 2 -avoid_negative_ts make_zero -fflags +genpts -threads 0 "{TEMP_DIR}/{book_title}/{chapter_file}"'
            )
        else:
            # For other models, use re-encoding with timestamp filtering to prevent truncation
            ffmpeg_cmd = f'ffmpeg -y -f concat -safe 0 -i "{chapter_lines_list}" -c:a aac -b:a 256k -avoid_negative_ts make_zero -fflags +genpts -threads 0 "{TEMP_DIR}/{book_title}/{chapter_file}"'

        try:
            result = subprocess.run(
                ffmpeg_cmd, shell=True, check=True, capture_output=True, text=True
            )
            print(f"[DEBUG] FFmpeg stdout: {result.stdout}")
            print(f"[DEBUG] FFmpeg stderr: {result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] FFmpeg failed for {chapter_file}:")
            print(f"[ERROR] Command: {ffmpeg_cmd}")
            print(f"[ERROR] Stdout: {e.stdout}")
            print(f"[ERROR] Stderr: {e.stderr}")
            raise e

        print(f"Assembled chapter: {chapter_file}")

        # Clean up the temporary file list
        os.remove(chapter_lines_list)
        return chapter_file

    # Process chapters in parallel (limit to 4 concurrent to avoid overwhelming system)
    import concurrent.futures

    max_workers = min(4, len(chapter_files))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(assemble_single_chapter, chapter_file)
            for chapter_file in chapter_files
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                chapter_file = future.result()
                chapter_assembly_bar.update(1)
            except Exception as e:
                print(f"Error assembling chapter: {e}")
                raise e

    chapter_assembly_bar.close()


async def parallel_post_processing(chapter_files, book_title, output_format):
    """
    Parallel post-processing of chapter files to add silence and convert formats.
    """

    def process_single_chapter(chapter_file):
        # Add silence to chapter file
        add_silence_to_audio_file_by_appending_pre_generated_silence(
            f"{TEMP_DIR}/{book_title}", chapter_file, output_format
        )

        # Convert to M4A format if needed
        chapter_name = chapter_file.split(f".{output_format}")[0]
        m4a_chapter_file = f"{chapter_name}.m4a"

        # Only convert if not already in M4A format
        if not chapter_file.endswith(".m4a"):
            convert_audio_file_formats(
                output_format, "m4a", f"{TEMP_DIR}/{book_title}", chapter_name
            )

        return m4a_chapter_file

    # Process chapters in parallel (limit to 4 concurrent to avoid overwhelming system)
    import concurrent.futures

    max_workers = min(4, len(chapter_files))
    m4a_chapter_files = []

    post_processing_bar = tqdm(
        total=len(chapter_files), unit="chapter", desc="Post Processing (Parallel)"
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_chapter, chapter_file)
            for chapter_file in chapter_files
        ]

        for future in concurrent.futures.as_completed(futures):
            try:
                m4a_chapter_file = future.result()
                m4a_chapter_files.append(m4a_chapter_file)
                post_processing_bar.update(1)
            except Exception as e:
                print(f"Error in post-processing: {e}")
                raise e

    post_processing_bar.close()

    # Sort to maintain chapter order
    m4a_chapter_files.sort()
    return m4a_chapter_files


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


async def generate_tts_with_retry(
    client, model, voice, text, response_format, speed=0.85, max_retries=5, task_id=None
):
    """
    Generate TTS audio with retry logic and exponential backoff.

    Args:
        client: AsyncOpenAI client
        model: TTS model to use
        voice: Voice to use
        text: Text to convert to speech
        response_format: Audio format (wav, aac, etc.)
        speed: Speech speed
        max_retries: Maximum number of retry attempts
        task_id: Task ID for cancellation checking

    Returns:
        bytearray: Audio data buffer

    Raises:
        Exception: If all retries are exhausted or task is cancelled
    """
    for attempt in range(max_retries + 1):
        try:
            # Check if task has been cancelled before each attempt
            if task_id and is_task_cancelled(task_id):
                raise asyncio.CancelledError("Task was cancelled by user")

            audio_buffer = bytearray()

            async with client.audio.speech.with_streaming_response.create(
                model=model,
                voice=voice,
                response_format=response_format,
                speed=speed,
                input=text,
            ) as response:
                if response.status_code != 200:
                    error_msg = f"TTS API returned status {response.status_code}"
                    try:
                        error_content = await response.aread()
                        error_msg += f": {error_content.decode()}"
                    except Exception:
                        pass

                    if attempt < max_retries:
                        print(
                            f"Attempt {attempt + 1} failed with status {response.status_code}, retrying..."
                        )
                        continue
                    else:
                        raise Exception(error_msg)

                async for chunk in response.iter_bytes():
                    # Check for cancellation during streaming
                    if task_id and is_task_cancelled(task_id):
                        raise asyncio.CancelledError("Task was cancelled by user")
                    audio_buffer.extend(chunk)

            if len(audio_buffer) > 0:
                return audio_buffer
            else:
                if attempt < max_retries:
                    print(f"Attempt {attempt + 1} returned 0 bytes, retrying...")
                    continue
                else:
                    raise Exception("TTS returned 0 bytes after all retries")

        except asyncio.CancelledError:
            # Don't retry on cancellation
            raise
        except Exception as e:
            error_msg = str(e)

            # Check if this is a retryable error
            retryable_errors = [
                "peer closed connection",
                "connection reset",
                "timeout",
                "network",
                "temporary failure",
                "service unavailable",
                "bad gateway",
                "gateway timeout",
                "connection aborted",
                "connection refused",
                "connection error",
                "read timeout",
                "write timeout",
                "incomplete read",
                "broken pipe",
                "socket error",
                "http error 5",  # 5xx server errors
                "internal server error",
                "server error",
            ]

            is_retryable = any(
                error_phrase in error_msg.lower() for error_phrase in retryable_errors
            )

            if attempt < max_retries and is_retryable:
                # Exponential backoff with jitter
                delay = (2**attempt) + random.uniform(0, 1)
                print(f"Attempt {attempt + 1} failed with retryable error: {error_msg}")
                print(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
                continue
            else:
                # Either max retries reached or non-retryable error
                if attempt >= max_retries:
                    print(f"All {max_retries + 1} attempts failed for TTS generation")
                else:
                    print(f"Non-retryable error: {error_msg}")
                raise e

    # This should never be reached, but just in case
    raise Exception("Unexpected end of retry loop")


async def generate_audio_with_single_voice(
    output_format,
    narrator_gender,
    generate_m4b_audiobook_file=False,
    book_path="",
    book_title="audiobook",
    task_id=None,
):
    # Read the text from the file
    """
    Generate an audiobook using a single voice for narration and dialogues.

    This asynchronous function reads text from a file, processes each line to determine
    if it is narration or dialogue, and generates corresponding audio using specified
    voices. The generated audio is organized by chapters, with options to create
    an M4B audiobook file or a standard audio file in the specified output format.

    Args:
        output_format (str): The desired output format for the final audiobook (e.g., "mp3", "wav").
        narrator_gender (str): The gender of the narrator ("male" or "female") to select appropriate voices.
        generate_m4b_audiobook_file (bool, optional): Flag to determine whether to generate an M4B file. Defaults to False.
        book_path (str, optional): The file path for the book to be used in M4B creation. Defaults to an empty string.

    Yields:
        str: Progress updates as the audiobook generation progresses through loading text, generating audio,
             organizing by chapters, assembling chapters, and post-processing steps.
    """

    with open(
        f"{TEMP_DIR}/{book_title}/converted_book.txt", "r", encoding="utf-8"
    ) as f:
        text = f.read()
    lines = text.split("\n")
    # Filter out empty lines
    lines = [line.strip() for line in lines if line.strip()]

    # Set the voices to be used
    narrator_voice = ""  # voice to be used for narration
    dialogue_voice = ""  # voice to be used for dialogue

    if narrator_gender == "male":
        if MODEL == "kokoro":
            narrator_voice = "am_puck"
            dialogue_voice = "af_alloy"
        else:
            narrator_voice = "leo"
            dialogue_voice = "dan"
    else:
        if MODEL == "kokoro":
            narrator_voice = "af_heart"
            dialogue_voice = "af_sky"
        else:
            narrator_voice = "tara"
            dialogue_voice = "leah"

    # Setup directories
    temp_line_audio_dir = os.path.join(TEMP_DIR, book_title, "line_segments")

    empty_directory(os.path.join(temp_line_audio_dir, book_title))

    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(temp_line_audio_dir, exist_ok=True)

    # Batch processing parameters
    semaphore = asyncio.Semaphore(10)
    print(f"MAX_PARALLEL_REQUESTS_BATCH_SIZE: {MAX_PARALLEL_REQUESTS_BATCH_SIZE}")

    # Initial setup for chapters
    chapter_index = 1

    if MODEL == "orpheus":
        current_chapter_audio = "Introduction.m4a"
    else:
        current_chapter_audio = f"Introduction.{output_format}"
    chapter_files = []

    # First pass: Generate audio for each line independently
    total_size = len(lines)

    progress_counter = 0

    # For tracking progress with tqdm in an async context
    progress_bar = tqdm(total=total_size, unit="line", desc="Audio Generation Progress")

    # Maps chapters to their line indices
    chapter_line_map = {}

    async def process_single_line(line_index, line):
        async with semaphore:
            nonlocal progress_counter
            if not line:
                return None

            # Check if task has been cancelled before processing this line
            if task_id and is_task_cancelled(task_id):
                print(
                    f"[DEBUG] Task {task_id} cancelled before processing line {line_index}"
                )
                raise asyncio.CancelledError("Task was cancelled by user")

            annotated_parts = split_and_annotate_text(line)
            part_files = []  # Store temporary files for each part

            for i, part in enumerate(annotated_parts):
                text_to_speak = part["text"]
                # Check if task has been cancelled
                if task_id and is_task_cancelled(task_id):
                    print(f"[DEBUG] Task {task_id} cancelled during line processing")
                    raise asyncio.CancelledError("Task was cancelled by user")

                # Preprocess text for Orpheus to prevent repetition issues
                if MODEL == "orpheus":
                    text_to_speak = preprocess_text_for_orpheus(text_to_speak)

                voice_to_speak_in = (
                    narrator_voice if part["type"] == "narration" else dialogue_voice
                )

                # Create temporary file for this part
                part_file_path = os.path.join(
                    temp_line_audio_dir,
                    f"line_{line_index:06d}_part_{i}.{API_OUTPUT_FORMAT}",
                )

                try:
                    # Use retry mechanism for TTS generation
                    current_part_audio_buffer = await generate_tts_with_retry(
                        async_openai_client,
                        MODEL,
                        voice_to_speak_in,
                        text_to_speak,
                        API_OUTPUT_FORMAT,
                        speed=0.85,
                        max_retries=5,
                        task_id=task_id,
                    )

                    # Save this part to a temporary file
                    if len(current_part_audio_buffer) > 0:
                        with open(part_file_path, "wb") as part_file:
                            part_file.write(current_part_audio_buffer)
                        part_files.append(part_file_path)
                    else:
                        print(
                            f"WARNING: TTS for part type '{part['type']}', voice '{voice_to_speak_in}', text: '{text_to_speak[:50]}...' returned 0 bytes after retries."
                        )

                except asyncio.CancelledError:
                    # Re-raise cancellation errors
                    raise
                except Exception as e:
                    print(
                        f"CRITICAL ERROR: TTS failed after all retries for part type '{part['type']}', voice '{voice_to_speak_in}', text: '{text_to_speak[:50]}...': {e}"
                    )
                    # This is now a critical error since we've exhausted retries
                    # We should not continue as this will result in missing audio
                    raise e

            # Concatenate all parts using FFmpeg
            if part_files:
                final_line_path = os.path.join(
                    temp_line_audio_dir, f"line_{line_index:06d}.{API_OUTPUT_FORMAT}"
                )

                if len(part_files) == 1:
                    # Single part, just rename the file
                    os.rename(part_files[0], final_line_path)
                else:
                    # Multiple parts, concatenate with FFmpeg
                    parts_list_file = os.path.join(
                        temp_line_audio_dir, f"parts_list_{line_index:06d}.txt"
                    )

                    # Create file list for FFmpeg with absolute paths
                    with open(parts_list_file, "w", encoding="utf-8") as f:
                        for part_file in part_files:
                            abs_path = os.path.abspath(part_file)
                            f.write(f"file '{abs_path}'\n")

                    with open(parts_list_file, "r") as f:
                        print(f.read())

                    normalized_parts = []

                    for i, part_file in enumerate(part_files):
                        normalized_file = os.path.join(
                            temp_line_audio_dir, f"norm_{line_index:06d}_{i}.wav"
                        )

                        # Normalize to consistent format: 22050Hz, mono, 16-bit PCM WAV
                        normalize_cmd = [
                            "ffmpeg",
                            "-y",
                            "-i",
                            part_file,
                            "-ar",
                            "22050",
                            "-ac",
                            "1",
                            "-c:a",
                            "pcm_s16le",
                            normalized_file,
                        ]

                        try:
                            result = subprocess.run(
                                normalize_cmd,
                                check=True,
                                capture_output=True,
                                text=True,
                            )
                            normalized_parts.append(normalized_file)
                            print(
                                f"  Normalized part {i}: {os.path.getsize(normalized_file)} bytes"
                            )
                        except subprocess.CalledProcessError as e:
                            print(f"ERROR normalizing part {i}: {e}")
                            print(f"FFmpeg stderr: {e.stderr}")
                            continue

                    if not normalized_parts:
                        print(
                            "ERROR: No parts could be normalized, using first original part"
                        )
                        import shutil

                        shutil.copy2(part_files[0], final_line_path)
                    else:
                        # Step 2: Concatenate normalized parts using simple file list
                        concat_list_file = os.path.join(
                            temp_line_audio_dir, f"concat_{line_index:06d}.txt"
                        )

                        with open(concat_list_file, "w", encoding="utf-8") as f:
                            for norm_file in normalized_parts:
                                f.write(f"file '{os.path.abspath(norm_file)}'\n")

                        # Final concatenation to target format
                        if API_OUTPUT_FORMAT == "wav":
                            concat_cmd = [
                                "ffmpeg",
                                "-y",
                                "-f",
                                "concat",
                                "-safe",
                                "0",
                                "-i",
                                concat_list_file,
                                "-c",
                                "copy",
                                final_line_path,
                            ]
                        else:
                            concat_cmd = [
                                "ffmpeg",
                                "-y",
                                "-f",
                                "concat",
                                "-safe",
                                "0",
                                "-i",
                                concat_list_file,
                                "-c:a",
                                "aac",
                                "-b:a",
                                "128k",
                                final_line_path,
                            ]

                        try:
                            result = subprocess.run(
                                concat_cmd, check=True, capture_output=True, text=True
                            )
                            print(
                                f"✅ Successfully concatenated {len(normalized_parts)} parts"
                            )

                            # Clean up normalized files
                            for norm_file in normalized_parts:
                                os.remove(norm_file)
                            os.remove(concat_list_file)

                        except subprocess.CalledProcessError as e:
                            print(f"ERROR in final concatenation: {e}")
                            print(f"FFmpeg stderr: {e.stderr}")
                            # Fallback: use first normalized part
                            import shutil

                            shutil.copy2(normalized_parts[0], final_line_path)
                            print("Used first normalized part as fallback")

                    # Clean up original parts
                    for part_file in part_files:
                        os.remove(part_file)
                    if os.path.exists(parts_list_file):
                        os.remove(parts_list_file)

            else:
                print(f"WARNING: Line {line_index} resulted in no valid audio parts.")

            progress_bar.update(1)
            progress_counter += 1
            # update the task status
            update_task_status(
                task_id,
                "generating",
                f"Generating audiobook. Progress: {progress_counter}/{total_size}",
            )

            return {
                "index": line_index,
                "is_chapter_heading": check_if_chapter_heading(line),
                "line": line,
            }

    # Create tasks and store them with their index for result collection
    tasks = []
    task_to_index = {}
    for i, line in enumerate(lines):
        task = asyncio.create_task(process_single_line(i, line))
        tasks.append(task)
        task_to_index[task] = i

    # Initialize results_all list
    results_all = [None] * len(lines)

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
                shutil.rmtree(f"{TEMP_DIR}/{book_title}")
                os.rmdir(f"{TEMP_DIR}/{book_title}")
                break

    # Start the cancellation monitor
    monitor_task = asyncio.create_task(cancellation_monitor())

    # Process tasks with progress updates
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
            percent = (progress_counter / total_size) * 100

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

    progress_bar.close()

    results = [r for r in results_all if r is not None]

    yield "Completed generating audio for all lines"

    # Second pass: Organize by chapters
    chapter_organization_bar = tqdm(
        total=len(results), unit="result", desc="Organizing Chapters"
    )

    for result in sorted(results, key=lambda x: x["index"]):
        # Check if this is a chapter heading
        if result["is_chapter_heading"]:
            chapter_index += 1

            if MODEL == "orpheus":
                current_chapter_audio = f"{sanitize_filename(result['line'])}.m4a"
            else:
                current_chapter_audio = (
                    f"{sanitize_filename(result['line'])}.{output_format}"
                )

        if current_chapter_audio not in chapter_files:
            chapter_files.append(current_chapter_audio)
            chapter_line_map[current_chapter_audio] = []

        # Add this line index to the chapter
        chapter_line_map[current_chapter_audio].append(result["index"])
        chapter_organization_bar.update(1)

    chapter_organization_bar.close()
    yield "Organizing audio by chapters complete"

    concatenate_chapters(
        chapter_files, book_title, chapter_line_map, temp_line_audio_dir
    )

    # Optimized parallel post-processing
    yield "Starting parallel post-processing..."
    m4a_chapter_files = await parallel_post_processing(
        chapter_files, book_title, output_format
    )
    yield f"Completed parallel post-processing of {len(m4a_chapter_files)} chapters"

    # Clean up temp line audio files
    shutil.rmtree(temp_line_audio_dir)
    yield "Cleaned up temporary files"

    if generate_m4b_audiobook_file:
        # Merge all chapter files into a final m4b audiobook
        yield "Creating M4B audiobook file..."
        merge_chapters_to_m4b(book_path, m4a_chapter_files, book_title)
        # clean the temp directory
        shutil.rmtree(f"{TEMP_DIR}/{book_title}")
        yield "M4B audiobook created successfully"
    else:
        # Merge all chapter files into a standard M4A audiobook
        yield "Creating final audiobook..."
        merge_chapters_to_standard_audio_file(m4a_chapter_files, book_title)
        safe_book_title = sanitize_book_title_for_filename(book_title)
        convert_audio_file_formats(
            API_OUTPUT_FORMAT, output_format, "generated_audiobooks", safe_book_title
        )
        yield f"Audiobook in {output_format} format created successfully"


async def generate_audio_with_multiple_voices(
    output_format,
    narrator_gender,
    generate_m4b_audiobook_file=False,
    book_path="",
    book_title="audiobook",
    task_id=None,
):
    # Path to the JSONL file containing speaker-attributed lines
    """
    Generate an audiobook in the specified format using multiple voices for each line

    Uses the provided JSONL file to map speaker names to voices. The JSONL file should contain
    entries with the following format:
    {
        "line": <string>,
        "speaker": <string>
    }

    The function will generate audio for each line independently and then concatenate the audio
    files for each chapter in order. The final audiobook will be saved in the "generated_audiobooks"
    directory with the name "audiobook.<format>".

    :param output_format: The desired format of the final audiobook (e.g. "m4a", "mp3")
    :param narrator_gender: The gender of the narrator voice (e.g. "male", "female")
    :param generate_m4b_audiobook_file: Whether to generate an M4B audiobook file instead of a standard
    M4A file
    :param book_path: The path to the book file (required for generating an M4B audiobook file)
    """
    file_path = "speaker_attributed_book.jsonl"
    json_data_array = []
    # Open the JSONL file and read it line by line
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            # Parse each line as a JSON object
            json_object = json.loads(line.strip())
            # Append the parsed JSON object to the array
            json_data_array.append(json_object)

    yield "Loaded speaker-attributed lines from JSONL file"

    # Load mappings for character gender and voice selection
    character_gender_map = read_json("character_gender_map.json")
    voice_map = None

    if narrator_gender == "male":
        if MODEL == "kokoro":
            voice_map = read_json("static_files/kokoro_voice_map_male_narrator.json")
        else:
            voice_map = read_json("static_files/orpheus_voice_map_male_narrator.json")
    else:
        if MODEL == "kokoro":
            voice_map = read_json("static_files/kokoro_voice_map_female_narrator.json")
        else:
            voice_map = read_json("static_files/orpheus_voice_map_female_narrator.json")

    narrator_voice = find_voice_for_gender_score(
        "narrator", character_gender_map, voice_map
    )
    yield "Loaded voice mappings and selected narrator voice"

    # Setup directories
    temp_line_audio_dir = os.path.join(TEMP_DIR, book_title, "line_segments")

    empty_directory(temp_line_audio_dir)

    os.makedirs(TEMP_DIR, exist_ok=True)
    os.makedirs(temp_line_audio_dir, exist_ok=True)
    yield "Set up temporary directories for audio processing"

    # Batch processing parameters
    semaphore = asyncio.Semaphore(MAX_PARALLEL_REQUESTS_BATCH_SIZE)

    # Initial setup for chapters
    chapter_index = 1
    if MODEL == "orpheus":
        current_chapter_audio = "Introduction.m4a"
    else:
        current_chapter_audio = f"Introduction.{output_format}"
    chapter_files = []

    # First pass: Generate audio for each line independently
    # and track chapter organization
    chapter_line_map = {}  # Maps chapters to their line indices

    progress_counter = 0

    # For tracking progress with tqdm in an async context
    total_lines = len(json_data_array)
    progress_bar = tqdm(
        total=total_lines, unit="line", desc="Audio Generation Progress"
    )

    yield "Generating audio..."

    async def process_single_line(line_index, doc):
        async with semaphore:
            nonlocal progress_counter

            line = doc["line"].strip()
            if not line:
                progress_bar.update(1)  # Update the progress bar even for empty lines
                return None

            # Check if task has been cancelled before processing this line
            if task_id and is_task_cancelled(task_id):
                print(
                    f"[DEBUG] Multi-voice task {task_id} cancelled before processing line {line_index}"
                )
                raise asyncio.CancelledError("Task was cancelled by user")

            speaker = doc["speaker"]

            speaker_voice = find_voice_for_gender_score(
                speaker, character_gender_map, voice_map
            )

            annotated_parts = split_and_annotate_text(line)
            part_files = []  # Store temporary files for each part

            for i, part in enumerate(annotated_parts):
                text_to_speak = part["text"]

                # Check if task has been cancelled during part processing
                if task_id and is_task_cancelled(task_id):
                    print(
                        f"[DEBUG] Multi-voice task {task_id} cancelled during part processing"
                    )
                    raise asyncio.CancelledError("Task was cancelled by user")

                # Preprocess text for Orpheus to prevent repetition issues
                if MODEL == "orpheus":
                    text_to_speak = preprocess_text_for_orpheus(text_to_speak)

                voice_to_speak_in = (
                    narrator_voice if part["type"] == "narration" else speaker_voice
                )

                # Create temporary file for this part
                part_file_path = os.path.join(
                    temp_line_audio_dir,
                    f"line_{line_index:06d}_part_{i}.{API_OUTPUT_FORMAT}",
                )

                try:
                    # Use retry mechanism for TTS generation (multi-voice)
                    current_part_audio_buffer = await generate_tts_with_retry(
                        async_openai_client,
                        MODEL,
                        voice_to_speak_in,
                        text_to_speak,
                        API_OUTPUT_FORMAT,
                        speed=0.85,
                        max_retries=5,
                        task_id=task_id,
                    )

                    # Save this part to a temporary file
                    if len(current_part_audio_buffer) > 0:
                        with open(part_file_path, "wb") as part_file:
                            part_file.write(current_part_audio_buffer)
                        part_files.append(part_file_path)
                    else:
                        print(
                            f"WARNING (multi-voice): TTS for part type '{part['type']}', voice '{voice_to_speak_in}', text: '{text_to_speak[:50]}...' returned 0 bytes after retries."
                        )

                except asyncio.CancelledError:
                    # Re-raise cancellation errors
                    raise
                except Exception as e:
                    print(
                        f"CRITICAL ERROR (multi-voice): TTS failed after all retries for part type '{part['type']}', voice '{voice_to_speak_in}', text: '{text_to_speak[:50]}...': {e}"
                    )
                    # This is now a critical error since we've exhausted retries
                    # We should not continue as this will result in missing audio
                    raise e

            # Concatenate all parts using FFmpeg
            if part_files:
                final_line_path = os.path.join(
                    temp_line_audio_dir, f"line_{line_index:06d}.{API_OUTPUT_FORMAT}"
                )

                if len(part_files) == 1:
                    # Single part, just rename the file
                    os.rename(part_files[0], final_line_path)
                else:
                    # Multiple parts, concatenate with FFmpeg
                    parts_list_file = os.path.join(
                        temp_line_audio_dir, f"parts_list_{line_index:06d}.txt"
                    )

                    # Create file list for FFmpeg with absolute paths
                    with open(parts_list_file, "w", encoding="utf-8") as f:
                        for part_file in part_files:
                            abs_path = os.path.abspath(part_file)
                            f.write(f"file '{abs_path}'\n")

                    # Step 1: Normalize all parts to ensure compatibility (multi-voice)
                    print(
                        f"Multi-voice: Normalizing {len(part_files)} parts for concatenation..."
                    )
                    normalized_parts = []

                    for i, part_file in enumerate(part_files):
                        normalized_file = os.path.join(
                            temp_line_audio_dir, f"mv_norm_{line_index:06d}_{i}.wav"
                        )

                        # Normalize to consistent format: 22050Hz, mono, 16-bit PCM WAV
                        normalize_cmd = [
                            "ffmpeg",
                            "-y",
                            "-i",
                            part_file,
                            "-ar",
                            "22050",
                            "-ac",
                            "1",
                            "-c:a",
                            "pcm_s16le",
                            normalized_file,
                        ]

                        try:
                            result = subprocess.run(
                                normalize_cmd,
                                check=True,
                                capture_output=True,
                                text=True,
                            )
                            normalized_parts.append(normalized_file)
                            print(
                                f"  Multi-voice normalized part {i}: {os.path.getsize(normalized_file)} bytes"
                            )
                        except subprocess.CalledProcessError as e:
                            print(f"ERROR (multi-voice) normalizing part {i}: {e}")
                            print(f"FFmpeg stderr: {e.stderr}")
                            continue

                    if not normalized_parts:
                        print(
                            "ERROR (multi-voice): No parts could be normalized, using first original part"
                        )
                        import shutil

                        shutil.copy2(part_files[0], final_line_path)
                    else:
                        # Step 2: Concatenate normalized parts using simple file list
                        concat_list_file = os.path.join(
                            temp_line_audio_dir, f"mv_concat_{line_index:06d}.txt"
                        )

                        with open(concat_list_file, "w", encoding="utf-8") as f:
                            for norm_file in normalized_parts:
                                f.write(f"file '{os.path.abspath(norm_file)}'\n")

                        # Final concatenation to target format
                        if API_OUTPUT_FORMAT == "wav":
                            concat_cmd = [
                                "ffmpeg",
                                "-y",
                                "-f",
                                "concat",
                                "-safe",
                                "0",
                                "-i",
                                concat_list_file,
                                "-c",
                                "copy",
                                final_line_path,
                            ]
                        else:
                            concat_cmd = [
                                "ffmpeg",
                                "-y",
                                "-f",
                                "concat",
                                "-safe",
                                "0",
                                "-i",
                                concat_list_file,
                                "-c:a",
                                "aac",
                                "-b:a",
                                "128k",
                                final_line_path,
                            ]

                        try:
                            result = subprocess.run(
                                concat_cmd, check=True, capture_output=True, text=True
                            )
                            print(
                                f"✅ Multi-voice: Successfully concatenated {len(normalized_parts)} parts"
                            )

                            # Clean up normalized files
                            for norm_file in normalized_parts:
                                os.remove(norm_file)
                            os.remove(concat_list_file)

                        except subprocess.CalledProcessError as e:
                            print(f"ERROR (multi-voice) in final concatenation: {e}")
                            print(f"FFmpeg stderr: {e.stderr}")
                            # Fallback: use first normalized part
                            import shutil

                            shutil.copy2(normalized_parts[0], final_line_path)
                            print("Multi-voice: Used first normalized part as fallback")

                    # Clean up original parts
                    for part_file in part_files:
                        os.remove(part_file)
                    if os.path.exists(parts_list_file):
                        os.remove(parts_list_file)
            else:
                print(
                    f"WARNING (multi-voice): Line {line_index} resulted in no valid audio parts."
                )

            progress_bar.update(1)
            progress_counter += 1

            return {
                "index": line_index,
                "is_chapter_heading": check_if_chapter_heading(line),
                "line": line,
            }

    # Create tasks and store them with their index for result collection
    tasks = []
    task_to_index = {}
    for i, doc in enumerate(json_data_array):
        task = asyncio.create_task(process_single_line(i, doc))
        tasks.append(task)
        task_to_index[task] = i

    # Initialize results_all list
    results_all = [None] * len(json_data_array)

    # Create a cancellation monitor task for multi-voice
    async def cancellation_monitor_mv():
        while tasks:
            await asyncio.sleep(0.5)  # Check every 500ms
            if task_id and is_task_cancelled(task_id):
                print(
                    f"[DEBUG] Multi-voice cancellation monitor detected task {task_id} is cancelled"
                )
                # Cancel all remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                        print(
                            f"[DEBUG] Multi-voice cancellation monitor cancelled a task"
                        )
                # clean up the temp directory
                shutil.rmtree(f"{TEMP_DIR}/{book_title}")
                os.rmdir(f"{TEMP_DIR}/{book_title}")
                break

    # Start the cancellation monitor
    monitor_task_mv = asyncio.create_task(cancellation_monitor_mv())

    # Process tasks with progress updates
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
                    f"[DEBUG] Multi-voice task {task_id} was cancelled, cancelling all pending tasks"
                )
                # Cancel all pending tasks immediately
                for task in tasks:
                    if not task.done():
                        task.cancel()
                        print(f"[DEBUG] Cancelled pending multi-voice task")
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
    if not monitor_task_mv.done():
        monitor_task_mv.cancel()

    progress_bar.close()

    # Filter out empty lines (same as in your original code)
    results = [r for r in results_all if r is not None]
    # update the task status
    update_task_status(
        task_id,
        "generating",
        f"Generating audiobook. Progress: {percent:.1f}%",
    )
    yield "Completed generating audio for all lines"

    # Second pass: Organize by chapters
    chapter_organization_bar = tqdm(
        total=len(results), unit="result", desc="Organizing Chapters"
    )
    # update the task status
    update_task_status(
        task_id,
        "generating",
        f"Generating audiobook. Progress: {percent:.1f}%",
    )
    yield "Organizing lines into chapters"

    for result in sorted(results, key=lambda x: x["index"]):
        # Check if this is a chapter heading
        if result["is_chapter_heading"]:
            chapter_index += 1

            if MODEL == "orpheus":
                current_chapter_audio = f"{sanitize_filename(result['line'])}.m4a"
            else:
                current_chapter_audio = (
                    f"{sanitize_filename(result['line'])}.{output_format}"
                )

        if current_chapter_audio not in chapter_files:
            chapter_files.append(current_chapter_audio)
            chapter_line_map[current_chapter_audio] = []

        # Add this line index to the chapter
        # The 'index' from 'result' is crucial for concatenate_chapters to find the correct audio files
        chapter_line_map[current_chapter_audio].append(result["index"])
        chapter_organization_bar.update(1)

    chapter_organization_bar.close()
    # update the task status
    update_task_status(
        task_id,
        "generating",
        f"Generating audiobook. Progress: {percent:.1f}%",
    )
    yield f"Organized {len(results)} lines into {len(chapter_files)} chapters"

    concatenate_chapters(
        chapter_files, book_title, chapter_line_map, temp_line_audio_dir
    )

    # Optimized parallel post-processing
    yield "Starting parallel post-processing..."
    # update the task status
    update_task_status(
        task_id,
        "generating",
        f"Generating audiobook. Progress: {percent:.1f}%",
    )
    m4a_chapter_files = await parallel_post_processing(
        chapter_files, book_title, output_format
    )
    yield f"Completed parallel post-processing of {len(m4a_chapter_files)} chapters"

    # Clean up temp line audio files
    yield "Cleaning up temporary files"
    # update the task status
    update_task_status(
        task_id,
        "generating",
        f"Generating audiobook. Progress: {percent:.1f}%",
    )
    shutil.rmtree(temp_line_audio_dir)
    yield "Temporary files cleanup complete"

    if generate_m4b_audiobook_file:
        # Merge all chapter files into a final m4b audiobook
        yield "Creating M4B audiobook file..."
        merge_chapters_to_m4b(book_path, m4a_chapter_files, book_title)
        # clean the temp directory
        shutil.rmtree(f"{TEMP_DIR}/{book_title}")
        yield "M4B audiobook created successfully"
    else:
        # Merge all chapter files into a standard M4A audiobook
        yield "Creating final audiobook..."
        merge_chapters_to_standard_audio_file(m4a_chapter_files, book_title)
        safe_book_title = sanitize_book_title_for_filename(book_title)
        convert_audio_file_formats(
            API_OUTPUT_FORMAT, output_format, "generated_audiobooks", safe_book_title
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
    # Select narrator voice string based on narrator_gender and MODEL
    if narrator_gender == "male":
        if MODEL == "kokoro":
            narrator_voice = "am_puck"
        else:
            narrator_voice = "leo"
    else:
        if MODEL == "kokoro":
            narrator_voice = "af_heart"
        else:
            narrator_voice = "tara"

    is_tts_api_up, message = await check_tts_api(
        async_openai_client, MODEL, narrator_voice
    )

    if not is_tts_api_up:
        raise Exception(message)

    generate_m4b_audiobook_file = False
    # Determine the actual format to use for intermediate files
    actual_output_format = output_format

    if output_format == "M4B (Chapters & Cover)":
        generate_m4b_audiobook_file = True
        actual_output_format = "m4a"  # Use m4a for intermediate files when creating M4B

    if voice_option == "Single Voice":
        yield "\n🎧 Generating audiobook with a **single voice**..."
        await asyncio.sleep(1)
        async for line in generate_audio_with_single_voice(
            actual_output_format.lower(),
            narrator_gender,
            generate_m4b_audiobook_file,
            book_path,
            book_title,
            task_id,
        ):
            yield line
    elif voice_option == "Multi-Voice":
        yield "\n🎭 Generating audiobook with **multiple voices**..."
        await asyncio.sleep(1)
        async for line in generate_audio_with_multiple_voices(
            actual_output_format.lower(),
            narrator_gender,
            generate_m4b_audiobook_file,
            book_path,
            book_title,
            task_id,
        ):
            yield line

    yield f"\n🎧 Audiobook is generated ! You can now download it in the Download section below. Click on the blue download link next to the file name."


async def main(book_title="audiobook"):
    os.makedirs(f"{TEMP_DIR}/{book_title}/generated_audiobooks", exist_ok=True)

    # Default values
    book_path = "./sample_book_and_audio/The Adventure of the Lost Treasure - Prakhar Sharma.epub"
    generate_m4b_audiobook_file = False
    output_format = "aac"

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

    if audiobook_type_option == "1":
        is_calibre_installed = check_if_calibre_is_installed()

        if not is_calibre_installed:
            print(
                "⚠️ Calibre is not installed. Please install it first and make sure **calibre** and **ebook-meta** commands are available in your PATH."
            )
            return

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
        else:
            # Ask user for book file path if not provided
            input_path = input(
                "\n📖 Enter the **path to the book file**, needed for metadata and cover extraction. (Press Enter to use default): "
            ).strip()
            if input_path:
                book_path = input_path
            print(f"📂 Using book file: **{book_path}**")

        print("✅ Book path set. Proceeding...\n")

        generate_m4b_audiobook_file = True
    else:
        # Prompt user for audio format selection
        print("\n🎙️ **Audiobook Output Format Selection**")
        output_format = input(
            "🔹 Choose between ['aac', 'm4a', 'mp3', 'wav', 'opus', 'flac', 'pcm']. "
        ).strip()

        if output_format not in ["aac", "m4a", "mp3", "wav", "opus", "flac", "pcm"]:
            print("\n⚠️ Invalid output format! Please choose from the give options")
            return

    # Prompt user for narrator's gender selection
    print("\n🎙️ **Audiobook Narrator Voice Selection**")
    narrator_gender = input(
        "🔹 Enter **male** if you want the book to be read in a male voice or **female** if you want the book to be read in a female voice: "
    ).strip()

    if narrator_gender not in ["male", "female"]:
        print("\n⚠️ Invalid narrator gender! Please choose from the give options")
        return

    start_time = time.time()

    if voice_option == "1":
        print("\n🎧 Generating audiobook with a **single voice**...")
        async for line in generate_audio_with_single_voice(
            output_format, narrator_gender, generate_m4b_audiobook_file, book_path
        ):
            print(line)
    elif voice_option == "2":
        print("\n🎭 Generating audiobook with **multiple voices**...")
        async for line in generate_audio_with_multiple_voices(
            output_format, narrator_gender, generate_m4b_audiobook_file, book_path
        ):
            print(line)
    else:
        print("\n⚠️ Invalid option! Please restart and enter either **1** or **2**.")
        return

    print(
        f"\n🎧 Audiobook is generated ! The audiobook is saved as **audiobook.{"m4b" if generate_m4b_audiobook_file else output_format}** in the **generated_audiobooks** directory in the current folder."
    )

    end_time = time.time()

    execution_time = end_time - start_time
    print(
        f"\n⏱️ **Execution Time:** {execution_time:.6f} seconds\n✅ Audiobook generation complete!"
    )


if __name__ == "__main__":
    asyncio.run(main())
