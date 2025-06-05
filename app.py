"""
Audiobook Creator
Copyright (C) 2025 Prakhar Sharma

Modified by:
- Chris Scott
- https://github.com/chris-scott
- added support for orpheus tts api
- Updated the UI to be more user friendly

Original code by:
- Prakhar Sharma
- https://github.com/prakharsharma


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

import gradio as gr
import os
import traceback
import shutil
import asyncio

from datetime import datetime
from fastapi import FastAPI
from book_to_txt import process_book_and_extract_text, save_book
from identify_characters_and_output_book_to_jsonl import (
    process_book_and_identify_characters,
)
from generate_audiobook import process_audiobook_generation, generate_audiobook_background
from utils.task_utils import (
    get_active_tasks,
    get_past_generated_files,
    load_tasks,
    save_tasks,
    update_task_status,
    cancel_task,
    register_running_task,
    unregister_running_task,
    clear_temp_files,
)
from utils.config_manager import config_manager
from config.constants import reload_constants

css = """
.step-heading {font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem}
"""

app = FastAPI()


def delete_audiobook_file(file_path):
    """Delete an audiobook file and return status message"""
    if not file_path:
        return gr.Warning("No file selected for deletion.")

    try:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            file_size = round(os.path.getsize(file_path) / (1024 * 1024), 2)
            os.remove(file_path)
            return gr.Info(
                f"Successfully deleted '{filename}' ({file_size} MB).", duration=5
            )
        else:
            return gr.Warning("File not found. It may have already been deleted.")
    except PermissionError:
        return gr.Warning("Permission denied. Cannot delete the file.")
    except Exception as e:
        return gr.Warning(f"Error deleting file: {str(e)}")


def get_selected_file_info(file_path):
    """Get information about the selected file for confirmation"""
    if not file_path:
        return "No file selected."

    try:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            file_size = round(os.path.getsize(file_path) / (1024 * 1024), 2)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime(
                "%Y-%m-%d %H:%M"
            )
            return f"**Selected for deletion:**\n\n📁 **{filename}**\n📊 Size: {file_size} MB\n📅 Modified: {mod_time}\n\n⚠️ **This action cannot be undone!**"
        else:
            return "File not found."
    except Exception as e:
        return f"Error reading file info: {str(e)}"


def delete_audiobook_file(file_path):
    """Delete an audiobook file and return status message"""
    if not file_path:
        return gr.Warning("No file selected for deletion.")

    try:
        if os.path.exists(file_path):
            filename = os.path.basename(file_path)
            file_size = round(os.path.getsize(file_path) / (1024 * 1024), 2)
            os.remove(file_path)
            return gr.Info(
                f"Successfully deleted '{filename}' ({file_size} MB).", duration=5
            )
        else:
            return gr.Warning("File not found. It may have already been deleted.")
    except PermissionError:
        return gr.Warning("Permission denied. Cannot delete the file.")
    except Exception as e:
        return gr.Warning(f"Error deleting file: {str(e)}")


def delete_all_audiobooks():
    """Delete all audiobook files and return status message"""
    try:
        audiobooks_dir = "generated_audiobooks"
        if not os.path.exists(audiobooks_dir):
            return gr.Warning("No audiobooks directory found.")
        
        # Get all files (excluding hidden files)
        files = [f for f in os.listdir(audiobooks_dir) 
                if os.path.isfile(os.path.join(audiobooks_dir, f)) and not f.startswith('.')]
        
        if not files:
            return gr.Warning("No audiobooks found to delete.")
        
        # Calculate total size before deletion
        total_size = 0
        for file in files:
            try:
                total_size += os.path.getsize(os.path.join(audiobooks_dir, file))
            except:
                pass
        
        total_size_mb = round(total_size / (1024 * 1024), 2)
        
        # Delete all files
        deleted_count = 0
        failed_files = []
        
        for file in files:
            file_path = os.path.join(audiobooks_dir, file)
            try:
                os.remove(file_path)
                deleted_count += 1
            except Exception as e:
                failed_files.append(f"{file}: {str(e)}")
        
        if failed_files:
            failed_msg = "\n".join(failed_files[:3])  # Show only first 3 failures
            if len(failed_files) > 3:
                failed_msg += f"\n... and {len(failed_files) - 3} more"
            return gr.Warning(f"Deleted {deleted_count}/{len(files)} files ({total_size_mb} MB). Failures:\n{failed_msg}")
        else:
            return gr.Info(f"Successfully deleted all {deleted_count} audiobooks ({total_size_mb} MB).", duration=5)
            
    except Exception as e:
        return gr.Warning(f"Error deleting audiobooks: {str(e)}")


def get_all_audiobooks_info():
    """Get information about all audiobooks for confirmation"""
    try:
        audiobooks_dir = "generated_audiobooks"
        if not os.path.exists(audiobooks_dir):
            return "No audiobooks directory found."
        
        files = [f for f in os.listdir(audiobooks_dir) 
                if os.path.isfile(os.path.join(audiobooks_dir, f)) and not f.startswith('.')]
        
        if not files:
            return "No audiobooks found to delete."
        
        total_size = 0
        for file in files:
            try:
                total_size += os.path.getsize(os.path.join(audiobooks_dir, file))
            except:
                pass
        
        total_size_mb = round(total_size / (1024 * 1024), 2)
        file_list = "\n".join([f"• {file}" for file in files[:10]])  # Show first 10 files
        if len(files) > 10:
            file_list += f"\n... and {len(files) - 10} more files"
        
        return f"**⚠️ Delete ALL audiobooks?**\n\n**Total: {len(files)} files ({total_size_mb} MB)**\n\nFiles to be deleted:\n{file_list}\n\n**This action cannot be undone!**"
    except Exception as e:
        return f"Error reading audiobooks: {str(e)}"


def refresh_past_files_with_continue():
    files = get_past_generated_files()
    active_tasks = get_active_tasks()
    tasks = load_tasks()

    # Build display text
    display_parts = []
    task_choices = []
    cancel_group_visible = False

    # Show active tasks first
    if active_tasks:
        display_parts.append("### 🔄 Currently Running:")
        for task in active_tasks:
            task_display = (
                f"⏳ **{task['id']}** - {task['progress']} ({task['timestamp'][:16]})"
            )
            display_parts.append(task_display)
            # Add task to choices for cancellation
            task_choices.append((f"{task['id']} - {task['progress']}", task['id']))
        display_parts.append("")  # Empty line
        
        # Show cancel group if there are active tasks
        if task_choices:
            cancel_group_visible = True

    # Show past files
    file_choices = []
    if not files:
        if not active_tasks:
            display_parts.append("### 📂 Generated Audiobooks")
            display_parts.append("✨ No audiobooks found. Generate your first audiobook above!")
        # If there are active tasks but no completed files, don't show the empty message
    else:
        if active_tasks:
            display_parts.append("### 📁 Completed Audiobooks:")
        else:
            display_parts.append("### 📂 Generated Audiobooks")

        for file_info in files:
            file_display = f"📁 **{file_info['filename']}** ({file_info['size_mb']} MB) - {file_info['modified']}"
            display_parts.append(file_display)
            file_choices.append((file_info["filename"], file_info["path"]))

    file_text = "\n\n".join(display_parts)
    
    return (
        gr.update(value=file_text, visible=True),  # past_files_display
        gr.update(
            choices=file_choices,
            value=file_choices[0][1] if file_choices else None,
            visible=bool(file_choices),
        ),  # past_files_dropdown
        gr.update(visible=bool(file_choices)),  # delete_btn
        gr.update(
            choices=task_choices,
            value=task_choices[0][1] if task_choices else None,
            visible=bool(task_choices),
        ),  # active_tasks_dropdown
        gr.update(visible=cancel_group_visible),  # cancel_task_group
    )


def validate_book_upload(book_file, book_title):
    """Validate book upload and return a notification"""
    if book_file is None:
        return gr.Warning("Please upload a book file first.")

    if not book_title:
        return gr.Warning("Please enter a book title.")

    return gr.Info(f"Book '{book_title}' ready for processing.", duration=5)


def text_extraction_wrapper(book_file, text_decoding_option, book_title):
    """Wrapper for text extraction with validation and progress updates"""
    if book_file is None or not book_title:
        yield None
        return gr.Warning("Please upload a book file and enter a title first.")

    try:
        # Copy uploaded file to temp dir and use that path
        safe_book_file = save_uploaded_file_to_temp(book_file, book_title)
        last_output = None
        # Pass through all yield values from the original function
        for output in process_book_and_extract_text(
            safe_book_file, text_decoding_option, book_title
        ):
            last_output = output
            yield output  # Yield each progress update

        # Final yield with success notification
        yield last_output
        return gr.Info(
            "Text extracted successfully! You can now edit the content.", duration=5
        )
    except Exception as e:
        print(e)
        traceback.print_exc()
        yield None
        return gr.Warning(f"Error extracting text: {str(e)}")


def save_book_wrapper(text_content, book_title):
    """Wrapper for saving book with validation"""
    if not text_content:
        return gr.Warning("No text content to save.")

    if not book_title:
        return gr.Warning("Please enter a book title before saving.")

    try:
        save_book(text_content, book_title)
        return gr.Info(
            "📖 Book saved successfully as 'converted_book.txt'!", duration=10
        )
    except Exception as e:
        print(e)
        traceback.print_exc()
        return gr.Warning(f"Error saving book: {str(e)}")


async def identify_characters_wrapper(book_title):
    """Wrapper for character identification with validation and progress updates"""
    if not book_title:
        yield gr.Warning("Please enter a book title first.")
        yield None
        return

    try:
        last_output = None
        # Pass through all yield values from the original function
        async for output in process_book_and_identify_characters(book_title):
            last_output = output
            yield output  # Yield each progress update

        # Final yield with success notification
        yield gr.Info(
            "Character identification complete! Proceed to audiobook generation.",
            duration=5,
        )
        yield last_output
        return
    except Exception as e:
        print(e)
        traceback.print_exc()
        yield gr.Warning(f"Error identifying characters: {str(e)}")
        yield None
        return


def save_uploaded_file_to_temp(book_file, book_title):
    """Copy uploaded file to a safe temp directory and return the new absolute path."""
    temp_dir = os.path.abspath(os.path.join("temp", book_title))
    os.makedirs(temp_dir, exist_ok=True)
    filename = os.path.basename(
        book_file.name if hasattr(book_file, "name") else book_file
    )
    dest_path = os.path.abspath(os.path.join(temp_dir, filename))
    # Get absolute path for source file
    src_path = os.path.abspath(
        book_file.name if hasattr(book_file, "name") else book_file
    )
    shutil.copy(src_path, dest_path)
    # Set permissions: readable and writable by user, group, and others
    import os as _os

    _os.chmod(dest_path, 0o666)
    return dest_path


async def generate_audiobook_wrapper(
    voice_type, narrator_gender, output_format, book_file, book_title
):
    """Wrapper for audiobook generation with validation and progress updates"""
    if book_file is None:
        yield gr.Warning("Please upload a book file first."), None
        yield None, None
        return
    if not book_title:
        yield gr.Warning("Please enter a book title first."), None
        yield None, None
        return
    if not voice_type or not output_format:
        yield gr.Warning("Please select voice type and output format."), None
        yield None, None
        return

    # Check if character identification is required and completed
    if voice_type == "Multi-Voice":
        # Construct the expected path for the character identification output file
        char_ident_file_path = os.path.join(
            "temp", book_title, "speaker_attributed_book.jsonl"
        )
        if not os.path.exists(char_ident_file_path):
            yield gr.Warning(
                "Multi-Voice narration requires character identification. "
                "Please complete Step 3: Character Identification first."
            ), None
            yield None, None
            return

    # Copy uploaded file to temp dir and use that path
    safe_book_file = save_uploaded_file_to_temp(book_file, book_title)

    # Create a unique task ID for tracking
    task_id = f"{book_title}_{voice_type}_{narrator_gender}_{output_format}"

    try:
        # Mark task as starting
        update_task_status(
            task_id,
            "starting",
            f"Initializing {voice_type} audiobook generation for '{book_title}'",
            params={
                "voice_type": voice_type,
                "narrator_gender": narrator_gender,
                "output_format": output_format,
                "book_file": safe_book_file,
                "book_title": book_title,
            },
        )

        # Create a background task that will continue even if UI disconnects
        async def background_audiobook_generation():
            """Background task that continues audiobook generation even if UI disconnects"""
            try:
                # Determine voice type for the background function
                voice_mode = "multi_voice" if voice_type == "Multi-Voice" else "single_voice"
                
                # Run the background generation without any yield statements
                await generate_audiobook_background(
                    output_format.lower(),
                    narrator_gender,
                    safe_book_file,
                    book_title,
                    voice_mode,
                    task_id,
                )
                
            except asyncio.CancelledError:
                update_task_status(task_id, "cancelled", "Task was cancelled by user")
                raise
            except Exception as e:
                # Don't overwrite progress when marking as failed - let update_task_status preserve it
                update_task_status(task_id, "failed", "", str(e))
                print(f"Background generation error: {e}")
                traceback.print_exc()
                raise
            finally:
                unregister_running_task(task_id)

        # Start the background task
        background_task = asyncio.create_task(background_audiobook_generation())
        
        # Register the background task for cancellation instead of current task
        register_running_task(task_id, background_task)
        
        last_output = None
        audiobook_path = None
        ui_disconnected = False
        
        # Try to provide UI updates, but continue background task if UI disconnects
        try:
            # Provide initial progress updates while UI is connected
            while not background_task.done():
                try:
                    # Check task status and yield progress
                    tasks = load_tasks()
                    if task_id in tasks:
                        current_progress = tasks[task_id].get("progress", "Processing...")
                        last_output = current_progress
                        yield current_progress, None
                    
                    # Wait a bit before next update
                    await asyncio.sleep(2)
                    
                except (GeneratorExit, StopAsyncIteration):
                    # UI has disconnected, but let background task continue
                    print(f"[DEBUG] UI disconnected for task {task_id}, but background generation continues...")
                    ui_disconnected = True
                    break
                except Exception as e:
                    print(f"[DEBUG] UI update error for task {task_id}: {e}, continuing background generation...")
                    ui_disconnected = True
                    break
            
            # Wait for background task to complete
            if not ui_disconnected:
                await background_task
                
        except (GeneratorExit, StopAsyncIteration):
            # UI disconnected, but background task continues
            print(f"[DEBUG] UI generator closed for task {task_id}, background generation continues...")
            # Don't await the background task here, let it continue independently
            
        except Exception as e:
            print(f"[DEBUG] UI error for task {task_id}: {e}, background generation continues...")
            # Don't await the background task here, let it continue independently

        # If UI is still connected, provide final status
        if not ui_disconnected and background_task.done():
            # Construct the expected audiobook file path
            audiobook_filename = f"{book_title}.{output_format.lower()}"
            audiobook_path = os.path.join("generated_audiobooks", audiobook_filename)
            
            # Verify the file exists before returning it
            if not os.path.exists(audiobook_path):
                audiobook_path = None

            # Final yield with success notification and file path
            try:
                yield gr.Info(
                    f"Audiobook generated successfully in {output_format} format! You can now download it below.",
                    duration=15,
                ), audiobook_path
                yield last_output, audiobook_path
            except (GeneratorExit, StopAsyncIteration):
                print(f"[DEBUG] UI disconnected during final update for task {task_id}")
                
        return
        
    except asyncio.CancelledError:
        # Handle task cancellation
        update_task_status(task_id, "cancelled", "Task was cancelled by user")
        unregister_running_task(task_id)
        try:
            yield gr.Warning("Audiobook generation was cancelled."), None
            yield None, None
        except (GeneratorExit, StopAsyncIteration):
            pass
        return
    except Exception as e:
        # Mark task as failed
        update_task_status(task_id, "failed", "Generation failed", str(e))
        unregister_running_task(task_id)
        print(e)
        traceback.print_exc()
        try:
            yield gr.Warning(f"Error generating audiobook: {str(e)}"), None
            yield None, None
        except (GeneratorExit, StopAsyncIteration):
            pass
        return





def get_temp_directory_info():
    """Get information about temp directories and files that would be cleared"""
    info_parts = []
    total_size_mb = 0
    total_files = 0
    
    # Check temp directory
    temp_dir = "temp"
    if os.path.exists(temp_dir):
        try:
            for item in os.listdir(temp_dir):
                # Skip hidden files and directories (starting with a period)
                if item.startswith('.'):
                    continue
                    
                item_path = os.path.join(temp_dir, item)
                if os.path.isdir(item_path):
                    # Count files and calculate size for this subdirectory
                    dir_files = 0
                    dir_size = 0
                    for root, dirs, files in os.walk(item_path):
                        # Filter out hidden files from the count
                        visible_files = [f for f in files if not f.startswith('.')]
                        dir_files += len(visible_files)
                        for file in visible_files:
                            file_path = os.path.join(root, file)
                            try:
                                dir_size += os.path.getsize(file_path)
                            except:
                                pass
                    
                    if dir_files > 0:
                        dir_size_mb = dir_size / (1024 * 1024)
                        info_parts.append(f"📁 **{item}** - {dir_files} files ({dir_size_mb:.1f} MB)")
                        total_size_mb += dir_size_mb
                        total_files += dir_files
                elif os.path.isfile(item_path):
                    # Single file in temp directory
                    try:
                        file_size = os.path.getsize(item_path) / (1024 * 1024)
                        info_parts.append(f"📄 **{item}** ({file_size:.1f} MB)")
                        total_size_mb += file_size
                        total_files += 1
                    except:
                        pass
        except Exception as e:
            info_parts.append(f"❌ Error reading temp directory: {str(e)}")
    
    # Check generated_audiobooks directory for old files
    generated_dir = "generated_audiobooks"
    if os.path.exists(generated_dir):
        try:
            audiobook_files = [f for f in os.listdir(generated_dir) 
                             if os.path.isfile(os.path.join(generated_dir, f)) and not f.startswith('.')]
            if audiobook_files:
                generated_size = 0
                for file in audiobook_files:
                    try:
                        generated_size += os.path.getsize(os.path.join(generated_dir, file))
                    except:
                        pass
                generated_size_mb = generated_size / (1024 * 1024)
                info_parts.append(f"🎵 **generated_audiobooks** - {len(audiobook_files)} files ({generated_size_mb:.1f} MB)")
        except:
            pass
    
    if not info_parts:
        return "✨ No temp files found - nothing to clear!"
    
    header = f"### 🧹 Files that will be cleared:\n**Total: {total_files} files ({total_size_mb:.1f} MB)**\n\n"
    return header + "\n".join(info_parts)





def cancel_task_wrapper(task_id):
    """Wrapper for cancelling a task with user feedback"""
    if not task_id:
        return gr.Warning("No task selected for cancellation.")

    try:
        success, message = cancel_task(task_id)
        if success:
            return gr.Info(f"Task cancelled: {message}", duration=5)
        else:
            return gr.Warning(f"Failed to cancel task: {message}")
    except Exception as e:
        print(f"Error cancelling task: {e}")
        return gr.Warning(f"Error cancelling task: {str(e)}")


async def continue_task_wrapper(task_id):
    """Wrapper for continuing/resuming a stuck task"""
    if not task_id:
        return gr.Warning("No task selected for continuation.")

    try:
        tasks = load_tasks()
        if task_id not in tasks:
            return gr.Warning(f"Task {task_id} not found.")
        
        task_info = tasks[task_id]
        params = task_info.get("params", {})
        
        # Extract parameters from the original task
        voice_type = params.get("voice_type", "Single Voice")
        narrator_gender = params.get("narrator_gender", "female")
        output_format = params.get("output_format", "m4b")
        book_file = params.get("book_file", "")
        book_title = params.get("book_title", "audiobook")
        
        if not all([voice_type, narrator_gender, output_format, book_file, book_title]):
            return gr.Warning("Incomplete task parameters. Cannot resume task.")
        
        # Debug: Check current progress before resuming
        from utils.task_utils import get_task_progress_index
        current_progress, total_lines = get_task_progress_index(task_id)
        print(f"[DEBUG] Resuming task {task_id}: current progress {current_progress}/{total_lines}")
        print(f"[DEBUG] Task info: {task_info}")
        
        # Check if we're very close to completion and should skip to assembly
        if total_lines > 0 and current_progress >= total_lines - 5:  # Within 5 lines of completion
            print(f"[DEBUG] Task {task_id} is near completion ({current_progress}/{total_lines}), proceeding to assembly phase")
            update_task_status(
                task_id,
                "running",
                f"Near completion, proceeding to audio assembly for '{book_title}'"
            )
        else:
            # Update task status to indicate it's being resumed
            update_task_status(
                task_id,
                "resuming",
                f"Resuming {voice_type} audiobook generation for '{book_title}' from line {current_progress}/{total_lines}"
            )
        
        # Start the background task again with the same parameters        
        async def resume_background_generation():
            """Resume the background audiobook generation"""
            try:
                voice_mode = "multi_voice" if voice_type == "Multi-Voice" else "single_voice"
                
                await generate_audiobook_background(
                    output_format.lower(),
                    narrator_gender,
                    book_file,
                    book_title,
                    voice_mode,
                    task_id,
                )
                
            except asyncio.CancelledError:
                update_task_status(task_id, "cancelled", "Task was cancelled by user")
                raise
            except Exception as e:
                # Don't overwrite progress when marking as failed - let update_task_status preserve it
                update_task_status(task_id, "failed", "", str(e))
                print(f"Resume generation error: {e}")
                traceback.print_exc()
                raise
            finally:
                unregister_running_task(task_id)

        # Start the resume task
        try:
            background_task = asyncio.create_task(resume_background_generation())
            register_running_task(task_id, background_task)
            print(f"[DEBUG] Successfully created resume task for {task_id}")
            return gr.Info(f"Task resumed: {task_id}", duration=5)
        except Exception as e:
            print(f"[ERROR] Failed to create resume task: {e}")
            return gr.Warning(f"Failed to create resume task: {str(e)}")
        
    except Exception as e:
        print(f"Error continuing task: {e}")
        traceback.print_exc()
        return gr.Warning(f"Error continuing task: {str(e)}")


def load_current_settings():
    """Load current configuration settings for the UI"""
    tts_config = config_manager.get_tts_config()
    llm_config = config_manager.get_llm_config()
    
    return (
        # TTS Settings
        tts_config.get("base_url", "http://localhost:8880/v1"),
        tts_config.get("api_key", "not-needed"),
        tts_config.get("model", "kokoro"),
        tts_config.get("max_parallel_requests", 2),
        # LLM Settings
        llm_config.get("base_url", "http://localhost:1234/v1"),
        llm_config.get("api_key", "lm-studio"),
        llm_config.get("model_name", "qwen3-14b"),
        llm_config.get("no_think_mode", True)
    )


def save_tts_settings(base_url, api_key, model, max_parallel):
    """Save TTS settings and reload constants"""
    try:
        config_manager.update_tts_config(
            base_url=base_url,
            api_key=api_key,
            model=model,
            max_parallel=int(max_parallel)
        )
        # Reload constants to pick up new values
        reload_constants()
        return gr.Info("TTS settings saved successfully!", duration=5)
    except Exception as e:
        return gr.Warning(f"Error saving TTS settings: {str(e)}")


def save_llm_settings(base_url, api_key, model_name, no_think_mode):
    """Save LLM settings"""
    try:
        config_manager.update_llm_config(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            no_think_mode=no_think_mode
        )
        return gr.Info("LLM settings saved successfully!", duration=5)
    except Exception as e:
        return gr.Warning(f"Error saving LLM settings: {str(e)}")


def test_tts_connection(base_url, api_key, model):
    """Test TTS API connection"""
    try:
        import requests
        # Test endpoint
        test_url = f"{base_url.rstrip('/')}/audio/voices"
        headers = {"Authorization": f"Bearer {api_key}"} if api_key != "not-needed" else {}
        
        response = requests.get(test_url, headers=headers, timeout=5)
        if response.status_code == 200:
            return gr.Info("✅ TTS connection successful!", duration=5)
        else:
            return gr.Warning(f"❌ TTS connection failed: HTTP {response.status_code}")
    except Exception as e:
        return gr.Warning(f"❌ TTS connection failed: {str(e)}")


def test_llm_connection(base_url, api_key, model_name):
    """Test LLM API connection"""
    try:
        from openai import OpenAI
        
        client = OpenAI(base_url=base_url, api_key=api_key)
        # Try to get model list or make a simple completion
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1
        )
        return gr.Info("✅ LLM connection successful!", duration=5)
    except Exception as e:
        return gr.Warning(f"❌ LLM connection failed: {str(e)}")








with gr.Blocks(css=css, theme=gr.themes.Default()) as gradio_app:
    gr.Markdown("# 📖 Audiobook Creator")
    gr.Markdown("Create professional audiobooks from your ebooks in just a few steps.")

    # Settings Section
    with gr.Accordion("⚙️ Settings (TTS & LLM Configuration)", open=False):
        gr.Markdown("Configure your Text-to-Speech and Language Model settings. Changes are saved automatically.")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎤 Text-to-Speech (TTS) Settings")
                
                tts_base_url = gr.Textbox(
                    label="TTS Base URL",
                    placeholder="http://localhost:8880/v1",
                    info="Base URL for your TTS API endpoint"
                )
                
                tts_api_key = gr.Textbox(
                    label="TTS API Key",
                    placeholder="not-needed",
                    type="password",
                    info="API key for TTS service (use 'not-needed' if not required)"
                )
                
                tts_model = gr.Radio(
                    choices=["kokoro", "orpheus"],
                    label="TTS Model",
                    value="kokoro",
                    info="Choose between Kokoro or Orpheus TTS models"
                )
                
                tts_max_parallel = gr.Slider(
                    minimum=1,
                    maximum=30,
                    step=1,
                    value=2,
                    label="Max Parallel Requests",
                    info="Number of parallel TTS requests (adjust based on your hardware)"
                )
                
                with gr.Row():
                    save_tts_btn = gr.Button("💾 Save TTS Settings", variant="primary", size="sm")
                    test_tts_btn = gr.Button("🔍 Test TTS Connection", variant="secondary", size="sm")
            
            with gr.Column():
                gr.Markdown("### 🧠 Language Model (LLM) Settings")
                
                llm_base_url = gr.Textbox(
                    label="LLM Base URL",
                    placeholder="http://localhost:1234/v1",
                    info="Base URL for your LLM API endpoint"
                )
                
                llm_api_key = gr.Textbox(
                    label="LLM API Key",
                    placeholder="lm-studio",
                    type="password",
                    info="API key for LLM service"
                )
                
                llm_model_name = gr.Textbox(
                    label="LLM Model Name",
                    placeholder="qwen3-14b",
                    info="Name of the LLM model to use"
                )
                
                llm_no_think_mode = gr.Checkbox(
                    label="Disable Think Mode",
                    value=True,
                    info="Disable thinking mode for faster inference (for models like Qwen3, R1)"
                )
                
                with gr.Row():
                    save_llm_btn = gr.Button("💾 Save LLM Settings", variant="primary", size="sm")
                    test_llm_btn = gr.Button("🔍 Test LLM Connection", variant="secondary", size="sm")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown('<div class="step-heading">📚 Step 1: Book Details</div>')

            book_title = gr.Textbox(
                label="Book Title",
                placeholder="Enter the title of your book",
                info="This will be used for finding the protagonist of the book in the character identification step",
            )

            book_input = gr.File(label="Upload Book", interactive=True)

            text_decoding_option = gr.Radio(
                ["textract", "calibre"],
                label="Text Extraction Method",
                value="textract",
                info="Use calibre for better formatted results, wider compatibility for ebook formats. You can try both methods and choose based on the output result.",
            )

            validate_btn = gr.Button("Validate Book", variant="primary")

    # Disable upload until title is entered
    def enable_upload(title):
        return gr.update(interactive=bool(title and title.strip()))

    book_title.change(
        enable_upload,
        inputs=[book_title],
        outputs=[book_input],
        queue=False,
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                '<div class="step-heading">✂️ Step 2: Extract & Edit Content</div>'
            )

            convert_btn = gr.Button("Extract Text", variant="primary")

            with gr.Accordion("Editing Tips", open=True):
                gr.Markdown(
                    """
                * Remove unwanted sections: Table of Contents, About the Author, Acknowledgements
                * Fix formatting issues or OCR errors
                * Check for chapter breaks and paragraph formatting
                """
                )

            text_output = gr.Textbox(
                label="Edit Book Content",
                placeholder="Extracted text will appear here for editing",
                interactive=True,
                lines=15,
                elem_id="edit-book-content",
            )

            with gr.Row():
                goto_start_btn = gr.Button(
                    "📄 Goto Beginning", variant="secondary", size="sm"
                )
                goto_end_btn = gr.Button("📄 Goto End", variant="secondary", size="sm")

            save_btn = gr.Button("Save Edited Text", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                '<div class="step-heading">🧩 Step 3: Character Identification (Optional)</div>'
            )

            identify_btn = gr.Button("Identify Characters", variant="primary")

            with gr.Accordion("Why Identify Characters?", open=True):
                gr.Markdown(
                    """
                * Improves multi-voice narration by assigning different voices to characters
                * Creates more engaging audiobooks with distinct character voices
                * Skip this step if you prefer single-voice narration
                """
                )

            character_output = gr.Textbox(
                label="Character Identification Progress",
                placeholder="Character identification progress will be shown here",
                interactive=False,
                lines=3,
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown('<div class="step-heading">🎧 Step 4: Generate Audiobook</div>')

            with gr.Row():
                voice_type = gr.Radio(
                    ["Single Voice", "Multi-Voice"],
                    label="Narration Type",
                    value="Single Voice",
                    info="Multi-Voice requires character identification",
                )

                narrator_gender = gr.Radio(
                    ["male", "female"],
                    label="Choose whether you want the book to be read in a male or female voice",
                    value="female",
                )

                output_format = gr.Dropdown(
                    choices=[
                        ("M4B (Chapters & Cover)", "m4b"),
                        ("AAC Audio", "aac"),
                        ("M4A Audio", "m4a"),
                        ("MP3 Audio", "mp3"),
                        ("WAV Audio", "wav"),
                        ("OPUS Audio", "opus"),
                        ("FLAC Audio", "flac"),
                        ("PCM Audio", "pcm"),
                    ],
                    label="Output Format",
                    value="m4b",
                    info="M4B supports chapters and cover art",
                )

            with gr.Row():
                generate_btn = gr.Button("Generate Audiobook", variant="primary")
                # Add a quick cancel button right next to generate (initially hidden)
                quick_cancel_btn = gr.Button("⏹️ Stop Generation", variant="stop", visible=False)

            audio_output = gr.Textbox(
                label="Generation Progress",
                placeholder="Generation progress will be shown here",
                interactive=False,
                lines=3,
            )

            # Add a new File component for downloading the audiobook
            with gr.Group(visible=False) as download_box:
                gr.Markdown("### 📥 Download Your Audiobook")
                audiobook_file = gr.File(
                    label="Download Generated Audiobook",
                    interactive=False,
                    type="filepath",
                )

        # Past generated files section
    with gr.Row():
        with gr.Column():
            gr.Markdown('<div class="step-heading">📂 Generated Audiobooks</div>')

            with gr.Row():
                refresh_btn = gr.Button(
                    "🔄 Refresh List", variant="secondary", size="sm"
                )
                clear_tasks_btn = gr.Button(
                    "🧹 Clear All Temp Files", variant="secondary", size="sm"
                )
                
            # Display temp directory information
            temp_files_info = gr.Markdown(
                value="Click refresh to see temp files information.",
                visible=True,
                elem_id="temp-files-info"
            )

            # Task management section
            with gr.Group(visible=False) as cancel_task_group:
                gr.Markdown("### ⏹️ Manage Running Tasks")
                active_tasks_dropdown = gr.Dropdown(
                    label="Select task to manage",
                    choices=[],
                    visible=True,
                    interactive=True,
                )
                with gr.Row():
                    cancel_task_btn = gr.Button("⏹️ Stop Task", variant="stop", size="sm")
                    continue_task_btn = gr.Button("▶️ Continue Task", variant="primary", size="sm")

            past_files_display = gr.Markdown(
                value="Click refresh to see generated audiobooks.", visible=True
            )

            with gr.Row():
                past_files_dropdown = gr.Dropdown(
                    label="Select a past audiobook",
                    choices=[],
                    visible=False,
                    interactive=True,
                )
                past_file_download = gr.File(
                    label="Download Selected Audiobook",
                    interactive=False,
                    type="filepath",
                    visible=False,
                )

            with gr.Row():
                delete_btn = gr.Button(
                    "🗑️ Delete Selected",
                    variant="stop",
                    size="sm",
                    visible=False,
                )
                delete_all_btn = gr.Button(
                    "🗑️💥 Delete All Audiobooks",
                    variant="stop",
                    size="sm",
                    visible=True,
                )

            # Confirmation dialog for single file deletion (hidden by default)
            with gr.Group(visible=False) as delete_confirmation:
                gr.Markdown("### ⚠️ Confirm Deletion")
                delete_info_display = gr.Markdown("")

                with gr.Row():
                    confirm_delete_btn = gr.Button(
                        "🗑️ Yes, Delete", variant="stop", size="sm"
                    )
                    cancel_delete_btn = gr.Button(
                        "❌ Cancel", variant="secondary", size="sm"
                    )

            # Confirmation dialog for delete all operation (hidden by default)
            with gr.Group(visible=False) as delete_all_confirmation:
                gr.Markdown("### ⚠️ Confirm Delete All")
                delete_all_info_display = gr.Markdown("")

                with gr.Row():
                    confirm_delete_all_btn = gr.Button(
                        "🗑️💥 Yes, Delete All", variant="stop", size="sm"
                    )
                    cancel_delete_all_btn = gr.Button(
                        "❌ Cancel", variant="secondary", size="sm"
                    )

    # Connections with proper handling of Gradio notifications
    validate_btn.click(
        validate_book_upload, inputs=[book_input, book_title], outputs=[]
    )

    convert_btn.click(
        text_extraction_wrapper,
        inputs=[book_input, text_decoding_option, book_title],
        outputs=[text_output],
        queue=True,
    )

    save_btn.click(
        save_book_wrapper, inputs=[text_output, book_title], outputs=[], queue=True
    )

    identify_btn.click(
        identify_characters_wrapper,
        inputs=[book_title],
        outputs=[character_output],
        queue=True,
    )

    # Update the generate_audiobook_wrapper to output both progress text and file path
    generate_btn.click(
        generate_audiobook_wrapper,
        inputs=[voice_type, narrator_gender, output_format, book_input, book_title],
        outputs=[audio_output, audiobook_file],
        queue=True,
    ).then(
        # Make the download box visible after generation completes successfully
        lambda x: (
            gr.update(visible=True) if x is not None else gr.update(visible=False)
        ),
        inputs=[audiobook_file],
        outputs=[download_box],
    ).then(
        # Always refresh past files list after generation (successful or failed)
        refresh_past_files_with_continue,
        outputs=[
            past_files_display,
            past_files_dropdown,
            delete_btn,
            active_tasks_dropdown,
            cancel_task_group,
        ],
    ).then(
        # Also refresh temp files info
        get_temp_directory_info,
        outputs=[temp_files_info],
    ).then(
        # Hide quick cancel button after generation
        lambda: gr.update(visible=False),
        outputs=[quick_cancel_btn],
    )

    # Show quick cancel button when generation starts
    generate_btn.click(
        lambda: gr.update(visible=True),
        outputs=[quick_cancel_btn],
        queue=False,
    )

    # Handle quick cancel button
    def quick_cancel_current_task():
        """Cancel the most recent active task"""
        active_tasks = get_active_tasks()
        if active_tasks:
            # Cancel the most recent task (last in the list)
            latest_task = active_tasks[-1]
            task_id = latest_task['id']
            success, message = cancel_task(task_id)
            if success:
                return gr.Info(f"Generation stopped: {message}", duration=5)
            else:
                return gr.Warning(f"Failed to stop generation: {message}")
        else:
            return gr.Warning("No active generation to stop.")

    quick_cancel_btn.click(
        quick_cancel_current_task,
        outputs=[],
    ).then(
        # Hide the quick cancel button after cancelling
        lambda: gr.update(visible=False),
        outputs=[quick_cancel_btn],
    ).then(
        # Refresh the display after cancelling
        refresh_past_files_with_continue,
        outputs=[
            past_files_display,
            past_files_dropdown,
            delete_btn,
            active_tasks_dropdown,
            cancel_task_group,
        ],
    )

    # Refresh past files when the refresh button is clicked
    refresh_btn.click(
        refresh_past_files_with_continue,
        outputs=[
            past_files_display,
            past_files_dropdown,
            delete_btn,
            active_tasks_dropdown,
            cancel_task_group,
            
        ],
    ).then(
        get_temp_directory_info,
        outputs=[temp_files_info],
    )

   


    # Cancel task when the cancel button is clicked
    cancel_task_btn.click(
        cancel_task_wrapper,
        inputs=[active_tasks_dropdown],
        outputs=[],
    ).then(
        # Refresh the display after cancelling task
        refresh_past_files_with_continue,
        outputs=[
            past_files_display,
            past_files_dropdown,
            delete_btn,
            active_tasks_dropdown,
            cancel_task_group,
            
        ],
    )

    # Continue task when the continue button is clicked
    continue_task_btn.click(
        continue_task_wrapper,
        inputs=[active_tasks_dropdown],
        outputs=[],
    ).then(
        # Refresh the display after continuing task
        refresh_past_files_with_continue,
        outputs=[
            past_files_display,
            past_files_dropdown,
            delete_btn,
            active_tasks_dropdown,
            cancel_task_group,
            
        ],
    )

    # Update the download file when a past file is selected
    past_files_dropdown.change(
        lambda x: gr.update(value=x, visible=True) if x else gr.update(visible=False),
        inputs=[past_files_dropdown],
        outputs=[past_file_download],
    )

    # Show confirmation dialog when delete button is clicked
    delete_btn.click(
        get_selected_file_info,
        inputs=[past_files_dropdown],
        outputs=[delete_info_display],
    ).then(lambda: gr.update(visible=True), outputs=[delete_confirmation])

    # Actually delete the file when confirmed
    confirm_delete_btn.click(
        delete_audiobook_file, inputs=[past_files_dropdown], outputs=[]
    ).then(
        # Hide confirmation dialog
        lambda: gr.update(visible=False),
        outputs=[delete_confirmation],
    ).then(
        # Refresh the list after deletion
        refresh_past_files_with_continue,
        outputs=[
            past_files_display,
            past_files_dropdown,
            delete_btn,
            active_tasks_dropdown,
            cancel_task_group,
            
        ],
    ).then(
        # Clear the download file after deletion
        lambda: gr.update(value=None, visible=False),
        outputs=[past_file_download],
    )

    # Cancel deletion
    cancel_delete_btn.click(
        lambda: gr.update(visible=False), outputs=[delete_confirmation]
    )

    # Show confirmation dialog when delete all button is clicked
    delete_all_btn.click(
        get_all_audiobooks_info,
        outputs=[delete_all_info_display],
    ).then(lambda: gr.update(visible=True), outputs=[delete_all_confirmation])

    # Actually delete all files when confirmed
    confirm_delete_all_btn.click(
        delete_all_audiobooks, outputs=[]
    ).then(
        # Hide confirmation dialog
        lambda: gr.update(visible=False),
        outputs=[delete_all_confirmation],
    ).then(
        # Refresh the list after deletion
        refresh_past_files_with_continue,
        outputs=[
            past_files_display,
            past_files_dropdown,
            delete_btn,
            active_tasks_dropdown,
            cancel_task_group,
        ],
    ).then(
        # Clear the download file after deletion
        lambda: gr.update(value=None, visible=False),
        outputs=[past_file_download],
    )

    # Cancel delete all
    cancel_delete_all_btn.click(
        lambda: gr.update(visible=False), outputs=[delete_all_confirmation]
    )

    # Load past files when the app starts
    gradio_app.load(
        refresh_past_files_with_continue,
        outputs=[
            past_files_display,
            past_files_dropdown,
            delete_btn,
            active_tasks_dropdown,
            cancel_task_group,
            
        ],
    ).then(
        get_temp_directory_info,
        outputs=[temp_files_info],
    )

    # Clear temp files when the clear button is clicked
    clear_tasks_btn.click(
        clear_temp_files,
        outputs=[],
    ).then(
        # Refresh the display after clearing temp files
        refresh_past_files_with_continue,
        outputs=[
            past_files_display,
            past_files_dropdown,
            delete_btn,
            active_tasks_dropdown,
            cancel_task_group,
        ],
    ).then(
        get_temp_directory_info,
        outputs=[temp_files_info],
    )

    # Wire up goto buttons for text scrolling
    goto_start_btn.click(
        None,
        js="() => { const el = document.querySelector('#edit-book-content textarea'); if (el) el.scrollTop = 0; }",
        outputs=[],
    )

    goto_end_btn.click(
        None,
        js="() => { const el = document.querySelector('#edit-book-content textarea'); if (el) el.scrollTop = el.scrollHeight; }",
        outputs=[],
    )

    # Settings event handlers
    save_tts_btn.click(
        save_tts_settings,
        inputs=[tts_base_url, tts_api_key, tts_model, tts_max_parallel],
        outputs=[]
    )
    
    save_llm_btn.click(
        save_llm_settings,
        inputs=[llm_base_url, llm_api_key, llm_model_name, llm_no_think_mode],
        outputs=[]
    )
    
    test_tts_btn.click(
        test_tts_connection,
        inputs=[tts_base_url, tts_api_key, tts_model],
        outputs=[]
    )
    
    test_llm_btn.click(
        test_llm_connection,
        inputs=[llm_base_url, llm_api_key, llm_model_name],
        outputs=[]
    )

    # Load current settings when the app starts
    gradio_app.load(
        load_current_settings,
        outputs=[
            tts_base_url, tts_api_key, tts_model, tts_max_parallel,
            llm_base_url, llm_api_key, llm_model_name, llm_no_think_mode
        ]
    )

app = gr.mount_gradio_app(app, gradio_app, path="/")  # Mount Gradio at root

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
