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

import subprocess
import shutil
import os
import traceback
import sys


def get_system_python_paths():
    """
    Returns a list of directories containing Python packages in the system
    excluding the virtual environment.

    The function works by iterating over common base directories for Python
    packages and checking if they exist. The directories are then added to a
    list which is returned.
    """

    # Get Python version
    python_version = subprocess.run(
        [
            "/usr/bin/python3",
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Common base directories for Python packages
    base_dirs = [
        "/usr/lib/python3",  # Default installation directory for Python packages
        "/usr/local/lib/python3",  # Directory for user-installed packages
        f"/usr/lib/python{python_version}",  # Specific Python version directory
        f"/usr/local/lib/python{python_version}",  # Specific Python version directory
    ]

    # Common package directories
    package_dirs = [
        "dist-packages",  # Debian/Ubuntu packages
        "site-packages",  # Python packages installed using pip
    ]

    # Find all existing paths
    system_paths = []
    for base in base_dirs:
        for package_dir in package_dirs:
            path = os.path.join(base, package_dir)
            if os.path.exists(path):
                system_paths.append(path)

        # Also check for direct dist-packages in python3 directory
        if os.path.exists(base) and os.path.isdir(base):
            system_paths.append(base)

    return list(set(system_paths))  # Remove duplicates


def check_if_calibre_is_installed():
    """
    Checks if Calibre is installed.

    Returns True if Calibre is installed and False otherwise.
    """
    # Check if Calibre is installed by checking if either the `calibre` or
    # `ebook-convert` command is available in the PATH.
    calibre_installed = shutil.which("calibre") or shutil.which("ebook-convert")

    if calibre_installed:
        return True
    else:
        return False


def check_if_ffmpeg_is_installed():
    """
    Checks if FFmpeg is installed.

    Returns True if FFmpeg is installed and False otherwise.
    """
    ffmpeg_installed = shutil.which("ffmpeg")

    if ffmpeg_installed:
        # If the command is available in the PATH, FFmpeg is installed
        return True
    else:
        # If the command is not available in the PATH, FFmpeg is not installed
        return False


def get_venv_python_path(venv_path):
    """
    Returns the path to the Python executable in the given virtual environment.
    """
    if not venv_path:
        return None
    python_bin = os.path.join(venv_path, "bin", "python")
    if os.path.exists(python_bin):
        return python_bin
    python3_bin = os.path.join(venv_path, "bin", "python3")
    if os.path.exists(python3_bin):
        return python3_bin
    return None


def run_shell_command_without_virtualenv(command, venv_path=None):
    """
    Runs a shell command without using a virtual environment, or within a specified venv if provided.
    Args:
        command (str): The shell command to run.
        venv_path (str, optional): Path to the virtual environment to use. Defaults to None.
    Returns:
        subprocess.CompletedProcess: The result of the command execution.
    """
    original_pythonpath = os.environ.get("PYTHONPATH", "")
    try:
        modified_env = os.environ.copy()
        # If venv_path is provided, use its bin directory in PATH and its Python
        if venv_path:
            venv_bin = os.path.join(venv_path, "bin")
            modified_env["PATH"] = venv_bin + os.pathsep + modified_env.get("PATH", "")
            venv_python = get_venv_python_path(venv_path)
        else:
            venv_python = None
        # Get system Python paths automatically if not using venv
        if not venv_path:
            system_paths = get_system_python_paths()
            if not system_paths:
                raise Exception("No system Python paths found")
            modified_env["PYTHONPATH"] = ":".join(system_paths + [original_pythonpath])
        # Run the command with modified environment
        if command.endswith(".py") or command.startswith("python"):
            if venv_python:
                cmd = f"{venv_python} {command if command.endswith('.py') else ' '.join(command.split()[1:])}"
            else:
                cmd = f"/usr/bin/python3 {command}"
        else:
            cmd = command
        result = subprocess.run(
            cmd, shell=True, env=modified_env, capture_output=True, text=True
        )
        return result
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return None


def run_shell_command(command, venv_path=None):
    """
    Runs a shell command, optionally within a specified virtual environment.
    Args:
        command (str): The shell command to run.
        venv_path (str, optional): Path to the virtual environment to use. Defaults to None.
    Returns:
        subprocess.CompletedProcess: The result of the command execution.
    """
    try:
        if venv_path:
            # Use the venv-aware function
            return run_shell_command_without_virtualenv(command, venv_path=venv_path)
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.stderr:
            raise Exception(result.stderr)
        return result
    except Exception as e:
        print(e)
        traceback.print_exc()
        print(
            "Error in run_shell_command, running  run_shell_command_without_virtualenv"
        )
        return run_shell_command_without_virtualenv(command, venv_path=venv_path)
