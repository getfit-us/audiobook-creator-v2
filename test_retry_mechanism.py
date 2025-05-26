#!/usr/bin/env python3
"""
Test script for the TTS retry mechanism
"""

import asyncio
import sys
import os

# Add the current directory to the path so we can import from generate_audiobook
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_audiobook import generate_tts_with_retry
from openai import AsyncOpenAI
from config.constants import API_KEY, BASE_URL, MODEL


async def test_retry_mechanism():
    """Test the retry mechanism with a simple TTS call"""

    # Create client
    client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

    # Test parameters
    voice = "af_heart" if MODEL == "kokoro" else "tara"
    text = "This is a test of the retry mechanism."
    response_format = "wav" if MODEL == "orpheus" else "aac"

    print(f"Testing retry mechanism with:")
    print(f"  Model: {MODEL}")
    print(f"  Voice: {voice}")
    print(f"  Text: {text}")
    print(f"  Format: {response_format}")
    print()

    try:
        # Test the retry mechanism
        audio_buffer = await generate_tts_with_retry(
            client=client,
            model=MODEL,
            voice=voice,
            text=text,
            response_format=response_format,
            speed=0.85,
            max_retries=3,  # Use fewer retries for testing
            task_id=None,
        )

        print(f"✅ Success! Generated {len(audio_buffer)} bytes of audio")

        # Save test audio file
        test_filename = f"test_retry_output.{response_format}"
        with open(test_filename, "wb") as f:
            f.write(audio_buffer)
        print(f"✅ Saved test audio to {test_filename}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("Testing TTS retry mechanism...")
    success = asyncio.run(test_retry_mechanism())

    if success:
        print("\n🎉 Retry mechanism test passed!")
    else:
        print("\n💥 Retry mechanism test failed!")
        sys.exit(1)
