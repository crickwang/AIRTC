from pydub import AudioSegment
import argparse

def convert_audio(input_file, output_file, format):
    """
    Convert audio file to a specified format.
    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to the output audio file.
        format (str): Output audio format (e.g., mp3, wav).
    """
    audio = AudioSegment.from_file(input_file)
    audio.export(output_file, format=format)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert audio files to different formats.")
    parser.add_argument("i", help="Path to the input audio file.", )
    parser.add_argument("o", help="Path to the output audio file.")
    parser.add_argument("f", help="Output audio format (e.g., mp3, wav).")
    args = parser.parse_args()

    convert_audio(args.i, args.o, args.f)
