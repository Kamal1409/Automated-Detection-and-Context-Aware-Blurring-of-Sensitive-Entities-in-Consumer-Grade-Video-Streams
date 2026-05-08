try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from llm_manager import LLM
from audio_processor import AUDIO
from video_processor import VIDEO


def main():
    llm = LLM()
    audio_processor = AUDIO()
    video_processor = VIDEO()
    print(f"LLM enabled: {llm.enabled}, model: {llm.model}")
    print(f"Audio processor: {audio_processor}")
    print(f"Video processor: {video_processor}")


if __name__ == "__main__":
    main()
