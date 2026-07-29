from dotenv import load_dotenv

from veto.evaluation.vqa.base import VqaBackend
from veto.evaluation.vqa.backends.gemma3_vl import Gemma3VqaBackend
from veto.evaluation.vqa.backends.gemini import GeminiVqaBackend
from veto.evaluation.vqa.backends.gpt import GptVqaBackend
from veto.evaluation.vqa.backends.llava_onevision import LlavaOnevisionVqaBackend
from veto.evaluation.vqa.backends.qwen25_vl import Qwen25VlVqaBackend

load_dotenv()


def build_vqa_backend(name: str) -> VqaBackend:
    key = name.strip().lower().replace("_", "-")
    if key in ("gemini",):
        return GeminiVqaBackend()
    if key in ("qwen", "qwen2.5-vl", "qwen25-vl"):
        return Qwen25VlVqaBackend()
    if key in ("llava", "llava-onevision", "onevision"):
        return LlavaOnevisionVqaBackend()
    if key in ("gemma3", "gemma-3", "gemma-3-it"):
        return Gemma3VqaBackend()
    if key in ("gpt", "openai", "gpt-5.5"):
        return GptVqaBackend()
    raise ValueError(
        f"Unknown VQA model {name!r}. Use: gemini, gpt, qwen, llava, gemma3"
    )
