"""Import shim for ragas 0.4.x with langchain-community >= 0.4.

ragas.llms.base unconditionally imports the Vertex AI classes, which newer
langchain-community releases removed. We never use Vertex AI, so we register
placeholder classes before ragas is imported. Import this module before `ragas`.
"""
import importlib
import sys
import types


def _ensure(module_name: str, attr: str) -> None:
    try:
        mod = importlib.import_module(module_name)
        if hasattr(mod, attr):
            return
    except ImportError:
        mod = types.ModuleType(module_name)
        sys.modules[module_name] = mod
    setattr(mod, attr, type(attr, (), {}))   # placeholder class, only used in isinstance checks


_ensure("langchain_community.chat_models.vertexai", "ChatVertexAI")
_ensure("langchain_community.llms", "VertexAI")
