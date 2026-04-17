from pcb.compressors.base import BaseCompressor, CompressionResult
from pcb.compressors.no_compression import NoCompressionCompressor
from pcb.compressors.tfidf import TFIDFCompressor
from pcb.compressors.selective_context import SelectiveContextCompressor
from pcb.compressors.llmlingua import LLMLinguaCompressor, LLMLingua2Compressor
from pcb.compressors.pipeline import PipelineCompressor

ALL_COMPRESSORS = {
    "no_compression": NoCompressionCompressor,
    "smart": PipelineCompressor,
    "tfidf": TFIDFCompressor,
    "selective_context": SelectiveContextCompressor,
    "llmlingua": LLMLinguaCompressor,
    "llmlingua2": LLMLingua2Compressor,
}

__all__ = [
    "BaseCompressor",
    "CompressionResult",
    "NoCompressionCompressor",
    "PipelineCompressor",
    "TFIDFCompressor",
    "SelectiveContextCompressor",
    "LLMLinguaCompressor",
    "LLMLingua2Compressor",
    "ALL_COMPRESSORS",
]
