from .encoder import SLatEncoder, ElasticSLatEncoder
from .decoder_gs import SLatGaussianDecoder, ElasticSLatGaussianDecoder
from .decoder_rf import SLatRadianceFieldDecoder, ElasticSLatRadianceFieldDecoder


def __getattr__(name):
    if name in {"SLatMeshDecoder", "ElasticSLatMeshDecoder"}:
        from .decoder_mesh import SLatMeshDecoder, ElasticSLatMeshDecoder
        value = {
            "SLatMeshDecoder": SLatMeshDecoder,
            "ElasticSLatMeshDecoder": ElasticSLatMeshDecoder,
        }[name]
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
