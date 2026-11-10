"""Explicit checkpoint architecture routing; old runs default to Transformer."""
from src.models.transformer_mixed import MixedForecastTransformer


def create_forecast_model(kind, config):
    if kind == 'transformer_mixed':
        return MixedForecastTransformer(**config)
    if kind == 'hybrid_mixed':
        from src.models.hybrid_mixed import HybridMixedForecast
        return HybridMixedForecast(**config)
    raise ValueError(f'Unsupported forecast model kind: {kind}')
