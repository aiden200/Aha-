import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers import Cache

class SlidingWindowCache(Cache):
    """
    A simple sliding window cache that keeps the last `window_length` tokens.
    This implementation does not perform RoPE re-rotation and is more robust for benchmarking.
    """
    def __init__(self, window_length: int) -> None:
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.window_length = window_length
        self._seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # [bsz, num_heads, seq_len, head_dim]
        # Append to the cache list if it's a new layer
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
            return key_states, value_states

        # Concatenate the old cache with the new states
        full_keys = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
        full_values = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # Slice the concatenated tensors to keep only the last `window_length` tokens
        sliding_keys = full_keys[:, :, -self.window_length:]
        sliding_values = full_values[:, :, -self.window_length:]

        # Update the cache with the new sliding window
        self.key_cache[layer_idx] = sliding_keys
        self.value_cache[layer_idx] = sliding_values

        return sliding_keys, sliding_values

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]

    def get_max_length(self) -> Optional[int]:
        """DEPRICATED"""
        return self.window_length