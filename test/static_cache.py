import torch
from typing import Any, Dict, List, Optional, Tuple
from transformers import Cache

class TrulyStaticCache(Cache):
    """
    A Cache that only stores the first `window_size` set of key-value states it sees and
    never updates, effectively creating a static prefix cache of a fixed size.
    """
    # --- MODIFIED ---
    def __init__(self, window_size: int):
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self.window_size = window_size
        self._seen_tokens = 0

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # On the first pass for each layer, store a slice of the states up to the window size.
        if len(self.key_cache) <= layer_idx:
            # --- MODIFIED ---
            # Slice the input tensors to store only the desired window size
            self.key_cache.append(key_states[:, :, :self.window_size])
            self.value_cache.append(value_states[:, :, :self.window_size])

        # On all subsequent passes, DO NOTHING.
        # The cache remains frozen with the initial window.
        
        # Always return the originally stored states.
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


    def get_max_length(self) -> Optional[int]:
        """DEPRICATED"""
        return self.window_size

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states."""
        if len(self.key_cache) <= layer_idx:
            return 0
        return self.key_cache[layer_idx].shape[-2]