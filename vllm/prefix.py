from typing import Dict, List, Sequence, Tuple, Optional
from collections import OrderedDict

from vllm.block import BlockTable


class Prefix:
    """Data and states associated with a prefix of prompt tokens for multiple
    sequence groups.

    Args:
        prefix_id: The id of the prefix in the prefix pool.
        token_ids: The token ids of the prefix.
        block_size: The block size of the executed model.
    """

    def __init__(
        self,
        token_ids: Sequence[int],
        block_size: int,
    ) -> None:
        self.token_ids = tuple(token_ids)
        self.block_size = block_size
        self.length = len(token_ids)
        self.hash = hash(token_ids)
        assert self.length % block_size == 0
        self.block_table: Optional[BlockTable] = None

    @property
    def allocated(self) -> bool:
        return self.block_table is not None

    def get_num_blocks(self) -> int:
        return self.length // self.block_size

    def get_block_numbers(self) -> List[int]:
        return [block.block_number for block in self.block_table]

    def get_length(self) -> int:
        return self.length

    def __hash__(self) -> int:
        return self.hash
    
    def __eq__(self, other) -> bool:
        if isinstance(other, Prefix):
            return self.hash == other.hash
        return NotImplemented
    
    def set_block_table(self, block_table: BlockTable) -> None:
        self.block_table = block_table.copy()


class PrefixPool:
    """Manages all the prompt prefixes.

    Args:
        block_size: The block size of the executed model.

    Attributes:
        prefixes: A list of all the prefixes.
        block_size: The block size of the executed model.
        max_capacity: The maximum number of prefixes to store. By default it stores all the prefixes,
            so adding this parameter does not modify the behavior of the previous version of the class.
    """

    def __init__(
        self,
        block_size: int,
        max_capacity: Optional[int] = None
    ) -> None:
        # Dictionary from hash of prefix token ids to prefix
        self.prefixes: Dict[int, Prefix] = OrderedDict()
        self.block_size = block_size
        self.max_capacity = max_capacity

    def _truncate_token_ids(self, token_ids: Sequence[int]) -> Tuple[int]:
        new_length = len(token_ids) // self.block_size * self.block_size
        return tuple(token_ids[:new_length])

    def add_or_get_prefix(self, token_ids: Sequence[int]) -> Optional[Prefix]:
        token_ids = self._truncate_token_ids(token_ids)
        if len(token_ids) == 0:
            # Prefix is empty.
            return None
        prefix = Prefix(token_ids, self.block_size)
        prefix_hash = hash(prefix)
        if prefix_hash not in self.prefixes:
            if self.max_capacity is not None and len(self.prefixes) >= self.max_capacity:
                # Remove the oldest prefix.
                self.prefixes.popitem(last=False)
            self.prefixes[prefix_hash] = prefix
        return self.prefixes[prefix_hash]