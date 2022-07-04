from typing import List, Tuple

import torch


class Hyp(object):

    # TODO(Mddct): refine two states to one cache
    def __init__(
            self, val: Tuple[List[int], float, torch.Tensor,
                             torch.Tensor]) -> None:
        self.ids = val[0]
        self.logp_ = val[1]
        self.state_m = val[2]
        self.state_c = val[3]

    def __lt__(self, other) -> bool:
        return self.logp > other.logp

    def __eq__(self, other) -> bool:
        return self.logp == other.logp

    def push(self, id: int, logp: float, state_m: torch.Tensor,
             state_c: torch.Tensor):
        self.push_add(id, logp=logp)
        self.state_m = state_m
        self.state_c = state_c

    def push_add(self, id: int, logp: float):
        if id != 0:
            self.ids.append(id)
        self.logp_ = self.logp_ + logp

    @property
    def logp(self) -> float:
        return self.logp_

    @property
    def prefix(self) -> List[int]:
        return self.ids

    def state(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.state_m, self.state_c

    def end(self) -> int:
        assert len(self.ids) > 0
        return self.ids[-1]


# a = Hyp(([0], 0.0, torch.ones(1, 1), torch.ones(1, 1)))
# b = Hyp(([1], 1.0, torch.ones(1, 1), torch.ones(1, 1)))

# l = []
# heapq.heappush(l, b)
# heapq.heappush(l, a)

# l2 = copy.deepcopy(l)
# heapq.heappop(l2)

# print(l2[0].prefix)
# print(l[0].prefix)
