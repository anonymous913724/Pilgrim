import torch
import torch.nn.functional as F
from typing import Optional

class MixupWithMemory():
    def __init__(self, num_classes, d_emb, lamb=0.5, device=None, memory_limit=500):
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        # dictionary mapping query to context and target
        self.memory_info = dict()
        self.num_memory_items = 0

        # example weight for sampling based on class occurence in each batch
        self.class_counter = dict(zip(range(num_classes), [0] * num_classes))

        self.d_emb = d_emb
        self.lamb = lamb
        self.memory_limit = memory_limit
        self.num_classes = num_classes

        self.mode = "train"

    def get_mixup_samples(
        self,
        context: torch.Tensor,
        query: torch.Tensor,
        target: torch.Tensor,
    ):
        """
        Retrieve mixup samples for a set (batch) of examples. Returns mixup information, where indicies align with corresponding mixup examples.

        Args:
            context: [d] or [n, d] tensor
            query: [d] or [n, d] tensor, ALL n VECTORS ARE THE SAME (i.e. same query)
            target: [1] or [n,] tensor

        Returns:
            torch.Tensor: [n, d] context tensors after mixup,
            torch.Tensor: [n] for binary targets, [n, num_classes] for multi-class, targets after performing mixup
            torch.Tensor: [n] original target of first sampled value of mixup pairs
            torch.Tensor: [n] original target of second sampled value of mixup pairs
            torch.Tensor: [n, d] query tensors (mixup not performed)
        """
        if context.ndim != 2:
            context = context.unsqueeze(0)
        if query.ndim != 2:
            query = query.unsqueeze(0)
        if target.ndim != 1:
            target = target.unsqueeze(0)

        n = context.shape[0]

        # 1) Mixup sampling
        # for each example in the batch, sample another example from memory with the same task

        # query as tuple for dictionary access/hashing
        q = tuple(query[0].tolist())

        # add a item for the new query
        if q not in self.memory_info:
            self.memory_info[q] = {
                "context": [],
                "target": [],
                "num_items": 0
            }

        
        context_mixup = []
        query_mixup = []
        target_mixup = []

        if self.memory_info[q]["num_items"] == 0:
            # if nothing in memory for the query, just do mixup with batch
            n_mixup_batch = n
        else:
            # 1/3 of mixup is done within batch
            n_mixup_batch = int(n * (1/3))

        # mixup within batch
        for i in range(0, n_mixup_batch):
            idx = torch.randint(n, size=()).to(torch.long)

            context_mixup.append(context[idx])
            query_mixup.append(query[idx])
            target_mixup.append(target[idx])


            self.class_counter[int(target[i])] += 1

        # mixup with memory
        for i in range(n_mixup_batch, n):
            idx = torch.randint(self.memory_info[q]["num_items"], size=()).to(torch.long)

            context_mixup.append(self.memory_info[q]["context"][idx])
            query_mixup.append(query[i])
            target_mixup.append(self.memory_info[q]["target"][idx])

            self.class_counter[int(target[i])] += 1
        
        # 2) Actual mixup
        context_mixup = torch.stack(context_mixup, dim=0)
        query_mixup = torch.stack(query_mixup, dim=0)
        target_mixup = torch.stack(target_mixup).squeeze()
        
        context_i = context # [n, d_emb]
        context_j = context_mixup # [n ,d_emb]
        mixup_query = query_mixup # [n, d_emb]
        target_i = target.to(torch.long) # [n]
        target_j = target_mixup.to(torch.long) # [n]

        
        target_i_one_hot = F.one_hot(target_i, num_classes=self.num_classes)
        target_j_one_hot = F.one_hot(target_j, num_classes=self.num_classes)
        
        context_i = self.lamb * context
        context_j = (1 - self.lamb) * context_mixup

        mixup_context = context_i + context_j # [n, d_emb]

        target_i_one_hot = self.lamb * target_i_one_hot
        target_j_one_hot = (1 - self.lamb) * target_j_one_hot

        mixup_target = target_i_one_hot + target_j_one_hot # [n, d_emb]

        # 3) Update memory based on minority class
        if self.mode == "train":
            min_class = min(self.class_counter, key=self.class_counter.get)

            min_class_indices = []

            for i in range(n):
                if int(target[i]) == min_class:
                    min_class_indices.append(i)

            # add min class info to memory
            for idx in min_class_indices:
                if self.memory_info[q]["num_items"] < self.memory_limit:
                    self.memory_info[q]["context"].append(context[idx].detach())
                    self.memory_info[q]["target"].append(target[idx].detach())
                    self.memory_info[q]["num_items"] += 1
                    self.num_memory_items += 1
                else:
                    # randomly sample an index from the memory info of the query
                    print("memory full, replacing")
                    rand_idx = torch.randint(self.memory_info[q]["num_items"], size=()).to(torch.long)

                    self.memory_info[q]["context"][rand_idx] = context[idx].detach()
                    self.memory_info[q]["target"][rand_idx] = target[idx].detach()

        return mixup_context, mixup_target, target_i, target_j, mixup_query
    
    def reset_memory(self):
        self.memory_info = dict()
        self.num_memory_items = 0

        self.class_counter = dict(zip(range(self.num_classes), [1] * self.num_classes))

    def train(self):
        self.mode = "train"

    def eval(self):
        self.mode = "eval"
