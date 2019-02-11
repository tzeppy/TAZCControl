

import numpy as np
import logging
import random

log = logging.getLogger(__name__)


class ReplayMemory(object):
    def __init__(self, size):
        self.size = size
        self.storage = list()
        self.ptr = 0

    def push(self, item):
        if len(self.storage) < self.size:
            self.storage.append(item)
        else:
            self.storage[self.ptr] = item
            self.ptr += 1
            self.ptr = self.ptr % self.size
        return self

    def sample(self, num_items):
        batch = random.sample(self.storage, num_items)
        return batch

    def __len__(self):
        return len(self.storage)

