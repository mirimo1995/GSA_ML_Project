import numpy as np
class SumTree:
    
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )
        self.pointer = 0
        self.max_pointer=0
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]
    def min_priority(self):
        return(self.tree[self.capacity-1:self.max_pointer+self.capacity-1].min())
    def add(self, p, data):
        idx = self.pointer + self.capacity - 1

        self.data[self.pointer] = data
        self.update(idx, p)

        self.pointer += 1
        self.max_pointer+=1
        if self.pointer >= self.capacity:
            self.pointer = 0
            self.max_pointer=self.capacity-1
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])