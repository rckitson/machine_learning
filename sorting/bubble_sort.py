"""Bubble sort algorithm."""
from typing import List
import numpy as np

def bubble_sort(A: List[int]) -> List[int]:
    """Sort a list of integers using the bubble sort algorithm.

    Args:
        A: The list to be sorted

    Returns:
        The sorted list
    """
    not_sorted: bool = True
    passes: int = 0
    while not_sorted:
        # Assume it's sorted until it's not
        not_sorted = False
        passes += 1
        for ii in range(len(A) - 1):
            if A[ii] > A[ii + 1]:
                # Swap
                A[ii], A[ii + 1] = A[ii + 1], A[ii]
                not_sorted = True

    print("Bubble sort passes:", passes)
    return A
        
if __name__ == "__main__":
    a = np.arange(0, 20)[::-1].tolist()
    print(a)
    print(bubble_sort(a))
