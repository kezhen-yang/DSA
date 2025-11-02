# Time and Space Complexity Examples in Python
# O(1) -> O(log n) -> O(sqrt(n)) -> O(n) -> O(n log n) -> O(n^2) -> O(2^n) -> O(n!)
# O(1) - Constant Time Complexity
# O(n) - Linear Time Complexity
# O(n^2) - Quadratic Time Complexity
# O(log n) - Logarithmic Time Complexity
# O(n log n) - Linearithmic Time Complexity
# O(2^n) - Exponential Time Complexity
# O(n!) - Factorial Time Complexity
# O(sqrt(n)) - Square Root Time Complexity


########################################################################
# Example of O(1) - Constant Time Complexity

# Array 
nums = [1,2,3,4,5]
nums.append(6) # push to end of array
nums.pop() # pop from end of array
value = nums[0] # lookup by index
value = nums[1] # lookup by index
value = nums[2] # lookup by index

# HashMap / Set
hash_map = {}
hash_map['key1'] = 'value1' # insert key-value pair
print("key1" in hash_map) # lookup by key
print(hash_map['key1']) # lookup by key
hash_map.pop('key1')  # delete by key


########################################################################
# Example of O(n) - Linear Time Complexity
nums = [1,2,3,4,5]
sum(nums) # sum of all elements
for num in nums: # iterate through all elements
    print(num)

nums.insert(1, 100) # inset middle, O(n) because need to shift elements. 
# When inserting at the beginning, O(n) because need to shift all elements to the right. 
# O(n) is the worst case.
nums.remove(100) # remove middle, if remove from the end its O(1).
# if remove from the beginning its O(n) because need to shift all elements to the left. 
# # O(n) is the worst case.
print(100 in nums) # lookup by value, in worst case need to check all elements O(n)

import heapq
heapq.heapify(nums) # build heap 

# Sometimes event nested loops can be O(n)
# (e.g. monotonic stack, sliding window, two pointers)


########################################################################
# Example of O(n^2) - Quadratic Time Complexity

# Traverse a square grid 
nums = [[1,2,3],[4,5,6],[7,8,9]]
for i in range(len(nums)):
    for j in range(len(nums[i])):
        print(nums[i][j])

# Get every pair of elements in array 
nums = [1,2,3,4,5]
for i in range(len(nums)):
    for j in range(i + 1, len(nums)):
        print(nums[i], nums[j])
# this is acutally (n^2)/2 but we drop the constant factor so its O(n^2)

# Insertion Sort 
# (insert in the middle n times -> O(n) each time -> O(n^2))

# for non square grid its O(m*n)

# O(n^3) is possible with triple nested loops but rare in practice
# Get every unique triplet of elements in array
nums = [1,2,3]
for i in range(len(nums)):
    for j in range(i + 1, len(nums)):
        for k in range(j + 1, len(nums)):
            print(nums[i], nums[j], nums[k])


########################################################################
# Example of O(log n) - Logarithmic Time Complexity
# on an array, every iteration of the loop, we eliminate half of the remaining elements. 
# So the number of iterations is log(n) base 2.
# O(log n) is common with divide and conquer algorithms.
# Given an array of size n, how many times can we divide it by 2 until we reach 1? 
# Another way to ask this is how many times can you take 1 and multiply it by 2 until you reach n?
# 2 ^ X = n, how to solve for X? X = log(n) base 2.
# Idea is the same for binary search tree, at least if you BST is balanced. 
# As you seach, we either go to the left of right and eliminate half of the remaining elements. 
# Until we find the element we looking for. 
# log(n) grows very slowly, so even for large n, log(n) is still a small number. 
# The difference between log(n) and n is huge.

# Binary Search
nums = [1,2,3,4,5]
target = 6
l, r = 0, len(nums) - 1
while l <= r:
    mid = (l + r) // 2
    if target < nums[mid]:
        r = mid - 1
    elif target > nums[mid]:
        l = mid + 1
    else:
        print("Found")
        break

# Binary Search on BST
def search(root, target):
    if not root:
        return False
    if target < root.val:
        return search(root.left, target)
    elif target > root.val:
        return search(root.right, target)
    else:
        return True
    
# Heap Push and Pop
import heapq
minHeap = []
heapq.heappush(minHeap, 3)
heapq.heappop(minHeap)


########################################################################
# Example of O(n log n) - Linearithmic Time Complexity
# O(n log n) is only marginally less efficient than O(n). 
# O(n log n) is a lot more efficient than O(n^2).
# Most common algorithms with O(n log n) time complexity are sorting algorithms.

# HeapSort 
import heapq
nums = [1,2,3,4,5]
heapq.heapify(nums) # O(n)
while nums:
    heapq.heappop(nums) # O(log n) each pop, n pops -> O(n log n)
# its actually O(n + n log n) but we drop the lower order term so its O(n log n).

# Merge Sort
# (and most built-in sort functions)


########################################################################
# Example of O(2^n) - Exponential Time Complexity
# O(2^n) time complexity is common with recursive algorithms that solve problems by breaking
# them down into smaller subproblems, such as the Fibonacci sequence or the Tower of Hanoi

# Recursion, tree height n, two branches 
def recursion(i, nums):
    if i == len(nums):
        return 0
    branch1 = recursion(i + 1, nums)
    branch2 = recursion(i + 2, nums)

# O(C^n) is possible with C branches at each level of the tree
# C branches, where c is sometimes n. If c is 3, we have a decision tree with 3 branches at each level.
# and the height of the tree is n, so the time complexity is O(3^n).
def recursion(i, nums, c):
    if i == len(nums):
        return 0
    for j in range(i, i + c):
        branches = recursion(j + 1, nums)


########################################################################
# Example of O(sqrt(n)) - Square Root Time Complexity

# Get all factors of n
import math
n = 12 
factors = set()
for i in range(1, int(math.sqrt(n))):
    if n % i == 0:
        factors.add(i)
        factors.add(n // i)
print(factors)


########################################################################
# Example of O(n!) - Factorial Time Complexity
# O(n!) time complexity is common with algorithms that generate all possible permutations or combinations of a
# set of elements, such as the traveling salesman problem or the N-Queens problem.
# Very inefficient even for small n.

# Permutations
# Travelling Salesman Problem
