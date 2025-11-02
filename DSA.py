########################################################################
############################ Static Arrays #############################
########################################################################

# Insert n into arr at the next open position.
# Length is the number of 'real' values in arr, and capacity
# is the size (aka memory allocated for the fixed size array).
def insertEnd(arr, n, length, capacity):
    if length < capacity:
        arr[length] = n

# Remove from the last position in the array if the array
# is not empty (i.e. length is non-zero).
def removeEnd(arr, length):
    if length > 0:
        # Overwrite last element with some default value.
        # We would also consider the length to be decreased by 1.
        arr[length - 1] = 0

# Insert n into index i after shifting elements to the right.
# Assuming i is a valid index and arr is not full.
def insertMiddle(arr, i, n, length):
    # Shift starting from the end to i.
    for index in range(length - 1, i - 1, -1):
        arr[index + 1] = arr[index]
    
    # Insert at i
    arr[i] = n

# Remove value at index i before shifting elements to the left.
# Assuming i is a valid index.
def removeMiddle(arr, i, length):
    # Shift starting from i + 1 to end.
    for index in range(i + 1, length):
        arr[index - 1] = arr[index]
    # No need to 'remove' arr[i], since we already shifted

def printArr(arr, capacity):
    for i in range(capacity):
        print(arr[i])

########################################################################
############################ Dynamic Arrays ############################
########################################################################

# Python arrays are dynamic by default, but this is an example of resizing.
class Array:
    def __init__(self):
        self.capacity = 2
        self.length = 0
        self.arr = [0] * 2 # Array of capacity = 2

    # Insert n in the last position of the array
    def pushback(self, n):
        if self.length == self.capacity:
            self.resize()
            
        # insert at next empty position
        self.arr[self.length] = n
        self.length += 1

    def resize(self):
        # Create new array of double capacity
        self.capacity = 2 * self.capacity
        newArr = [0] * self.capacity 
        
        # Copy elements to newArr
        for i in range(self.length):
            newArr[i] = self.arr[i]
        self.arr = newArr
        
    # Remove the last element in the array
    def popback(self):
        if self.length > 0:
            self.length -= 1
    
    # Get value at i-th index
    def get(self, i):
        if i < self.length:
            return self.arr[i]
        # Here we would throw an out of bounds exception

    # Insert n at i-th index
    def insert(self, i, n):
        if i < self.length:
            self.arr[i] = n
            return
        # Here we would throw an out of bounds exception       

    def print(self):
        for i in range(self.length):
            print(self.arr[i])
        print()

########################################################################
############################ Stacks ####################################
########################################################################

# Implementing a stack is trivial using a dynamic array
# (which we implemented earlier).
class Stack:
    def __init__(self):
        self.stack = []

    def push(self, n):
        self.stack.append(n)

    def pop(self):
        return self.stack.pop()

########################################################################
############################ Singly Linked List ########################
########################################################################

class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

# Implementation for Singly Linked List
class LinkedList:
    def __init__(self):
        # Init the list with a 'dummy' node which makes 
        # removing a node from the beginning of list easier.
        self.head = ListNode(-1)
        self.tail = self.head
    
    def insertEnd(self, val):
        self.tail.next = ListNode(val)
        self.tail = self.tail.next

    def remove(self, index):
        i = 0
        curr = self.head
        while i < index and curr:
            i += 1
            curr = curr.next
        
        # Remove the node ahead of curr
        if curr and curr.next:
            if curr.next == self.tail:
                self.tail = curr
            curr.next = curr.next.next

    def print(self):
        curr = self.head.next
        while curr:
            print(curr.val, " -> ", end="")
            curr = curr.next
        print()

########################################################################
############################ Doubly Linked List ########################
########################################################################

class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None
        self.prev = None

# Implementation for Doubly Linked List
class LinkedList:
    def __init__(self):
        # Init the list with 'dummy' head and tail nodes which makes 
        # edge cases for insert & remove easier.
        self.head = ListNode(-1)
        self.tail = ListNode(-1)
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def insertFront(self, val):
        newNode = ListNode(val)
        newNode.prev = self.head
        newNode.next = self.head.next

        self.head.next.prev = newNode
        self.head.next = newNode

    def insertEnd(self, val):
        newNode = ListNode(val)
        newNode.next = self.tail
        newNode.prev = self.tail.prev

        self.tail.prev.next = newNode
        self.tail.prev = newNode

    # Remove first node after dummy head (assume it exists)
    def removeFront(self):
        self.head.next.next.prev = self.head
        self.head.next = self.head.next.next

    # Remove last node before dummy tail (assume it exists)
    def removeEnd(self):
        self.tail.prev.prev.next = self.tail
        self.tail.prev = self.tail.prev.prev

    def print(self):
        curr = self.head.next
        while curr != self.tail:
            print(curr.val, " -> ")
            curr = curr.next
        print()

########################################################################
############################ Queues ####################################
########################################################################

class ListNode:
    def __init__(self, val):
        self.val = val
        self.next = None

class Queue:
    # Implementing this with dummy nodes would be easier!
    def __init__(self):
        self.left = self.right = None
    
    def enqueue(self, val):
        newNode = ListNode(val)

        # Queue is non-empty
        if self.right:
            self.right.next = newNode
            self.right = self.right.next
        # Queue is empty
        else:
            self.left = self.right = newNode

    def dequeue(self):
        # Queue is empty
        if not self.left:
            return None
        
        # Remove left node and return value
        val = self.left.val
        self.left = self.left.next
        if not self.left:
            self.right = None
        return val

    def print(self):
        cur = self.left
        while cur:
            print(cur.val, ' -> ', end ="")
            cur = cur.next
        print() # new line

########################################################################
############################ Factorial #################################
########################################################################

# Recursive implementation of n! (n-factorial) calculation
def factorial(n):
    # Base case: n = 0 or 1
    if n <= 1:
        return 1

    # Recursive case: n! = n * (n - 1)!
    return n * factorial(n - 1)

########################################################################
############################ Fibonacci Sequence ########################
########################################################################

# Recursive implementation to calculate the n-th Fibonacci number
def fibonacci(n):
    # Base case: n = 0 or 1
    if n <= 1:
        return n

    # Recursive case: fib(n) = fib(n - 1) + fib(n - 2)
    return fibonacci(n - 1) + fibonacci(n - 2)

########################################################################
############################ Insertion Sort ############################
########################################################################

# Python implementation of Insertion Sort
def insertionSort(arr):
	# Traverse through 1 to len(arr)
    for i in range(1, len(arr)):
        j = i - 1
        while j >= 0 and arr[j + 1] < arr[j]:
            # arr[j] and arr[j + 1] are out of order so swap them 
            tmp = arr[j + 1]
            arr[j + 1] = arr[j]
            arr[j] = tmp
            j -= 1
    return arr

########################################################################
############################ Merge Sort ################################
########################################################################

# Implementation of MergeSort
def mergeSort(arr, s, e):
    if e - s + 1 <= 1:
        return arr

    # The middle index of the array
    m = (s + e) // 2

    # Sort the left half
    mergeSort(arr, s, m) 
    ## mergeSort function works by modifyign the input list arr in-place 
    ## -- it doesnot return a new sorted list.
    ## Instead, it changes the original list as it recurses and merges the sorted halves 

    # Sort the right half
    mergeSort(arr, m + 1, e)

    # Merge sorted halfs
    merge(arr, s, m, e)
    
    return arr

# Merge in-place
def merge(arr, s, m, e):
    # Copy the sorted left & right halfs to temp arrays
    L = arr[s: m + 1]
    R = arr[m + 1: e + 1]

    i = 0 # index for L
    j = 0 # index for R
    k = s # index for arr

    # Merge the two sorted halfs into the original array
    while i < len(L) and j < len(R):
        if L[i] <= R[j]:
            arr[k] = L[i]
            i += 1
        else:
            arr[k] = R[j]
            j += 1
        k += 1

    # One of the halfs will have elements remaining
    while i < len(L):
        arr[k] = L[i]
        i += 1
        k += 1
    while j < len(R):
        arr[k] = R[j]
        j += 1
        k += 1

########################################################################
############################ Quick Sort ################################
########################################################################

# Implementation of QuickSort
def quickSort(arr: list[int], s: int, e: int) -> list[int]:
    if e - s + 1 <= 1:
        return arr

    pivot = arr[e]
    left = s # pointer for left side

    # Partition: elements smaller than pivot on left side
    for i in range(s, e):
        if arr[i] < pivot:
            tmp = arr[left]
            arr[left] = arr[i]
            arr[i] = tmp
            left += 1

    # Move pivot in-between left & right sides
    arr[e] = arr[left]
    arr[left] = pivot
    
    # Quick sort left side
    quickSort(arr, s, left - 1)

    # Quick sort right side
    quickSort(arr, left + 1, e)

    return arr
    
########################################################################
############################ Bucket Sort ###############################
########################################################################

def bucketSort(arr):
    # Assuming arr only contains 0, 1 or 2
    counts = [0, 0, 0]

    # Count the quantity of each val in arr
    for n in arr:
        counts[n] += 1
    
    # Fill each bucket in the original array
    i = 0
    for n in range(len(counts)):
        for j in range(counts[n]):
            arr[i] = n
            i += 1
    return arr

########################################################################
############################ Search Array ##############################
########################################################################

arr = [1, 3, 3, 4, 5, 6, 7, 8]

# Python implementation of Binary Search
def binarySearch(arr, target):
    L, R = 0, len(arr) - 1

    while L <= R:
        mid = (L + R) // 2

        if target > arr[mid]:
            L = mid + 1
        elif target < arr[mid]:
            R = mid - 1
        else:
            return mid
    return -1

########################################################################
############################ Search Range ##############################
########################################################################

# low = 1, high = 100

# Binary search on some range of values
def binarySearch(low, high):

    while low <= high:
        mid = (low + high) // 2

        if isCorrect(mid) > 0:
            high = mid - 1
        elif isCorrect(mid) < 0:
            low = mid + 1
        else:
            return mid
    return -1

# Return 1 if n is too big, -1 if too small, 0 if correct
def isCorrect(n):
    if n > 10:
        return 1
    elif n < 10:
        return -1
    else:
        return 0

########################################################################
############################ Binary Tree ###############################
########################################################################

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

########################################################################
############################ Binary Search Tree ########################
########################################################################

# Motivation: 
# With BSTs we can search for values in O(log n) time just like with a sorted array. 
# BSTs are often preferred over sorted arrays because they allow for insertion and deletion in O(log n) time. 
# This is not possible with sorted arrays, which require O(n) time for these operations.

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def search(root, target):
    if not root:
        return False
    
    if target > root.val:
        return search(root.right, target)
    elif target < root.val:
        return search(root.left, target)
    else:
        return True

########################################################################
########################## BST Insert and Remove #######################
########################################################################

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

# Insert a new node and return the root of the BST.
def insert(root, val):
    if not root:
        return TreeNode(val)
    
    if val > root.val:
        root.right = insert(root.right, val)
    elif val < root.val:
        root.left = insert(root.left, val)
    return root

# Return the minimum value node of the BST.
def minValueNode(root):
    curr = root
    while curr and curr.left:
        curr = curr.left
    return curr

# Remove a node and return the root of the BST.
def remove(root, val):
    if not root:
        return None
    
    if val > root.val:
        root.right = remove(root.right, val)
    elif val < root.val:
        root.left = remove(root.left, val)
    else:
        if not root.left:
            return root.right
        elif not root.right:
            return root.left
        else:
            minNode = minValueNode(root.right)
            root.val = minNode.val
            root.right = remove(root.right, minNode.val)
    return root

########################################################################
########################## Depth-First Search ##########################
########################################################################

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def inorder(root):
    if not root:
        return    
    inorder(root.left)
    print(root.val)
    inorder(root.right)

def preorder(root):
    if not root:
        return    
    print(root.val)
    preorder(root.left)
    preorder(root.right)

def postorder(root):
    if not root:
        return    
    postorder(root.left)
    postorder(root.right)
    print(root.val)

########################################################################
########################## Breadth-First Search ########################
########################################################################

from collections import deque

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def bfs(root):
    queue = deque()

    if root:
        queue.append(root)
    
    level = 0
    while len(queue) > 0:
        print("level: ", level)
        for i in range(len(queue)):
            curr = queue.popleft()
            print(curr.val)
            if curr.left:
                queue.append(curr.left)
            if curr.right:
                queue.append(curr.right)
        level += 1

########################################################################
########################## BST Sets and Maps ###########################
########################################################################

# Binary Search Tree Node
class TreeNode:
    def __init__(self, key: int, val: int):
        self.key = key
        self.val = val
        self.left = None
        self.right = None

# Implementation for Binary Search Tree Map
class TreeMap:
    def __init__(self):
        self.root = None

    def insert(self, key: int, val: int) -> None:
        newNode = TreeNode(key, val)
        if self.root == None:
            self.root = newNode
            return

        current = self.root
        while True:
            if key < current.key:
                if current.left == None:
                    current.left = newNode
                    return
                current = current.left
            elif key > current.key:
                if current.right == None:
                    current.right = newNode
                    return
                current = current.right
            else:
                current.val = val
                return

    def get(self, key: int) -> int:
        current = self.root
        while current != None:
            if key < current.key:
                current = current.left
            elif key > current.key:
                current = current.right
            else:
                return current.val
        return -1

    def getMin(self) -> int:
        current = self.findMin(self.root)
        return current.val if current else -1

    # Returns the node with the minimum key in the subtree
    def findMin(self, node: TreeNode) -> TreeNode:
        while node and node.left:
            node = node.left
        return node

    def getMax(self) -> int:
        current = self.root
        while current and current.right:
            current = current.right
        return current.val if current else -1
    
    def remove(self, key: int) -> None:
        self.root = self.removeHelper(self.root, key)

    # Returns the new root of the subtree after removing the key
    def removeHelper(self, curr: TreeNode, key: int) -> TreeNode:
        if curr == None:
            return None

        if key > curr.key:
            curr.right = self.removeHelper(curr.right, key)
        elif key < curr.key:
            curr.left = self.removeHelper(curr.left, key)
        else:
            if curr.left == None:
                # Replace curr with right child
                return curr.right
            elif curr.right == None:
                # Replace curr with left child
                return curr.left
            else:
                # Swap curr with inorder successor
                minNode = self.findMin(curr.right)
                curr.key = minNode.key
                curr.val = minNode.val
                curr.right = self.removeHelper(curr.right, minNode.key)
        return curr

    def getInorderKeys(self) -> List[int]:
        result = []
        self.inorderTraversal(self.root, result)
        return result

    def inorderTraversal(self, root: TreeNode, result: List[int]) -> None:
        if root != None:
            self.inorderTraversal(root.left, result)
            result.append(root.key)
            self.inorderTraversal(root.right, result)

########################################################################
############################ Tree Maze #################################
########################################################################

class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None

def canReachLeaf(root):
    if not root or root.val == 0:
        return False
    
    if not root.left and not root.right:
        return True
    if canReachLeaf(root.left):
        return True
    if canReachLeaf(root.right):
        return True
    return False

def leafPath(root, path):
    if not root or root.val == 0:
        return False
    path.append(root.val)

    if not root.left and not root.right:
        return True
    if leafPath(root.left, path):
        return True
    if leafPath(root.right, path):
        return True
    path.pop()
    return False

########################################################################
############################ Heap Properties ###########################
########################################################################
"""
A heap is a specialized, tree-based data structure.

It implements an abstract data type called the Priority Queue, but sometimes 'Heap' and 'Priority Queue' are used interchangeably.

We already learned that queues operate with a first-in-first-out basis but with a priority queue, the values are removed based on a specific priority. The element with the highest priority is removed first, regardless of the order in which it was added.


Two types of heaps
1. Min Heap: Min heaps have the smallest value at the root node. The smallest value has the highest priority to be removed.
2. Max Heap: Max heaps have the largest value at the root node. The largest value has the highest priority to be removed.


Heap Properties
For a binary tree to qualify as a heap, it must satisfy the following properties:

1. Structure Property
A binary heap is a binary tree that is a complete binary tree, where every single level of the tree is filled completely, except possibly the lowest level nodes, which are filled contiguously from left to right.

2. Order Property
The order property for a min-heap is that all of the descendents should be greater than their ancestors. In other words, if we have a root with value y, every node in the right and the left sub-tree should have values greater than or equal to y. This is a recursive property, similar to binary search trees.

In a max-heap, every node in the right and the left sub-tree is smaller than or equal to y.

"""

# leftChild of i = heap[2 * i]
# rightChild of i = heap[(2 * i) + 1] 
# parent of i = heap[i // 2]

########################################################################
############################ Push and Pop ##############################
########################################################################

# Min Heap
class Heap:
    def __init__(self):
        self.heap = [0]

    def push(self, val):
        self.heap.append(val)
        i = len(self.heap) - 1

        # Percolate up
        while i > 1 and self.heap[i] < self.heap[i // 2]:
            tmp = self.heap[i]
            self.heap[i] = self.heap[i // 2]
            self.heap[i // 2] = tmp
            i = i // 2

    def pop(self):
        if len(self.heap) == 1:
            return None # If our heap is empty, there is nothing to pop, hence the return null.
        if len(self.heap) == 2:
            return self.heap.pop() # Our heap also could have just one node, in which case, we will just pop that node and don't need to make any adjustments.

        # If the above two statements have not executed, it must be the case that we have children, meaning we need to perform a swap.
        res = self.heap[1] # We store our root node (min value in a min Heap) into a variable called res so that we don't lose it. 
        # Move last value to root
        self.heap[1] = self.heap.pop() # Then we can replace last node to be at the root node.
        i = 1
        # Percolate down
        while 2 * i < len(self.heap): # Our while loop runs as long as we have a left child and we determine this by making sure 2 * i is not out of bounds. Then, there are three cases we concern ourselves with:
            if (2 * i + 1 < len(self.heap) and 
            self.heap[2 * i + 1] < self.heap[2 * i] and 
            self.heap[i] > self.heap[2 * i + 1]): # The node has two children. We also make sure that the current node is greater than its children because of the order property. We replace the node with the minimum of its two children. When considering a binary heap, it is not possible to have only a right child because then it no longer is a complete binary tree and violates the structure property.
                # Swap right child
                tmp = self.heap[i]
                self.heap[i] = self.heap[2 * i + 1]
                self.heap[2 * i + 1] = tmp
                i = 2 * i + 1
            elif self.heap[i] > self.heap[2 * i]: # The node only has a left child. If no right child exists and the current node's value is greater than its left child, we swap it with the left child.
                # Swap left child
                tmp = self.heap[i]
                self.heap[i] = self.heap[2 * i]
                self.heap[2 * i] = tmp
                i = 2 * i
            else: # If none of the above cases exectue, then it must be the case that our node is in the proper position already, satisfying both the order and the structural property.
                break
        return res

    def top(self):
        if len(self.heap) > 1:
            return self.heap[1]
        return None

########################################################################
############################ Heapify ###################################
########################################################################

# Min Heap
class Heap:
    # ... not showing push, pop to save space.
    
    def heapify(self, arr):
        # 0-th position is moved to the end
        arr.append(arr[0])

        self.heap = arr
        cur = (len(self.heap) - 1) // 2
        while cur > 0:
            # Percolate down
            i = cur
            while 2 * i < len(self.heap):
                if (2 * i + 1 < len(self.heap) and 
                self.heap[2 * i + 1] < self.heap[2 * i] and 
                self.heap[i] > self.heap[2 * i + 1]):
                    # Swap right child
                    tmp = self.heap[i]
                    self.heap[i] = self.heap[2 * i + 1]
                    self.heap[2 * i + 1] = tmp
                    i = 2 * i + 1
                elif self.heap[i] > self.heap[2 * i]:
                    # Swap left child
                    tmp = self.heap[i]
                    self.heap[i] = self.heap[2 * i]
                    self.heap[2 * i] = tmp
                    i = 2 * i
                else:
                    break
            cur -= 1

########################################################################
############################ Hash Usage ################################
########################################################################

names = ["alice", "brad", "collin", "brad", "dylan", "kim"]

countMap = {}
for name in names:
    # If countMap does not contain name
    if name not in countMap:
        countMap[name] = 1
    else:
        countMap[name] += 1

########################################################################
############################ Hash Implementation #######################
########################################################################

class Pair:
    def __init__(self, key, val):
        self.key = key
        self.val = val

class HashMap:
    def __init__(self):
        self.size = 0
        self.capacity = 2
        self.map = [None, None]
    
    def hash(self, key):
        index = 0
        for c in key:
            index += ord(c)
        return index % self.capacity

    def get(self, key):
        index = self.hash(key)
        
        while self.map[index] != None:
            if self.map[index].key == key:
                return self.map[index].val
            index += 1
            index = index % self.capacity
        return None

    def put(self, key, val):
        index = self.hash(key)

        while True:
            if self.map[index] == None:
                self.map[index] = Pair(key, val)
                self.size += 1
                if self.size >= self.capacity // 2:
                    self.rehash()
                return
            elif self.map[index].key == key:
                self.map[index].val = val
                return
            
            index += 1
            index = index % self.capacity
    
    def remove(self, key):
        if not self.get(key):
            return
        
        index = self.hash(key)
        while True:
            if self.map[index].key == key:
                # Removing an element using open-addressing actually causes a bug,
                # because we may create a hole in the list, and our get() may 
                # stop searching early when it reaches this hole.
                self.map[index] = None
                self.size -= 1
                return
            index += 1
            index = index % self.capacity

    def rehash(self):
        self.capacity = 2 * self.capacity
        newMap = []
        for i in range(self.capacity):
            newMap.append(None)

        oldMap = self.map
        self.map = newMap
        self.size = 0
        for pair in oldMap:
            if pair:
                self.put(pair.key, pair.val)
    
    def print(self):
        for pair in self.map:
            if pair:
                print(pair.key, pair.val)

########################################################################
############################ Intro to Graphs ###########################
########################################################################

# Matrix (2D Grid)
grid = [[0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]]

# Adjacency matrix
adjMatrix = [[0, 0, 0, 0],
             [1, 1, 0, 0],
             [0, 0, 0, 1],
             [0, 1, 0, 0]]

# GraphNode used for adjacency list
class GraphNode:
    def __init__(self, val):
        self.val = val
        self.neighbors = []

########################################################################
############################ Matrix DFS ################################
########################################################################

# Matrix (2D Grid)
grid = [[0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]]

# Count paths (backtracking)
def dfs(grid, r, c, visit):
    ROWS, COLS = len(grid), len(grid[0])
    if (min(r, c) < 0 or
        r == ROWS or c == COLS or
        (r, c) in visit or grid[r][c] == 1):
        return 0
    if r == ROWS - 1 and c == COLS - 1:
        return 1

    visit.add((r, c))

    count = 0
    count += dfs(grid, r + 1, c, visit)
    count += dfs(grid, r - 1, c, visit)
    count += dfs(grid, r, c + 1, visit)
    count += dfs(grid, r, c - 1, visit)

    visit.remove((r, c))
    return count

########################################################################
############################ Matrix BFS ################################
########################################################################

from collections import deque

# Matrix (2D Grid)
grid = [[0, 0, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0]]

# Shortest path from top left to bottom right
def bfs(grid):
    ROWS, COLS = len(grid), len(grid[0])
    visit = set()
    queue = deque()
    queue.append((0, 0))
    visit.add((0, 0))

    length = 0
    while queue:
        for i in range(len(queue)):
            r, c = queue.popleft()
            if r == ROWS - 1 and c == COLS - 1:
                return length

            neighbors = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            for dr, dc in neighbors:
                if (min(r + dr, c + dc) < 0 or
                    r + dr == ROWS or c + dc == COLS or
                    (r + dr, c + dc) in visit or grid[r + dr][c + dc] == 1):
                    continue
                queue.append((r + dr, c + dc))
                visit.add((r + dr, c + dc))
        length += 1

########################################################################
############################ Adjacency List ############################
########################################################################

from collections import deque

# GraphNode used for adjacency list
class GraphNode:
    def __init__(self, val):
        self.val = val
        self.neighbors = []

# Or use a HashMap
adjList = { "A": [], "B": [] }

# Given directed edges, build an adjacency list
edges = [["A", "B"], ["B", "C"], ["B", "E"], ["C", "E"], ["E", "D"]]

adjList = {}

for src, dst in edges:
    if src not in adjList:
        adjList[src] = []
    if dst not in adjList:
        adjList[dst] = []
    adjList[src].append(dst)


# Count paths (backtracking)
def dfs(node, target, adjList, visit):
    if node in visit:
        return 0
    if node == target:
        return 1
    
    count = 0
    visit.add(node)
    for neighbor in adjList[node]:
        count += dfs(neighbor, target, adjList, visit)
    visit.remove(node)

    return count

# Shortest path from node to target
def bfs(node, target, adjList):
    length = 0
    visit = set()
    visit.add(node)
    queue = deque()
    queue.append(node)

    while queue:
        for i in range(len(queue)):
            curr = queue.popleft()
            if curr == target:
                return length

            for neighbor in adjList[curr]:
                if neighbor not in visit:
                    visit.add(neighbor)
                    queue.append(neighbor)
        length += 1
    return length

########################################################################
############################ 1-Dimension DP ############################
########################################################################

# Brute Force # Fibonacci 
def bruteForce(n):
    if n <= 1:
        return n
    return bruteForce(n - 1) + bruteForce(n - 2)

# Memoization # Top down dynamic programming 
def memoization(n, cache):
    if n <= 1:
        return n
    if n in cache:
        return cache[n]

    cache[n] = memoization(n - 1, cache) + memoization(n - 2, cache)
    return cache[n]

# Dynamic Programming # Bottom up dynamic programming 
def dp(n):
    if n < 2:
        return n

    dp = [0, 1]
    i = 2
    while i <= n:
        tmp = dp[1]
        dp[1] = dp[0] + dp[1]
        dp[0] = tmp
        i += 1
    return dp[1]

########################################################################
############################ 2-Dimension DP ############################
########################################################################

# Brute Force - Time: O(2 ^ (n + m)), Space: O(n + m)
def bruteForce(r, c, rows, cols):
    if r == rows or c == cols:
        return 0
    if r == rows - 1 and c == cols - 1:
        return 1
    
    return (bruteForce(r + 1, c, rows, cols) +  
            bruteForce(r, c + 1, rows, cols))

print(bruteForce(0, 0, 4, 4))

# Memoization - Time and Space: O(n * m)
def memoization(r, c, rows, cols, cache):
    if r == rows or c == cols:
        return 0
    if cache[r][c] > 0:
        return cache[r][c]
    if r == rows - 1 and c == cols - 1:
        return 1
    
    cache[r][c] = (memoization(r + 1, c, rows, cols, cache) +  
        memoization(r, c + 1, rows, cols, cache))
    return cache[r][c]

print(memoization(0, 0, 4, 4, [[0] * 4 for i in range(4)]))

# Bottom up or Dynamic Programming - Time: O(n * m), Space: O(m), where m is num of cols
def dp(rows, cols):
    prevRow = [0] * cols

    for r in range(rows - 1, -1, -1):
        curRow = [0] * cols
        curRow[cols - 1] = 1
        for c in range(cols - 2, -1, -1):
            curRow[c] = curRow[c + 1] + prevRow[c]
        prevRow = curRow
    return prevRow[0] 

########################################################################
############################ Bit Operations ############################
########################################################################

# AND
n = 1 & 1

# OR
n = 1 | 0

# XOR
n = 0 ^ 1

# NOT (negation)
n = ~n

# Bit shifting
n = 1
n = n << 1
n = n >> 1

# Counting Bits
def countBits(n):
    count = 0
    while n > 0:
        if n & 1 == 1:
            count += 1
        n = n >> 1 # same as n // 2
    return count

########################################################################
############################ Kadane's Algorithm ########################
########################################################################
# Kadane's algorithm is a greedy/dynamic programming algorithm that can be used on an array. 
# It is used to calculate the maximum sum subarray ending at a particular position 
# and typically runs in O(n) time.

# Brute Force: O(n^2)
def bruteForce(nums):
    maxSum = nums[0]

    for i in range(len(nums)):
        curSum = 0
        for j in range(i, len(nums)):
            curSum += nums[j]
            maxSum = max(maxSum, curSum)
    return maxSum

# Kadane's Algorithm: O(n)
def kadanes(nums):
    maxSum = nums[0]
    curSum = 0

    for n in nums:
        curSum = max(curSum, 0)
        curSum += n
        maxSum = max(maxSum, curSum)
    return maxSum

# Return the left and right index of the max subarray sum,
# assuming there's exactly one result (no ties).
# Sliding window variation of Kadane's: O(n)
def slidingWindow(nums):
    maxSum = nums[0]
    curSum = 0
    maxL, maxR = 0, 0
    L = 0

    for R in range(len(nums)):
        if curSum < 0:
            curSum = 0
            L = R

        curSum += nums[R]
        if curSum > maxSum:
            maxSum = curSum
            maxL, maxR = L, R 

    return [maxL, maxR]

########################################################################
############################ Sliding Window Fixed Size #################
########################################################################
# Having a fixed sliding window is to maintain two pointers that are k apart 
# from each other and fit a certain constraint.

def closeDuplicatesBruteForce(nums, k):
    for L in range(len(nums)):
        for R in range(L + 1, min(len(nums), L + k + 1)):
            if nums[L] == nums[R]:
                return True
    return False

# Same problem using sliding window. 
# O(n)
def closeDuplicates(nums, k):
    window = set() # Cur window of size <= k 
    L = 0 

    for R in range(len(nums)):
        if R - L + 1 > k:
            window.remove(nums[L])
            L += 1
        if nums[R] in window:
            return True
        window.add(nums[R])

    return False

########################################################################
############################ Sliding Window Variable Size ##############
########################################################################
# Another variation of the sliding window technique is the variable size sliding window. 
# This is useful when we don't have a fixed window size and we need to keep expanding our 
# window as long as our window meets a certain constraint.

# Find the length of longest subarray with the same value in each position: O(n)
def longestSubarray(nums):
    length = 0 
    L = 0

    for R in range(len(nums)):
        if nums[L] != nums[R]:
            L = R
        length = max(length, R - L + 1)
    return length 

# Find length of the minimum size subarray where the sum is 
# greater than or equal to the target.
# Assume all values in the input are positive.
# O(n)
def shortestSubarray(nums, target):
    L, total = 0, 0
    length = float("inf")
    
    for R in range(len(nums)):
        total += nums[R]
        while total >= target:
            length = min(R - L + 1, length)
            total -= nums[L]
            L += 1
    return 0 if length == float("inf") else length

########################################################################
############################ Two Pointers ##############################
########################################################################

# Given a string of characters, return true if it's a palindrome,
# return false otherwise: O(n)
def isPalindrome(word):
    L, R = 0, len(word) - 1
    while L < R:
        if word[L] != word[R]:
            return False
        L += 1
        R -= 1
    return True

# Given a sorted array of integers, return the indices
# of two elements (in different positions) that sum up to
# the target value. Assume there is exactly one solution.
# O(n)
def targetSum(nums, target):
    L, R = 0, len(nums) - 1
    while L < R:
        if nums[L] + nums[R] > target:
            R -= 1
        elif nums[L] + nums[R] < target:
            L += 1
        else:
            return [L, R]

########################################################################
############################ Prefix Sums ###############################
########################################################################

class PrefixSum:

    def __init__(self, nums):
        self.prefix = []
        total = 0
        for n in nums:
            total += n
            self.prefix.append(total)
        
    def rangeSum(self, left, right):
        preRight = self.prefix[right]
        preLeft = self.prefix[left - 1] if left > 0 else 0
        return (preRight - preLeft)

########################################################################
######################### Fast and Slow Pointers #######################
########################################################################

# Find the middle of a linked list with two pointers.
# Time: O(n), Space: O(1)
def middleOfList(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

# Determine if the linked list contains a cycle.
# Time: O(n), Space: O(1)
def hasCycle(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# Determine if the linked list contains a cycle and
# return the beginning of the cycle, otherwise return null.
# Time: O(n), Space: O(1)
def cycleStart(head):
    slow, fast = head, head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break

    if not fast or not fast.next:
        return None
    
    slow2 = head
    while slow != slow2:
        slow = slow.next
        slow2 = slow2.next
    return slow

########################################################################
############################ Trie ######################################
########################################################################
# A trie (pronounced as "try") or prefix tree is a tree data structure 
# used to efficiently store and retrieve keys in a dataset of strings. 
# There are various applications of this data structure, 
# such as autocomplete and spellchecker.

class TrieNode:
    def __init__(self):
        self.children = {}
        self.word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        curr = self.root
        for c in word:
            if c not in curr.children:
                curr.children[c] = TrieNode()
            curr = curr.children[c]
        curr.word = True

    def search(self, word):
        curr = self.root
        for c in word:
            if c not in curr.children:
                return False
            curr = curr.children[c]
        return curr.word

    def startsWith(self, prefix):
        curr = self.root
        for c in prefix:
            if c not in curr.children:
                return False
            curr = curr.children[c]
        return True

########################################################################
############################ Union-Find ################################
########################################################################

"""
Union Find (Disjoint Sets)
Union-Find is a useful tool for keeping track of nodes connected in a graph and detecting cycles in a graph. 
Of course, we can achieve this with DFS as well by using a hashset, however this is only efficient when there is a static graph. 
If we are adding edges overtime, that makes the graph dynamic, and Union-Find is a better choice.

Disjoint sets
Union-Find operates on disjoint sets. Let's briefly go over them.

Disjoint sets are sets that don't have any element(s) in common. Formally, two disjoint sets are sets whose intersection is the empty set. 
Suppose we have two sets, S1 = {1,2,3} and S2 = {4,5,6}. S1 and S2 are referred to as disjoint sets, while two sets, S3 = {1,2,5}, and S4 = {5,6,7} are not disjoint.

Union-Find operates on disjoint sets. If we want to perform a union of two vertices, we need to ensure that those vertices belong to disjoint sets.
"""

class UnionFind:
    def __init__(self, n):
        self.par = {}
        self.rank = {}

        for i in range(1, n + 1):
            self.par[i] = i
            self.rank[i] = 0 # Every node starts with rank 0 (a tree of height 0 - just a single node).
            # rank helps keep the tree structure balanced and shallow.

    # Find parent of n, with path compression.
    def find(self, n):
        p = self.par[n]
        while p != self.par[p]:
            self.par[p] = self.par[self.par[p]]
            p = self.par[p]
        return p

    # Union by height / rank.
    # Return false if already connected, true otherwise.
    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if p1 == p2:
            return False
        
        if self.rank[p1] > self.rank[p2]:
            self.par[p2] = p1 # Attach shorter tree under taller tree
        elif self.rank[p1] < self.rank[p2]:
            self.par[p1] = p2 # Attach shorter tree under taller tree
        else:
            self.par[p1] = p2 # Same rank: attach either way
            self.rank[p2] += 1 # Height increases by 1
        return True

########################################################################
############################ Segment Tree ##############################
########################################################################
# A Segment Tree is a binary tree where each node represents a range 
# and stores the sum of elements in that range. This allows us to: 
# Update a single element in O(log N) time 
# Query a range sum in O(log N) time

class SegmentTree:
    def __init__(self, total, L, R):
        self.sum = total
        self.left = None
        self.right = None
        self.L = L
        self.R = R
        
    # O(n)
    @staticmethod
    def build(nums, L, R):
        if L == R:
            return SegmentTree(nums[L], L, R)

        M = (L + R) // 2
        root = SegmentTree(0, L, R)
        root.left = SegmentTree.build(nums, L, M)
        root.right = SegmentTree.build(nums, M + 1, R)
        root.sum = root.left.sum + root.right.sum
        return root

    # O(logn)
    def update(self, index, val):
        if self.L == self.R:
            self.sum = val
            return
        
        M = (self.L + self.R) // 2
        if index > M:
            self.right.update(index, val)
        else:
            self.left.update(index, val)
        self.sum = self.left.sum + self.right.sum
        
    # O(logn)
    def rangeQuery(self, L, R):
        if L == self.L and R == self.R:
            return self.sum
        
        M = (self.L + self.R) // 2
        if L > M:
            return self.right.rangeQuery(L, R)
        elif R <= M:
            return self.left.rangeQuery(L, R)
        else:
            return (self.left.rangeQuery(L, M) +
                    self.right.rangeQuery(M + 1, R))


########################################################################
############################ Iterative DFS #############################
########################################################################

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val, left, right):
        self.val = val
        self.left = left
        self.right = right

# Time and space: O(n)
def inorder(root):
    stack = []
    curr = root

    while curr or stack:
        if curr:
            stack.append(curr)
            curr = curr.left
        else:
            curr = stack.pop()
            print(curr.val)
            curr = curr.right

# Time and space: O(n)
def preorder(root):
    stack = []
    curr = root
    while curr or stack:
        if curr:
            print(curr.val)
            if curr.right:
                stack.append(curr.right)
            curr = curr.left
        else:
            curr = stack.pop()

# Time and space: O(n)
def postorder(root):
    stack = [root]
    visit = [False]
    while stack:
        curr, visited = stack.pop(), visit.pop()
        if curr:
            if visited:
                print(curr.val)
            else:
                stack.append(curr)
                visit.append(True)
                stack.append(curr.right)
                visit.append(False)
                stack.append(curr.left)
                visit.append(False)


########################################################################
############################ Two Heaps #################################
########################################################################

"""
Two Heaps Approach
We will maintain a max-heap and a min-heap.

1. In the max-heap, we store the smaller half of values from the stream.
2. Conversely, in the min-heap, we store the larger half of values from the stream.

To ensure each value goes in the correct heap and both heaps are roughly the same size, we might need to shift some elements around.

1. The max-heap will store the smaller half of the values. Thus, the top of the max-heap will be the largest value in the smaller half.
2. The min-heap will store the larger half of the values. Thus, the top of the min-heap will be the smallest value in the larger half.

This would mean that the largest value in our max-heap will be smaller than the smallest value in our min-heap. Thus, the median can be calculated by just retrieving the min and the max values from our respective heaps.

If the number of elements in our array is even, both of our heaps will have an equal number of elements. If we have an odd number of elements, one heap's size will be greater than the other. At any given iteration, the sizes of both heaps should only differ by at most 1 element.
"""

import heapq

class Median:
    def __init__(self):
        self.small, self.large = [], []

    def insert(self, num):
        # Push to the max heap and swap with min heap if needed.
        heapq.heappush(self.small, -1 * num) # python only has min heap, so we invert the values to simulate a max heap.
        if (self.small and self.large and (-1 * self.small[0]) > self.large[0]): # make sure every num in small is <= every num in large
            val = -1 * heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        
        # Handle uneven size
        if len(self.small) > len(self.large) + 1:
            val = -1 * heapq.heappop(self.small)
            heapq.heappush(self.large, val)
        if len(self.large) > len(self.small) + 1:
            val = heapq.heappop(self.large)
            heapq.heappush(self.small, -1 * val)
        
    def getMedian(self):
        if len(self.small) > len(self.large):
            return -1 * self.small[0]
        elif len(self.large) > len(self.small):
            return self.large[0]
        
        # Even # of elements, return avg of two middle nums
        return (-1 * self.small[0] + self.large[0]) / 2


########################################################################
############################ Subsets ###################################
########################################################################

"""
Formally, in mathematics, if we have two sets, Set A and Set B, Set A is a subset to Set B if all of its elements are found in Set B.

Additionally, any set is a subset of itself and an empty set is a subset of every set. Order is not important in subsets, but the elements within a set must be distinct.

For example, suppose we are given Set A = {1,2,3,4,5} and Set B = {5,2,1}. Set B is a subset of Set A because it only contain elements that can be found in Set A.

Changing the order of Set B to {2,5,1} does not change the set itself, so it is still a subset of Set A.

Given Set C = {9,10,11,12} and Set D = {6, 9, 10}, Set D is not a subset of Set C because it contains a 6, which Set C does not have.

"""

# Time: O(n * 2^n), Space: O(n)
def subsetsWithoutDuplicates(nums):
    subsets, curSet = [], []
    helper(0, nums, curSet, subsets)
    return subsets

def helper(i, nums, curSet, subsets):
    if i >= len(nums):
        subsets.append(curSet.copy())
        return
    
    # decision to include nums[i]
    curSet.append(nums[i])
    helper(i + 1, nums, curSet, subsets)
    curSet.pop()

    # decision NOT to include nums[i]
    helper(i + 1, nums, curSet, subsets)


# Time: O(n * 2^n), Space: O(n)
def subsetsWithDuplicates(nums):
    nums.sort()
    subsets, curSet = [], []
    helper2(0, nums, curSet, subsets)
    return subsets

def helper2(i, nums, curSet, subsets):
    if i >= len(nums):
        subsets.append(curSet.copy())
        return
    
    # decision to include nums[i]
    curSet.append(nums[i])
    helper2(i + 1, nums, curSet, subsets)
    curSet.pop()

    # decision NOT to include nums[i]
    while i + 1 < len(nums) and nums[i] == nums[i + 1]:
        i += 1
    helper2(i + 1, nums, curSet, subsets)


########################################################################
############################ Combinations ##############################
########################################################################

"""
Combinations are very similar to subsets in that we are choosing elements from a set. The order the elements are placed in also does not matter.

Generally, the main difference is that a combination is a subset of a specific size k, where k is a number that is less than or equal to the size of the original set.

"""

# Given n numbers (1 - n), return all possible combinations
# of size k. (n choose k math problem).
# Time: O(k * 2^n)
def combinations(n, k):
    combs = []
    helper(1, [], combs, n, k)
    return combs

def helper(i, curComb, combs, n, k):
    if len(curComb) == k:
        combs.append(curComb.copy())
        return
    if i > n:
        return
    
    # decision to include i
    curComb.append(i)
    helper(i + 1, curComb, combs, n, k)
    curComb.pop()
    
    # decision to NOT include i
    helper(i + 1, curComb, combs, n, k)


# Time: O(k * C(n, k))
def combinations2(n, k):
    combs = []
    helper2(1, [], combs, n, k)
    return combs

def helper2(i, curComb, combs, n, k):
    if len(curComb) == k:
        combs.append(curComb.copy())
        return
    if i > n:
        return
    
    for j in range(i, n + 1):
        curComb.append(j)
        helper2(j + 1, curComb, combs, n, k)
        curComb.pop()


########################################################################
############################ Permutations ##############################
########################################################################

"""
We can take a permutation of a set of elements by creating an ordered arrangement of those elements.

For example, a permutation from the set {1,2,3} could be {1,2,3} or {1,3,2} or {2,1,3}.

Notice that the order of the elements is important in permutations. Unlike with combinations, we want to use all of the elements from the given set.
"""

# Time: O(n^2 * n!)
def permutationsRecursive(nums):
    return helper(0, nums)
        
def helper(i, nums):   
    if i == len(nums):
        return [[]]
    
    resPerms = []
    perms = helper(i + 1, nums)
    for p in perms:
        for j in range(len(p) + 1):
            pCopy = p.copy()
            pCopy.insert(j, nums[i])
            resPerms.append(pCopy)
    return resPerms


# Time: O(n^2 * n!)
def permutationsIterative(nums):
    perms = [[]]

    for n in nums:
        nextPerms = []
        for p in perms:
            for i in range(len(p) + 1):
                pCopy = p.copy()
                pCopy.insert(i, n)
                nextPerms.append(pCopy)
        perms = nextPerms
    return perms


########################################################################
############################ Dijkstra's ################################
########################################################################

"""
Having already covered BFS, we will now cover another shortest path algorithm - Dijkstra's algorithm. The downside of using BFS is that it only works when the graph is unweighted - i.e. where the default weight of each edge is 1.

If we are given the following unweighted graph: edges = [[A,B], [A,C], [B,D], [C,B], [C,D], [C,E], [D,E]], and are asked to find the shortest path from A (source node) to D (destination node), we can apply classic BFS and keep adding the weights until we get to D.

Because all the weights are the same, BFS gives the shortest path in terms of the amount of vertices we visit. However, this fails when we have different weights for different edges as shown below.

"""

import heapq

# Given a connected graph represented by a list of edges, where
# edge[0] = src, edge[1] = dst, and edge[2] = weight,
# find the shortest path from src to every other node in the 
# graph. There are n nodes in the graph.
# O(E * logV), O(E * logE) is also correct.
def shortestPath(edges, n, src):
    adj = {}
    for i in range(1, n + 1):
        adj[i] = []
        
    # s = src, d = dst, w = weight
    for s, d, w in edges:
        adj[s].append([d, w])

    shortest = {}
    minHeap = [[0, src]]
    while minHeap:
        w1, n1 = heapq.heappop(minHeap)
        if n1 in shortest:
            continue
        shortest[n1] = w1

        for n2, w2 in adj[n1]:
            if n2 not in shortest:
                heapq.heappush(minHeap, [w1 + w2, n2])
    return shortest


########################################################################
############################ Prim's ####################################
########################################################################

"""
Prim's algorithm is used to find a spanning tree with the minimum cost. Similar to Dijkstra, Prim's is also a greedy algorithm and works on weighted undirected graphs.

But, what is a spanning tree? If we are given a graph G, a spanning tree is a subset of edges from G where the total weight of the edges is minimized. So, it is similar to Dijkstra in the sense of minimizing the cost.

Recall that by definition a tree is a connected graph, the same is true for a minimum spanning tree. However, a tree cannot have any cycles. This means that if G has n nodes, our minimum spanning tree will have at most n−1 edges.

The idea: 
G could have multiple valid minimum spanning trees with the same cost. So, given G, how do we go about finding the minimum spanning tree? 
We will start with an empty spanning tree and and at each vertex, we pick the connected vertex with the smallest weight, among all of its neighboring vertices. 
When we reach n−1 edges, we can stop our algorithm. In the visual below, for the MST, we start at vertex A, where the minimum cost of the connected component is 1 A->B, after which we have 2 B->C, which gives us n−1 edges and a minimum cost of 3. 
The spanning tree at the bottom of the visual is another valid spanning tree but it is not a minimum cost spanning tree.

"""

import heapq

# Given a list of edges of a connected undirected graph,
# with nodes numbered from 1 to n,
# return a list edges making up the minimum spanning tree.
def minimumSpanningTree(edges, n):
    adj = {}
    for i in range(1, n + 1):
        adj[i] = []
    for n1, n2, weight in edges:
        adj[n1].append([n2, weight])
        adj[n2].append([n1, weight])

    # Initialize the heap by choosing a single node
    # (in this case 1) and pushing all its neighbors.
    minHeap = []
    for neighbor, weight in adj[1]:
        heapq.heappush(minHeap, [weight, 1, neighbor])

    mst = []
    visit = set()
    visit.add(1)
    while len(visit) < n:
        weight, n1, n2 = heapq.heappop(minHeap)
        if n2 in visit:
            continue

        mst.append([n1, n2])
        visit.add(n2)
        for neighbor, weight in adj[n2]:
            if neighbor not in visit:
                heapq.heappush(minHeap, [weight, n2, neighbor])
    return mst


########################################################################
############################ Kruskal's #################################
########################################################################

"""
Kruskal's Minimum Spanning Tree Algorithm
Kruskal's algorithm, similar to Prim's algorithm, is another algorithm for finding the minimum spanning tree in an undirected weighted graph. 
Kruskal's is also a greedy algorithm that works better on sparse graphs. while Prim's works better on denser graphs.

The idea
Kruskal's algorithm works by sorting the edges in increasing (or rather non-decreasing) order of weights; 
then, starting with an initially empty tree, considers all edges, and adds the edge with the minimum weight to our MST, if it does not result in a cycle. 
Kruskal's will discard an edge if it results in a cycle.

Cycle Detection
Kruskal's makes use of the union-find data structure to detect if adding an edge would result in a cycle. 
Recall that union-find data structure operates on disjoint sets. 
It does so by keeping track of the parent of each vertex and if the two vertices we are trying to union have the same parent, 
it returns false since there is a cycle.

Of course, at any point there will be multiple valid spanning trees, which can occur if multiple edges have the same weight. 
It does not really matter which one we pick as all of them will lead to the same cost. Although, the resultant tree might look different.
"""

import heapq 

class UnionFind:
    def __init__(self, n):
        self.par = {}
        self.rank = {}

        for i in range(1, n + 1):
            self.par[i] = i
            self.rank[i] = 0
    
    # Find parent of n, with path compression.
    def find(self, n):
        p = self.par[n]
        while p != self.par[p]:
            self.par[p] = self.par[self.par[p]]
            p = self.par[p]
        return p

    # Union by height / rank.
    # Return false if already connected, true otherwise.
    def union(self, n1, n2):
        p1, p2 = self.find(n1), self.find(n2)
        if p1 == p2:
            return False
        
        if self.rank[p1] > self.rank[p2]:
            self.par[p2] = p1
        elif self.rank[p1] < self.rank[p2]:
            self.par[p1] = p2
        else:
            self.par[p1] = p2
            self.rank[p2] += 1
        return True

# Given an list of edges of a connected undirected graph,
# with nodes numbered from 1 to n,
# return a list edges making up the minimum spanning tree.
def minimumSpanningTree(edges, n):
    minHeap = []
    for n1, n2, weight in edges:
        heapq.heappush(minHeap, [weight, n1, n2])

    unionFind = UnionFind(n)
    mst = []
    while len(mst) < n - 1:
        weight, n1, n2 = heapq.heappop(minHeap)
        if not unionFind.union(n1, n2):
            continue
        mst.append([n1, n2])
    return mst


########################################################################
############################ Topological Sort ##########################
########################################################################

"""
The idea
Topological sort is a way of sorting a directed acyclic graph (DAG) such that each node comes before its dependent nodes. 
A simple example of this is university courses. 
There are some courses that can be taken without any pre-requisites and then there are those that have pre-requisites, i.e. you cannot take them unless you have taken other courses first.

In other words, some courses can be taken independent of other courses and others have to be taken in a specific order. 
We can represent this scenario using a DAG, where the edges represent the dependencies between the courses.

So, if we have node C and it has node A and B as its dependents, A and B will appear before C in the topological ordering. 
What order they appear in is not important unless A and B also have a dependency on each other.
"""

# Given a directed acyclical graph, return a valid
# topological ordering of the graph. 
def topologicalSort(edges, n):
    adj = {}
    for i in range(1, n + 1):
        adj[i] = []
    for src, dst in edges:
        adj[src].append(dst)

    topSort = []
    visit = set()
    for i in range(1, n + 1):
        dfs(i, adj, visit, topSort)
    topSort.reverse()
    return topSort

def dfs(src, adj, visit, topSort):
    if src in visit:
        return
    visit.add(src)
    
    for neighbor in adj[src]:
        dfs(neighbor, adj, visit, topSort)
    topSort.append(src)


########################################################################
############################ 0 / 1 Knapsack ############################
########################################################################

"""The idea
Suppose we are given a bag/knapsack with a fixed capacity, along with some items' weights and profits we reap by choosing to put that item in the bag. 
We want to maximize the profit while also ensuring that our backpack doesn't run out of space. 
The reason this algorithm is called 0/1 is because at each point, we can either choose to include an item or not include it - a binary decision.
"""

# Given a list of N items, and a backpack with a
# limited capacity, return the maximum total profit that 
# can be contained in the backpack. The i-th item's profit
# is profit[i] and it's weight is weight[i]. Assume you can
# only add each item to the bag at most one time. 

# Brute force Solution
# Time: O(2^n), Space: O(n)
# Where n is the number of items.
def dfs(profit, weight, capacity):
    return dfsHelper(0, profit, weight, capacity)

def dfsHelper(i, profit, weight, capacity):
    if i == len(profit):
        return 0

    # Skip item i
    maxProfit = dfsHelper(i + 1, profit, weight, capacity)

    # Include item i
    newCap = capacity - weight[i]
    if newCap >= 0:
        p = profit[i] + dfsHelper(i + 1, profit, weight, newCap)
        # Compute the max
        maxProfit = max(maxProfit, p)

    return maxProfit


# Memoization Solution
# Time: O(n * m), Space: O(n * m)
# Where n is the number of items & m is the capacity.
def memoization(profit, weight, capacity):
    # A 2d array, with N rows and M + 1 columns, init with -1's
    N, M = len(profit), capacity
    cache = [[-1] * (M + 1) for _ in range(N)]
    return memoHelper(0, profit, weight, capacity, cache)

def memoHelper(i, profit, weight, capacity, cache):
    if i == len(profit):
        return 0
    if cache[i][capacity] != -1:
        return cache[i][capacity]

    # Skip item i
    cache[i][capacity] = memoHelper(i + 1, profit, weight, capacity, cache)
    
    # Include item i
    newCap = capacity - weight[i]
    if newCap >= 0:
        p = profit[i] + memoHelper(i + 1, profit, weight, newCap, cache)
        # Compute the max
        cache[i][capacity] = max(cache[i][capacity], p)

    return cache[i][capacity]


# Dynamic Programming Solution
# Time: O(n * m), Space: O(n * m)
# Where n is the number of items & m is the capacity.
def dp(profit, weight, capacity):
    N, M = len(profit), capacity
    dp = [[0] * (M + 1) for _ in range(N)]

    # Fill the first column and row to reduce edge cases
    for i in range(N):
        dp[i][0] = 0
    for c in range(M + 1):
        if weight[0] <= c:
            dp[0][c] = profit[0] 

    for i in range(1, N):
        for c in range(1, M + 1):
            skip = dp[i-1][c]
            include = 0
            if c - weight[i] >= 0:
                include = profit[i] + dp[i-1][c - weight[i]]
            dp[i][c] = max(include, skip)
    return dp[N-1][M]


# Memory optimized Dynamic Programming Solution
# Time: O(n * m), Space: O(m)
def optimizedDp(profit, weight, capacity):
    N, M = len(profit), capacity
    dp = [0] * (M + 1)

    # Fill the first row to reduce edge cases
    for c in range(M + 1):
        if weight[0] <= c:
            dp[c] = profit[0] 

    for i in range(1, N):
        curRow = [0] * (M + 1)
        for c in range(1, M + 1):
            skip = dp[c]
            include = 0
            if c - weight[i] >= 0:
                include = profit[i] + dp[c - weight[i]]
            curRow[c] = max(include, skip)
        dp = curRow
    return dp[M]


########################################################################
############################ Unbounded Knapsack ########################
########################################################################

# Given a list of N items, and a backpack with a
# limited capacity, return the maximum total profit that 
# can be contained in the backpack. The i-th item's profit
# is profit[i] and it's weight is weight[i]. Assume you can
# have an unlimited number of each item available. 

# Brute force Solution
# Time: O(2^m), Space: O(m)
# Where m is the capacity.
def dfs(profit, weight, capacity):
    return dfsHelper(0, profit, weight, capacity)

def dfsHelper(i, profit, weight, capacity):
    if i == len(profit):
        return 0

    # Skip item i
    maxProfit = dfsHelper(i + 1, profit, weight, capacity)

    # Include item i
    newCap = capacity - weight[i]
    if newCap >= 0:
        p = profit[i] + dfsHelper(i, profit, weight, newCap)
        # Compute the max
        maxProfit = max(maxProfit, p)

    return maxProfit


# Memoization Solution
# Time: O(n * m), Space: O(n * m)
# Where n is the number of items & m is the capacity.
def memoization(profit, weight, capacity):
    # A 2d array, with N rows and M + 1 columns, init with -1's
    N, M = len(profit), capacity
    cache = [[-1] * (M + 1) for _ in range(N)]
    return memoHelper(0, profit, weight, capacity, cache)

def memoHelper(i, profit, weight, capacity, cache):
    if i == len(profit):
        return 0
    if cache[i][capacity] != -1:
        return cache[i][capacity]

    # Skip item i
    cache[i][capacity] = memoHelper(i + 1, profit, weight, capacity, cache)
    
    # Include item i
    newCap = capacity - weight[i]
    if newCap >= 0:
        p = profit[i] + memoHelper(i, profit, weight, newCap, cache)
        # Compute the max
        cache[i][capacity] = max(cache[i][capacity], p)

    return cache[i][capacity]


# Dynamic Programming Solution
# Time: O(n * m), Space: O(n * m)
# Where n is the number of items & m is the capacity.
def dp(profit, weight, capacity):
    N, M = len(profit), capacity
    dp = [[0] * (M + 1) for _ in range(N)]

    # Fill the first column and row to reduce edge cases
    for i in range(N):
        dp[i][0] = 0
    for c in range(M + 1):
        if weight[0] <= c:
            dp[0][c] = (c // weight[0]) * profit[0] 

    for i in range(1, N):
        for c in range(1, M + 1):
            skip = dp[i-1][c]
            include = 0
            if c - weight[i] >= 0:
                include = profit[i] + dp[i][c - weight[i]]
            dp[i][c] = max(include, skip)
    return dp[N-1][M]


# Memory optimized Dynamic Programming Solution
# Time: O(n * m), Space: O(m)
def optimizedDp(profit, weight, capacity):
    N, M = len(profit), capacity
    dp = [0] * (M + 1)

    for i in range(N):
        curRow = [0] * (M + 1)
        for c in range(1, M + 1):
            skip = dp[c]
            include = 0
            if c - weight[i] >= 0:
                include = profit[i] + curRow[c - weight[i]]
            curRow[c] = max(include, skip)
        dp = curRow
    return dp[M]


########################################################################
############################ LCS #######################################
########################################################################

# Time: O(2^(n + m)), Space: O(n + m)
def dfs(s1, s2):
    return dfsHelper(s1, s2, 0, 0)

def dfsHelper(s1, s2, i1, i2):
    if i1 == len(s1) or i2 == len(s2):
        return 0
    
    if s1[i1] == s2[i2]:
        return 1 + dfsHelper(s1, s2, i1 + 1, i2 + 1)
    else:
        return max(dfsHelper(s1, s2, i1 + 1, i2),
                dfsHelper(s1, s2, i1, i2 + 1))


# Time: O(n * m), Space: O(n + m)
def memoization(s1, s2):
    N, M = len(s1), len(s2)
    cache = [[-1] * M for _ in range(N)]
    return memoHelper(s1, s2, 0, 0, cache)

def memoHelper(s1, s2, i1, i2, cache):
    if i1 == len(s1) or i2 == len(s2):
        return 0
    if cache[i1][i2] != -1:
        return cache[i1][i2]

    if s1[i1] == s2[i2]:
        cache[i1][i2] = 1 + memoHelper(s1, s2, i1 + 1, i2 + 1, cache)
    else:
        cache[i1][i2] = max(memoHelper(s1, s2, i1 + 1, i2, cache),
                memoHelper(s1, s2, i1, i2 + 1, cache))
    return cache[i1][i2]


# Time: O(n * m), Space: O(n + m)
def dp(s1, s2):
    N, M = len(s1), len(s2)
    dp = [[0] * (M+1) for _ in range(N+1)]

    for i in range(N):
        for j in range(M):
            if s1[i] == s2[j]:
                dp[i+1][j+1] = 1 + dp[i][j]
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[N][M]


# Time: O(n * m), Space: O(m)
def optimizedDp(s1, s2):
    N, M = len(s1), len(s2)
    dp = [0] * (M + 1)

    for i in range(N):
        curRow = [0] * (M + 1)
        for j in range(M):
            if s1[i] == s2[j]:
                curRow[j+1] = 1 + dp[j]
            else:
                curRow[j+1] = max(dp[j + 1], curRow[j])
        dp = curRow
    return dp[M]


########################################################################
############################ Palindromes ###############################
########################################################################

"""
Palindromes
Surprisingly, palindrome problems can also be solved with dynamic programming. Although the pattern by which they are solved is not the typical DFS/memoization/bottom-up approach. 

Suppose we are faced with the following question:

Q: Given a string S, return the length of the longest palindromic substring within S.
A palindrome is a sequence of characters that read the same backwards as forwards. Think "racecar", or "aba". So, by definition, a palindromic substring is a contiguous part of a string that is a palindrome.

We discussed how to determine whether a string is a palindrome in the two pointers lesson. However, here we are faced with a variation, i.e. finding the length of the longest palindromic substring.
"""

# Time: O(n^2), Space: O(n)
def longest(s):
    length = 0
    
    for i in range(len(s)):
        # odd length
        l, r = i, i
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if (r - l + 1) > length:
                length = r - l + 1
            l -= 1
            r += 1
        
        # even length
        l, r = i, i + 1
        while l >= 0 and r < len(s) and s[l] == s[r]:
            if (r - l + 1) > length:
                length = r - l + 1
            l -= 1
            r += 1
            
    return length


# Same solution, without duplicate code.
# Time: O(n^2), Space: O(n)
def longest(s):
    length = 0
    for i in range(len(s)):
        # odd length
        length = max(length, helper(s, i, i))
        
        # even length
        length = max(length, helper(s, i, i + 1)) 
    return length

def helper(s, l, r):
    maxLength = 0
    while l >= 0 and r < len(s) and s[l] == s[r]:
        if (r - l + 1) > maxLength:
            maxLength = r - l + 1
        l -= 1
        r += 1
    return maxLength