class Node:
    def __init__(self, value):
        self.left = None
        self.right = None
        self.value = value

def build(array):
    root = Node(array[0])
    
    for element in array[1:]:
        curr = root

        while True:
            if element > curr.value:
                next = curr.right
            else:
                next = curr.left

            if next is None:
                break
                
            curr = next

        if element > curr.value:
            curr.right = Node(element)
        else:
            curr.left = Node(element)

    return root

def pre(root):
    print(root.value)
    if root.left is not None: pre(root.left)
    if root.right is not None: pre(root.right)

def post(root):
    if root.left is not None: post(root.left)
    if root.right is not None: post(root.right)
    print(root.value)

def ino(root):
    if root.left is not None: ino(root.left)
    print(root.value)
    if root.right is not None: ino(root.right)

# Generate root
root = build([17, 8, 4, 5, 12, 14, 22, 19, 30, 25])
print('------')
pre(root)
print('------')
post(root)
print('------')
ino(root)