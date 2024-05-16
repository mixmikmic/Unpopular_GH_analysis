# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

def oddEvenList(self, head):
    """
    :type head: ListNode
    :rtype: ListNode
    """
    odd_head = None
    odd_tail = None
    even_head = None
    even_tail = None
    while not (head==None):
        if head.val % 2 == 0:
            if even_tail == None:
                even_tail = head
                even_head = head
            else:
                even_tail.next = head
                even_tail = head
        else:
            if odd_tail == None:
                odd_tail = head
                odd_head = head
            else:
                odd_tail.next = head
                odd_tail = head

        head = head.next

    odd_tail.next = even_head
    return odd_head



