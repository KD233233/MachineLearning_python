# coding:gbk
import random
# class Node(object):
#     def __init__(self, sName):
#         self._lChildren = []
#         self.sName = sName
#
#     def __repr__(self):
#         return "<Node '{}'>".format(self.sName)
#
#     def append(self, *args, **kwargs):
#         self._lChildren.append(*args, **kwargs)
#
#     def print_all_1(self):
#         print(self)
#         for oChild in self._lChildren:
#             oChild.print_all_1()
#
#     def print_all_2(self):
#         def gen(o):
#             lAll = [o, ]
#             while lAll:
#                 oNext = lAll.pop(0)
#                 lAll.extend(oNext._lChildren)
#                 yield oNext
#
#         for oNode in gen(self):
#             print(oNode)
#
#
# oRoot = Node("root")
# oChild1 = Node("child1")
# oChild2 = Node("child2")
# oChild3 = Node("child3")
# oChild4 = Node("child4")
# oChild5 = Node("child5")
# oChild6 = Node("child6")
# oChild7 = Node("child7")
# oChild8 = Node("child8")
# oChild9 = Node("child9")
# oChild10 = Node("child10")
#
# oRoot.append(oChild1)
# oRoot.append(oChild2)
# oRoot.append(oChild3)
# oChild1.append(oChild4)
# oChild1.append(oChild5)
# oChild2.append(oChild6)
# oChild4.append(oChild7)
# oChild3.append(oChild8)
# oChild3.append(oChild9)
# oChild6.append(oChild10)

def f1(lIn):
    l1 = sorted(lIn)
    l2 = [i for i in l1 if i<0.5]
    return [i*i for i in l2]
def f2(lIn):
    l1 = [i for i in lIn if i<0.5]
    l2 = sorted(l1)
    return [i*i for i in l2]
def f3(lIn):
    l1 = [i*i for i in lIn]
    l2 = sorted(l1)
    return [i for i in l1 if i<(0.5*0.5)]


if __name__ == '__main__':
    # oRoot.print_all_2()
    import cProfile
    lIn = [random.random() for i in range(100000)]
    cProfile.run('f1(lIn)')
    cProfile.run('f2(lIn)')
    cProfile.run('f3(lIn)')
