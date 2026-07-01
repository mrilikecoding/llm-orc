step6.py
from step5 import f5
def f6(x):
    return f5(x) + 1

step7.py
from step6 import f6
def f7(x):
    return f6(x) + 1

step8.py
from step7 import f7
def f8(x):
    return f7(x) + 1

step9.py
from step8 import f8
def f9(x):
    return f8(x) + 1

step10.py
from step9 import f9
def f10(x):
    return f9(x) + 1

step11.py
from step10 import f10
def f11(x):
    return f10(x) + 1

step12.py
from step11 import f11
def f12(x):
    return f11(x) + 1

step13.py
from step12 import f12
def f13(x):
    return f12(x) + 1

step14.py
from step13 import f13
def f14(x):
    return f13(x) + 1

step15.py
from step14 import f14
print(f14(0))