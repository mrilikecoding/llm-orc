step11.py
from step10 import f10
def f11(x): return f10(x) + 1

step12.py
from step11 import f11
def f12(x): return f11(x) + 1

step13.py
from step12 import f12
def f13(x): return f12(x) + 1

step14.py
from step13 import f13
def f14(x): return f13(x) + 1

step15.py
from step14 import f14
print(f14(0))