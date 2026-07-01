step14.py
from step13 import f13
def f14(x): return f13(x) + 1

step15.py
from step14 import f14
def f15(x): return f14(x) + 1

step16.py
from step15 import f15
def f16(x): return f15(x) + 1

step17.py
from step16 import f16
def f17(x): return f16(x) + 1

step18.py
from step17 import f17
def f18(x): return f17(x) + 1

step19.py
from step18 import f18
def f19(x): return f18(x) + 1

step20.py
from step19 import f19
print(f19(0))