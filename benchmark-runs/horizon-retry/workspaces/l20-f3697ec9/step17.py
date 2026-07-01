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