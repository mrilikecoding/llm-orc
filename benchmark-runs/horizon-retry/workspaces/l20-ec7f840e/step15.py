step15.py
import step14
def f15(x): return step14.f14(x) + 1

step16.py
import step15
def f16(x): return step15.f15(x) + 1

step17.py
import step16
def f17(x): return step16.f16(x) + 1

step18.py
import step17
def f18(x): return step17.f17(x) + 1

step19.py
import step18
def f19(x): return step18.f18(x) + 1

step20.py
import step19
print(step19.f19(0))