step4.py
from step3 import f3
def f4(x): return f3(x) + 1

step5.py
from step4 import f4
def f5(x): return f4(x) + 1

step6.py
from step5 import f5
def f6(x): return f5(x) + 1

step7.py
from step6 import f6
def f7(x): return f6(x) + 1

step8.py
from step7 import f7
def f8(x): return f7(x) + 1

step9.py
from step8 import f8
def f9(x): return f8(x) + 1