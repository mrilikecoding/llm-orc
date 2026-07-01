base.py
def start(x):
    return x + 1

step1.py
from base import start
def step1(x):
    return start(x) * 2

step2.py
from step1 import step1
def step2(x):
    return step1(x) * 2

step3.py
from step2 import step2
def step3(x):
    return step2(x) * 2

step4.py
from step3 import step3
def step4(x):
    return step3(x) * 2

step5.py
from step4 import step4
def step5(x):
    return step4(x) * 2

step6.py
from step5 import step5
def step6(x):
    return step5(x) * 2

step7.py
from step6 import step6
def step7(x):
    return step6(x) * 2

step8.py
from step7 import step7
def step8(x):
    return step7(x) * 2

step9.py
from step8 import step8
def step9(x):
    return step8(x) * 2

step10.py
from step9 import step9
def step10(x):
    return step9(x) * 2

step11.py
from step10 import step10
def step11(x):
    return step10(x) * 2

step12.py
from step11 import step11
def step12(x):
    return step11(x) * 2

step13.py
from step12 import step12
def step13(x):
    return step12(x) * 2

step14.py
from step13 import step13
def step14(x):
    return step13(x) * 2

step15.py
from step14 import step14
def step15(x):
    return step14(x) * 2

step16.py
from step15 import step15
def step16(x):
    return step15(x) * 2

step17.py
from step16 import step16
def step17(x):
    return step16(x) * 2

step18.py
from step17 import step17
def step18(x):
    return step17(x) * 2

step19.py
from step18 import step18
def step19(x):
    return step18(x) * 2

step20.py
from step19 import step19
def step20(x):
    return step19(x) * 2

step21.py
from step20 import step20
def step21(x):
    return step20(x) * 2

step22.py
from step21 import step21
def step22(x):
    return step21(x) * 2

step23.py
from step22 import step22
def step23(x):
    return step22(x) * 2

step24.py
from step23 import step23
def step24(x):
    return step23(x) * 2

step25.py
from step24 import step24
def step25(x):
    return step24(x) * 2

step26.py
from step25 import step25
def step26(x):
    return step25(x) * 2

step27.py
from step26 import step26
def step27(x):
    return step26(x) * 2

step28.py
from step27 import step27
def step28(x):
    return step27(x) * 2

main.py
from step28 import step28
print(step28(1))