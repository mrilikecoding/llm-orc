step1.py
def step1(x):
    return x * 2

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

main.py
from step8 import step8
print(step8(1))