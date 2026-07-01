base.py
def start(x): return x + 1

step1.py
import base
def step1(x): return base.start(x) * 2

step2.py
import step1
def step2(x): return step1.step1(x) * 2

step3.py
import step2
def step3(x): return step2.step2(x) * 2

step4.py
import step3
def step4(x): return step3.step3(x) * 2

step5.py
import step4
def step5(x): return step4.step4(x) * 2

step6.py
import step5
def step6(x): return step5.step5(x) * 2

step7.py
import step6
def step7(x): return step6.step6(x) * 2

step8.py
import step7
def step8(x): return step7.step7(x) * 2

main.py
import step8
print(step8.step8(1))