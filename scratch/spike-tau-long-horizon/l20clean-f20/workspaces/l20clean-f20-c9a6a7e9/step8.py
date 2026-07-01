def step8(x):
    from step7 import step7
    return step7(x) * 2

def step9(x):
    from step8 import step8
    return step8(x) * 2

def step10(x):
    from step9 import step9
    return step9(x) * 2

def step11(x):
    from step10 import step10
    return step10(x) * 2

def step12(x):
    from step11 import step11
    return step11(x) * 2

def step13(x):
    from step12 import step12
    return step12(x) * 2

def step14(x):
    from step13 import step13
    return step13(x) * 2

def step15(x):
    from step14 import step14
    return step14(x) * 2

def step16(x):
    from step15 import step15
    return step15(x) * 2

def step17(x):
    from step16 import step16
    return step16(x) * 2

def step18(x):
    from step17 import step17
    return step17(x) * 2

import step18

print(step18.step18(1))