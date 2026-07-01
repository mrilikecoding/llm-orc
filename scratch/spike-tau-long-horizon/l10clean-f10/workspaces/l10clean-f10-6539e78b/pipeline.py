def start(x): return x + 1
def step1(x): return start(x) * 2
def step2(x): return step1(x) * 2
def step3(x): return step2(x) * 2
def step4(x): return step3(x) * 2
def step5(x): return step4(x) * 2
def step6(x): return step5(x) * 2
def step7(x): return step6(x) * 2
def step8(x): return step7(x) * 2
def main(): print(step8(1))