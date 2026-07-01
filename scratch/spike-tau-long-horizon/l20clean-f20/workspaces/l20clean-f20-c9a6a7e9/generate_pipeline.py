with open('base.py', 'w') as f:
    f.write("def start(x):\n    return x + 1\n")

for n in range(1, 19):
    if n == 1:
        import_mod = 'base'
        func_name = 'start'
    else:
        import_mod = f'step{n-1}'
        func_name = f'step{n-1}'
    content = f"import {import_mod}\n\ndef step{n}(x):\n    return {import_mod}.{func_name}(x) * 2\n"
    with open(f'step{n}.py', 'w') as f:
        f.write(content)

with open('main.py', 'w') as f:
    f.write("import step18\n\nprint(step18.step18(1))\n")