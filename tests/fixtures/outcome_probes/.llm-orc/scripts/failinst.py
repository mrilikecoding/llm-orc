import sys
d = sys.stdin.read()
if '"b"' in d or '\\"b\\"' in d or d.strip().endswith('b'):
    sys.stderr.write("inst b fails\n"); sys.exit(4)
print("inst-ok " + d[:200].replace("\n"," "))
