import json,time,urllib.request
SYS=("You are a coding assistant. Given a programming task or question,\n"
"respond with the most useful code or guidance you can produce.\n"
"Keep responses focused. Show the code change or example directly.\n"
"When you are uncertain, say so rather than fabricating APIs or\n"
"file paths. Format code with appropriate fenced blocks.\n")
USER=("Add a 'done' command to the CLI in todo/cli.py that marks a todo complete by id "
"using TodoStore.complete. The existing cli.py uses argparse with subparsers for 'add' and 'list'. "
"Show the full updated build_parser and main functions.")
p={"model":"qwen3:8b","messages":[{"role":"system","content":SYS},{"role":"user","content":USER}],
   "stream":False,"options":{"num_predict":1500,"temperature":0,"seed":11}}   # NOTE: no "think" key at all
req=urllib.request.Request("http://127.0.0.1:11434/api/chat",data=json.dumps(p).encode(),
    headers={"Content-Type":"application/json"})
t0=time.time()
with urllib.request.urlopen(req,timeout=600) as r: b=json.load(r)
ns=1e9
print(json.dumps(dict(arm="think key ABSENT (what llm-orc sends today for code-generator)",
  wall=round(time.time()-t0,2), pe_tok=b.get("prompt_eval_count"),
  gen_tok=b.get("eval_count"), gen_s=round(b.get("eval_duration",0)/ns,2),
  thinking_chars=len(b["message"].get("thinking") or ""),
  content_chars=len(b["message"].get("content") or "")),indent=1))
