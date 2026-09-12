import json,sys,time,urllib.request
BLOB="/Users/nathangreen/.ollama/models/blobs/sha256-a3de86cd1c132c822487ededd47a324c50491393e6565cd14bafa40d0b8e686f"
P_SHORT="Write a Python function that merges two sorted lists into one sorted list. Explain the algorithm step by step, then show the code.\n"
para=("The storage module keeps todos in a JSON file on disk. Each todo is a dict with an id, a text field, and a done flag. "
      "The TodoStore class loads the file on construction, exposes add, list, and complete methods, and writes the file back after every mutation. ")
P_LONG = "# Repository notes\n" + para*55 + "\nSummarize the above in one sentence.\n"
def post(url,payload,timeout=300):
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"})
    t0=time.time()
    with urllib.request.urlopen(req,timeout=timeout) as r: body=json.load(r)
    return time.time()-t0, body
def ollama(prompt,npred):
    wall,b=post("http://127.0.0.1:11434/api/generate",
        {"model":"qwen3:8b","prompt":prompt,"raw":True,"stream":False,
         "options":{"num_predict":npred,"temperature":0,"seed":7}})
    ns=1e9
    return dict(wall=wall, load_s=b.get("load_duration",0)/ns,
        pe_tok=b.get("prompt_eval_count",0), pe_s=b.get("prompt_eval_duration",0)/ns,
        gen_tok=b.get("eval_count",0), gen_s=b.get("eval_duration",0)/ns,
        total_s=b.get("total_duration",0)/ns)
def llamacpp(prompt,npred,port):
    wall,b=post(f"http://127.0.0.1:{port}/completion",
        {"prompt":prompt,"n_predict":npred,"temperature":0,"seed":7,"cache_prompt":True,"stream":False})
    t=b.get("timings",{})
    return dict(wall=wall, load_s=0.0,
        pe_tok=t.get("prompt_n",0), pe_s=t.get("prompt_ms",0)/1000,
        gen_tok=t.get("predicted_n",0), gen_s=t.get("predicted_ms",0)/1000,
        total_s=(t.get("prompt_ms",0)+t.get("predicted_ms",0))/1000)
def row(tag,r):
    pet = r['pe_tok']/r['pe_s'] if r['pe_s'] else 0
    gt  = r['gen_tok']/r['gen_s'] if r['gen_s'] else 0
    print(f"{tag:28s} wall={r['wall']:7.2f}s load={r['load_s']:6.2f}s "
          f"pe={r['pe_tok']:5d}tok/{r['pe_s']:6.2f}s={pet:7.1f}t/s "
          f"gen={r['gen_tok']:5d}tok/{r['gen_s']:6.2f}s={gt:6.2f}t/s")
    return dict(tag=tag,**r,pe_tps=pet,gen_tps=gt)
if __name__=="__main__":
    which=sys.argv[1]; port=sys.argv[2] if len(sys.argv)>2 else "8089"
    f = ollama if which=="ollama" else (lambda p,n: llamacpp(p,n,port))
    out=[]
    for rep in (1,2):
        out.append(row(f"{which} A-short-gen300 r{rep}", f(P_SHORT,300)))
    for rep in (1,2):
        out.append(row(f"{which} B-long3k-gen32 r{rep}", f(P_LONG,32)))
    for rep in (1,2):
        out.append(row(f"{which} C-repeatA-gen64 r{rep}", f(P_SHORT,64)))
    json.dump(out,open(f"/private/tmp/claude-501/-Users-nathangreen-Development-eddi-lab-llm-orc/32dba98e-5297-4a48-8e23-9f8b5e88d5e8/scratchpad/90-eval/bench-{which}.json","w"),indent=1)
