import json, os, re, time, threading, pandas as pd
from collections import deque
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
BATCH = 10

# (model, RPM limit, RPD limit) -- daily quotas reset, using the full confirmed-working set.
# gemini-3.6-flash dropped: its 503 "high demand" errors burned its RPD budget without
# producing usable output.
MODELS = [
    ("gemini-3.1-flash-lite", 15, 500),
    ("gemini-3.5-flash-lite", 15, 500),
    ("gemini-3.8-flash", 5, 20),
    ("gemini-3-flash-preview", 5, 20),
    ("gemini-3.7-flash", 5, 20),
]

LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA',
          'PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

DEFINITIONS = """Definitions:
- ACL: anterior cruciate ligament tear/rupture/injury (partial or complete). Mere mild sprain-free "intact" = 0.
- MCL: medial collateral ligament injury/sprain/tear.
- Medial Meniscus: tear of medial meniscus (degeneration alone, grade I/II signal without tear = low).
- Lateral Meniscus: tear of lateral meniscus (same rule).
- Medial OA / Lateral OA / PF OA: osteoarthritis (cartilage loss, osteophytes, subchondral change) in medial tibiofemoral / lateral tibiofemoral / patellofemoral compartment.
- Effusion: joint effusion / excess fluid in the joint (not just a small physiologic amount).
- Synovitis: synovial inflammation/thickening/hypertrophy.
- Baker's: popliteal/Baker's/synovial cyst behind the knee.
- Contusion: bone contusion / bone bruise / marrow edema from trauma.
- Fracture: any fracture (incl. stress/insufficiency/avulsion)."""

PROMPT_ONE = """You are a musculoskeletal radiologist. The knee MRI report below may be in any language.
For each finding output the probability (0-1) that it is PRESENT in this knee per the report.
{defs}
If the report explicitly says normal or does not mention a finding, give a low probability.
Return JSON only, keys exactly: {keys}.

REPORT:
{report}"""

PROMPT_BATCH = """You are a musculoskeletal radiologist. Below are {n} knee MRI reports, each in any language, tagged with an "index" number.
For each report, and for each finding, output the probability (0-1) that it is PRESENT in that knee per that report.
{defs}
If a report explicitly says normal or does not mention a finding, give a low probability.
Return a JSON array with exactly {n} objects, one per report, each with its matching "index" field (copy it exactly, do not renumber or skip) plus keys exactly: {keys}.

REPORTS:
{reports}"""

schema_one = types.Schema(type=types.Type.OBJECT,
    properties={k: types.Schema(type=types.Type.NUMBER) for k in LABELS},
    required=LABELS)

batch_item_schema = types.Schema(type=types.Type.OBJECT,
    properties={"index": types.Schema(type=types.Type.INTEGER),
                **{k: types.Schema(type=types.Type.NUMBER) for k in LABELS}},
    required=["index"] + LABELS)
schema_batch = types.Schema(type=types.Type.ARRAY, items=batch_item_schema)


class ModelExhausted(Exception):
    pass


class RateLimiter:
    """Per-model limiter: enforces RPM via a sliding window, RPD via a hard counter."""
    def __init__(self, model, rpm, rpd):
        self.model = model
        self.rpm = rpm
        self.rpd = rpd
        self.used = 0
        self.timestamps = deque()
        self.lock = threading.Lock()
        self.alive = True

    def acquire(self):
        """Blocks until it's safe to make one more call. Raises ModelExhausted if RPD is spent."""
        with self.lock:
            if not self.alive or self.used >= self.rpd:
                self.alive = False
                raise ModelExhausted(self.model)
            now = time.monotonic()
            while self.timestamps and now - self.timestamps[0] > 60:
                self.timestamps.popleft()
            if len(self.timestamps) >= self.rpm:
                sleep_for = 60 - (now - self.timestamps[0]) + 0.1
            else:
                sleep_for = 0
        if sleep_for > 0:
            time.sleep(sleep_for)
        with self.lock:
            self.timestamps.append(time.monotonic())
            self.used += 1

    def mark_exhausted(self):
        with self.lock:
            self.alive = False


def _is_daily_quota_error(e):
    msg = str(e)
    if "generate_content_free_tier_requests" not in msg:
        return False
    m = re.search(r"retry in (?:(\d+)h)?", msg)
    return bool(m and m.group(1))


def label_one(limiter, uid_report):
    uid, rep = uid_report
    for attempt in range(5):
        limiter.acquire()
        try:
            r = client.models.generate_content(
                model=limiter.model,
                contents=PROMPT_ONE.format(keys=LABELS, report=rep, defs=DEFINITIONS),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema_one, temperature=0))
            d = json.loads(r.text)
            return {"StudyInstanceUID": uid, **{k: float(d[k]) for k in LABELS}}
        except Exception as e:
            if _is_daily_quota_error(e):
                limiter.mark_exhausted()
                raise ModelExhausted(limiter.model) from e
            time.sleep(min(2 ** attempt, 20))
    return {"StudyInstanceUID": uid, **{k: None for k in LABELS}}


def label_batch(limiter, batch):
    uids = [uid for uid, _ in batch]
    reports_text = "\n\n".join(f'[index={i}]\n{rep}' for i, (_, rep) in enumerate(batch))
    for attempt in range(5):
        limiter.acquire()
        try:
            r = client.models.generate_content(
                model=limiter.model,
                contents=PROMPT_BATCH.format(n=len(batch), keys=LABELS, reports=reports_text, defs=DEFINITIONS),
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema_batch, temperature=0))
            items = json.loads(r.text)
            by_index = {int(it["index"]): it for it in items if "index" in it}
            results, missing = {}, []
            for i, uid in enumerate(uids):
                it = by_index.get(i)
                if it is None:
                    missing.append((uid, batch[i][1]))
                else:
                    results[uid] = {"StudyInstanceUID": uid, **{k: float(it[k]) for k in LABELS}}
            rows = [results[uid] for uid in uids if uid in results]
            if missing:
                rows.extend(label_one(limiter, m) for m in missing)
            return rows
        except ModelExhausted:
            raise
        except Exception as e:
            if _is_daily_quota_error(e):
                limiter.mark_exhausted()
                raise ModelExhausted(limiter.model) from e
            time.sleep(min(2 ** attempt, 20))
    return [label_one(limiter, item) for item in batch]


train = pd.read_csv("train.csv")   # upload/download train.csv to wherever you run this
out_path = "report_labels.jsonl"
done = set()
if os.path.exists(out_path):
    done = {json.loads(l)["StudyInstanceUID"] for l in open(out_path)}
todo = [(u, r) for u, r in zip(train.StudyInstanceUID, train.Report) if u not in done]
batches = deque(todo[i:i+BATCH] for i in range(0, len(todo), BATCH))

write_lock = threading.Lock()
f = open(out_path, "a")

def worker(model, rpm, rpd):
    limiter = RateLimiter(model, rpm, rpd)
    while True:
        try:
            batch = batches.popleft()
        except IndexError:
            return
        try:
            rows = label_batch(limiter, batch)
        except ModelExhausted:
            batches.appendleft(batch)  # give it back for another model to pick up
            return
        with write_lock:
            for row in rows:
                f.write(json.dumps(row) + "\n")
            f.flush()

threads = [threading.Thread(target=worker, args=m) for m in MODELS]
for t in threads: t.start()
for t in threads: t.join()
f.close()

print(f"{len(todo) - len(batches)*BATCH} rows attempted, {len(batches)} batches left unprocessed (quota exhausted for today).")
pd.read_json(out_path, lines=True).to_csv("report_labels.csv", index=False)
