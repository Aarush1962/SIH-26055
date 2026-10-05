from pathlib import Path
from collections import deque
import json
import math
import uuid

import numpy as np
import torch
from torch import nn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "train_final"
RESULTS = ROOT / "final_results"
VAL = ROOT / "val_RFEnvs"
STATIC = Path(__file__).resolve().parent / "static"

FREQ_MIN_MHZ = 0
FREQ_MAX_MHZ = 18000
FREQ_BIN_MHZ = 500
TIME_BIN_S = 0.05
N_BANDS = 36
SEQ_LEN = 12
STATE_DIM = 7 * N_BANDS + 2
CANDIDATE_DIM = 7
GRU_HIDDEN = 128
ACTION_EMBED = 16
Q_HIDDEN = 128
N_HEADS = 3
MAX_SCAN_GAP = 40
STALE_BONUS = 0.12

FREQ_CENTRES = (np.arange(N_BANDS) * FREQ_BIN_MHZ + FREQ_BIN_MHZ / 2).astype(float)
DEVICE = torch.device("cpu")

app = FastAPI(title="Smart Scan Strategy API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class BeliefUpdater(nn.Module):
    def __init__(self, n_bands=36, hidden_size=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * n_bands + 1, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, n_bands)
        )
        self.n_bands = n_bands

    def forward(self, probabilities, actions, observations):
        one_hot = torch.zeros(actions.shape[0], self.n_bands, device=actions.device)
        one_hot.scatter_(1, actions.unsqueeze(1), 1.0)
        obs = observations.float().unsqueeze(1)
        return self.network(torch.cat([probabilities, one_hot, obs], dim=1))


class NeuralBanditHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(STATE_DIM, GRU_HIDDEN, batch_first=True)
        self.action_embedding = nn.Embedding(N_BANDS, ACTION_EMBED)
        self.q_network = nn.Sequential(
            nn.Linear(GRU_HIDDEN + CANDIDATE_DIM + ACTION_EMBED, Q_HIDDEN),
            nn.LayerNorm(Q_HIDDEN), nn.ReLU(),
            nn.Linear(Q_HIDDEN, Q_HIDDEN // 2), nn.ReLU(),
            nn.Linear(Q_HIDDEN // 2, 1)
        )

    def forward(self, state_sequence, candidate_features):
        _, hidden = self.gru(state_sequence)
        latent = hidden[-1].unsqueeze(1).expand(-1, N_BANDS, -1)
        ids = torch.arange(N_BANDS, device=state_sequence.device)
        emb = self.action_embedding(ids).unsqueeze(0).expand(state_sequence.shape[0], -1, -1)
        return self.q_network(torch.cat([latent, candidate_features, emb], dim=-1)).squeeze(-1)


# Frozen deployment artifacts are loaded once at startup.
recursive_ckpt = torch.load(TRAIN / "recursive_belief_updater.pt", map_location=DEVICE, weights_only=False)
recursive_cfg = recursive_ckpt["model_config"]
recursive_model = BeliefUpdater(recursive_cfg["n_freq_bins"], recursive_cfg["hidden_size"]).to(DEVICE)
recursive_model.load_state_dict(recursive_ckpt["model_state_dict"])
recursive_model.eval()
P_BASELINE = np.asarray(recursive_ckpt["P_baseline"], dtype=np.float32)

bandit_ckpt = torch.load(TRAIN / "experiment52_counterfactual_offline_rl_best.pt", map_location=DEVICE, weights_only=False)
bandit_heads = []
for state_dict in bandit_ckpt["model_state_dict"]:
    head = NeuralBanditHead().to(DEVICE)
    head.load_state_dict(state_dict)
    head.eval()
    bandit_heads.append(head)

ENV_CACHE = {}
SESSIONS = {}


def load_env(name: str) -> np.ndarray:
    safe = Path(name).name
    if not safe.endswith(".npy"):
        safe += ".npy"
    path = VAL / safe
    if not path.exists():
        raise HTTPException(404, f"Environment not found: {safe}")
    if safe not in ENV_CACHE:
        ENV_CACHE[safe] = np.asarray(np.load(path), dtype=np.float32)
    env = ENV_CACHE[safe]
    if env.ndim != 2 or env.shape[1] != N_BANDS:
        raise HTTPException(500, f"Invalid environment shape: {env.shape}")
    return env


def events_for(env):
    events = []
    eid = 0
    for band in range(N_BANDS):
        active = env[:, band] > 0
        start = None
        for t, on in enumerate(active):
            if on and start is None:
                start = t
            elif not on and start is not None:
                events.append({"id": eid, "band": band, "start": start, "end": t - 1, "duration": t - start})
                eid += 1; start = None
        if start is not None:
            events.append({"id": eid, "band": band, "start": start, "end": len(env) - 1, "duration": len(env) - start})
            eid += 1
    events.sort(key=lambda x: (x["start"], x["band"]))
    for i, e in enumerate(events): e["id"] = i
    return events


class Memory:
    def __init__(self): self.reset()
    def reset(self):
        self.scan_counts = np.zeros(N_BANDS, np.float32)
        self.hit_counts = np.zeros(N_BANDS, np.float32)
        self.age = np.full(N_BANDS, np.inf, np.float32)
        self.time_since_hit = np.full(N_BANDS, np.inf, np.float32)
        self.last_obs = np.zeros(N_BANDS, np.float32)
        self.recent = [deque(maxlen=10) for _ in range(N_BANDS)]
        self.previous = -1
        self.step = 0
    def norm(self, x):
        x = np.asarray(x, np.float32)
        return np.minimum(np.where(np.isfinite(x), x, MAX_SCAN_GAP), MAX_SCAN_GAP) / MAX_SCAN_GAP
    def hit_rates(self):
        return np.array([sum(h)/len(h) if h else 0.0 for h in self.recent], np.float32)
    def state(self, belief):
        scan = np.clip(np.log1p(self.scan_counts)/np.log1p(50.0), 0, 1)
        prev = np.zeros(N_BANDS, np.float32)
        if 0 <= self.previous < N_BANDS: prev[self.previous] = 1
        phase = 2*np.pi*self.step/200
        return np.concatenate([belief, self.norm(self.age), self.hit_rates(), self.norm(self.time_since_hit), self.last_obs, scan, prev, [np.sin(phase), np.cos(phase)]]).astype(np.float32)
    def candidates(self, belief):
        scan = np.clip(np.log1p(self.scan_counts)/np.log1p(50.0), 0, 1)
        pos = np.arange(N_BANDS, dtype=np.float32)/35.0
        return np.stack([belief, self.norm(self.age), self.hit_rates(), self.norm(self.time_since_hit), self.last_obs, scan, pos], axis=1).astype(np.float32)
    def update(self, action, obs):
        hit = int(obs > 0)
        self.age += 1
        finite = np.isfinite(self.time_since_hit)
        self.time_since_hit[finite] += 1
        self.age[action] = 0
        self.scan_counts[action] += 1
        self.hit_counts[action] += hit
        self.last_obs[action] = hit
        self.recent[action].append(hit)
        if hit: self.time_since_hit[action] = 0
        self.previous = int(action); self.step += 1


class Scheduler:
    def __init__(self):
        self.memory = Memory(); self.history = deque(maxlen=SEQ_LEN)
    def reset(self):
        self.memory.reset(); self.history.clear()
    def choose(self, belief):
        state = self.memory.state(belief)
        cand = self.memory.candidates(belief)
        self.history.append(state.copy())
        seq = list(self.history)
        if len(seq) < SEQ_LEN: seq = [seq[0].copy()]*(SEQ_LEN-len(seq)) + seq
        if self.memory.step < N_BANDS: return self.memory.step, None
        st = torch.tensor(np.stack(seq)[None], dtype=torch.float32)
        cf = torch.tensor(cand[None], dtype=torch.float32)
        with torch.no_grad():
            qs = np.stack([h(st, cf)[0].numpy() for h in bandit_heads])
        mean_q = qs.mean(axis=0)
        score = mean_q + STALE_BONUS * self.memory.norm(self.memory.age)
        return int(np.argmax(score)), score
    def update(self, action, obs): self.memory.update(action, obs)


def belief_step(belief, action, obs):
    with torch.no_grad():
        b = torch.tensor(belief[None], dtype=torch.float32)
        a = torch.tensor([action], dtype=torch.long)
        o = torch.tensor([obs], dtype=torch.float32)
        return torch.sigmoid(recursive_model(b, a, o))[0].numpy()


def make_step(session, env, t):
    belief = session["belief"]
    action, scores = session["scheduler"].choose(belief)
    obs = float(env[t, action])
    next_belief = belief_step(belief, action, obs)
    session["scheduler"].update(action, obs)
    event_ids = session["active_lookup"][t]
    new_hits = []
    for eid in event_ids:
        if eid not in session["intercepted"] and session["events"][eid]["band"] == action:
            session["intercepted"].add(eid)
            e = session["events"][eid]
            new_hits.append({**e, "interception_time": t, "delay_s": (t-e["start"])*TIME_BIN_S})
    session["belief"] = next_belief
    session["t"] = t + 1
    if new_hits: session["hits"].extend(new_hits)
    priorities = np.argsort(-next_belief)[:5].tolist()
    result = {
        "timestep": t, "time_s": t*TIME_BIN_S,
        "selected_band": action, "selected_frequency_mhz": float(FREQ_CENTRES[action]),
        "observation": int(obs), "belief": np.round(next_belief, 5).tolist(),
        "top_priority_bands": [{"band": int(b), "frequency_mhz": float(FREQ_CENTRES[b]), "belief": float(next_belief[b])} for b in priorities],
        "new_intercepts": new_hits,
        "intercepted_count": len(session["hits"]),
        "total_events": len(session["events"]),
        "scan_hit_rate": float(sum(session["observations"])/len(session["observations"])) if session["observations"] else 0.0,
        "scores": np.round(scores, 5).tolist() if scores is not None else None,
    }
    session["observations"].append(obs); session["actions"].append(action); session["history"].append(result)
    return result


@app.get("/")
def index(): return FileResponse(STATIC / "index.html")

@app.get("/api/health")
def health():
    return {"status":"ok", "models_loaded": True, "device":"cpu"}

@app.get("/api/metadata")
def metadata():
    return {"project":"SIH-26055", "title":"Smart Scan Strategy for Electronic Warfare", "frequency":{"min_mhz":0,"max_mhz":18000,"bin_mhz":500,"bands":36}, "time_bin_s":TIME_BIN_S, "scheduler":{"model":"Experiment-52 frozen ensemble","heads":len(bandit_heads),"sequence_length":SEQ_LEN,"state_dim":STATE_DIM,"initial_sweep":True,"stale_bonus":STALE_BONUS}, "belief_model":{"architecture":"73 → 128 → 128 → 36"}}

@app.get("/api/results/summary")
def results_summary():
    return json.loads((RESULTS/"final_report.json").read_text())

@app.get("/api/results/environments")
def result_envs():
    data=json.loads((RESULTS/"final_report.json").read_text())
    return data["per_file_metrics"]

@app.get("/api/environments")
def environments():
    return [{"id":p.stem,"file":p.name,"time_steps":int(np.load(p,mmap_mode='r').shape[0]),"duration_s":float(np.load(p,mmap_mode='r').shape[0]*TIME_BIN_S)} for p in sorted(VAL.glob('*.npy'))]

@app.get("/api/simulation/{session_id}")
def simulation_state(session_id: str):
    s=SESSIONS.get(session_id)
    if not s: raise HTTPException(404,"Simulation not found")
    return {"session_id":session_id,"environment":s["environment"],"done":s["t"]>=len(s["env"]),"timestep":s["t"],"steps":s["history"][-30:],"latest":s["history"][-1] if s["history"] else None,"intercepted":len(s["hits"]),"total_events":len(s["events"])}

class StartRequest(BaseModel):
    environment: str = "config_100.npy"

@app.post("/api/simulation/start")
def start_sim(req: StartRequest):
    env=load_env(req.environment)
    ev=events_for(env)
    lookup=[[] for _ in range(len(env))]
    for e in ev:
        for t in range(e["start"],e["end"]+1): lookup[t].append(e["id"])
    sid=str(uuid.uuid4())
    SESSIONS[sid]={"environment":Path(req.environment).stem,"env":env,"events":ev,"active_lookup":lookup,"scheduler":Scheduler(),"belief":P_BASELINE.copy(),"t":0,"hits":[],"intercepted":set(),"observations":[],"actions":[],"history":[]}
    return {"session_id":sid,"environment":Path(req.environment).stem,"time_steps":len(env),"total_events":len(ev)}

@app.post("/api/simulation/{session_id}/step")
def step_sim(session_id: str):
    s=SESSIONS.get(session_id)
    if not s: raise HTTPException(404,"Simulation not found")
    if s["t"]>=len(s["env"]): return {"done":True}
    out=make_step(s,s["env"],s["t"])
    return {"done":s["t"]>=len(s["env"]),**out}

@app.get("/api/simulation/{session_id}/spectrum")
def spectrum(session_id: str):
    s=SESSIONS.get(session_id)
    if not s: raise HTTPException(404,"Simulation not found")
    t=max(0,min(s["t"],len(s["env"])-1))
    return {"timestep":t,"time_s":t*TIME_BIN_S,"activity":s["env"][t].astype(int).tolist()}

app.mount("/static", StaticFiles(directory=STATIC), name="static")
