# -*- coding: utf-8 -*-
"""run_eval.py — Evaluation harness cho Bo_Cai_Chatbot_Luat_v2.

Import app.py truc tiep (khong qua Streamlit). Timeout 60s/case.
Ghi dung 10 file CSV: test_cases, eval_retrieval, eval_generation,
eval_tier_scores, eval_summary, eval_refusal, error_log,
regression_tests, model_comparison, failure_cases.

Usage:
  python run_eval.py --phase=1    --tier=all --config=baseline
  python run_eval.py --phase=full --tier=all --config=improved
"""
import argparse
import csv
import math
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import app as A  # noqa: E402

CASE_TIMEOUT_S = 60
TOP_K = 8

REFUSAL_PHRASES = [
    "không tìm thấy", "không thể", "ngoài phạm vi", "không thuộc phạm vi",
    "không hỗ trợ", "không tư vấn", "không dự đoán", "không trả lời",
    "từ chối", "không có thông tin", "chưa có thông tin",
]

# ---------- text metrics (native, khong cai package nang) ----------
_TOKEN_RE = re.compile(r"[a-z0-9àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ]+", re.IGNORECASE)


def tokenize(s):
    return _TOKEN_RE.findall(s or "")


def lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            cur[j] = prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def rouge_l(pred, ref):
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r:
        return 0.0
    lcs = lcs_len(p, r)
    prec, rec = lcs / len(p), lcs / len(r)
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def bleu(pred, ref):
    p, r = tokenize(pred), tokenize(ref)
    if not p or not r:
        return 0.0
    precs = []
    for n in range(1, 5):
        pn = Counter(tuple(p[i:i + n]) for i in range(len(p) - n + 1))
        rn = Counter(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
        match = sum(min(c, rn[g]) for g, c in pn.items())
        precs.append(match / max(1, sum(pn.values())))
    geo = math.exp(sum(math.log(x) for x in precs if x > 0) / 4) if any(x > 0 for x in precs) else 0.0
    bp = min(1.0, math.exp(1 - len(r) / len(p))) if len(p) > 0 else 0.0
    return geo * bp


def token_f1(pred, ref):
    p, r = Counter(tokenize(pred)), Counter(tokenize(ref))
    if not p or not r:
        return 0.0
    inter = sum(min(c, r[g]) for g, c in p.items())
    prec, rec = inter / sum(p.values()), inter / sum(r.values())
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def extract_articles(text):
    seen, out = set(), []
    for m in re.findall(r"[Đđ]iều\s*(\d+)", text or ""):
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def count_numbered_steps(text):
    n = 0
    for line in (text or "").split("\n"):
        if re.match(r"^\s*\d{1,2}[\.\)]\s+\S", line.strip()):
            n += 1
    return n


def is_refusal_text(text):
    t = (text or "").lower()
    return any(p in t for p in REFUSAL_PHRASES)


# ---------- load ----------
def load_cases(phase, tier_filter):
    rows = list(csv.DictReader(open(BASE / "test_cases.csv", encoding="utf-8")))
    if tier_filter != "all":
        rows = [r for r in rows if r["tier"] == tier_filter]
    if phase == "1":
        sel = []
        refuse = [r for r in rows if r["expected_behavior"] == "refuse"]
        sel.extend(refuse)
        sel.extend(r for r in rows if "REGRESSION_CASE" in r["notes"])
        per_tier = {}
        for r in rows:
            if r["expected_behavior"] == "refuse" or "REGRESSION_CASE" in r["notes"]:
                continue
            if r["tier"] not in per_tier:
                per_tier[r["tier"]] = []
            if len(per_tier[r["tier"]]) < 2:
                per_tier[r["tier"]].append(r)
        for t in sorted(per_tier, key=int):
            sel.extend(per_tier[t])
        # dedupe giu thu tu
        seen, out = set(), []
        for r in sel:
            if r["id"] not in seen:
                seen.add(r["id"])
                out.append(r)
        return out
    return rows


def parse_sets(case):
    gold_arts = [a for a in case["ground_truth_articles"].split(";") if a.strip()]
    gold_laws = [l for l in case["ground_truth_sources"].split(";") if l.strip()]
    return gold_arts, gold_laws


def compute_metrics(case, answer, sources, latency_ms, retr_ms, error):
    qid = case["id"]
    gold_arts, gold_laws = parse_sets(case)
    gold_set = set(gold_arts)

    src_titles = [s.get("title", "") for s in (sources or [])]
    retr_arts, seen_a = [], set()
    for t in src_titles:
        for m in re.findall(r"[Đđ]iều\s*(\d+)", t):
            if m not in seen_a:
                seen_a.add(m)
                retr_arts.append(m)
    retr_laws, seen_l = [], set()
    for t in src_titles:
        law = t.split(" - ")[0].strip()
        if law and law not in seen_l:
            seen_l.add(law)
            retr_laws.append(law)

    def hit_at(k):
        return 1 if any(a in retr_arts[:k] for a in gold_set) else 0

    hit1, hit3, hit5 = hit_at(1), hit_at(3), hit_at(5)
    mrr = 0.0
    if gold_set:
        for rank, a in enumerate(retr_arts, 1):
            if a in gold_set:
                mrr = 1.0 / rank
                break
    dcg = sum((1.0 / math.log2(i + 2)) for i, a in enumerate(retr_arts[:5]) if a in gold_set)
    idcg = sum((1.0 / math.log2(i + 2)) for i in range(min(len(gold_set), 5)))
    ndcg5 = dcg / idcg if idcg > 0 else 0.0

    cross_count = ""
    if len(gold_laws) >= 2:
        cross_count = sum(1 for l in gold_laws if l in retr_laws)
        cross_count = min(cross_count, 2)

    cited = extract_articles(answer)
    cited_set = set(cited)
    if cited_set:
        precision = len(cited_set & gold_set) / len(cited_set)
    else:
        precision = 1.0 if not gold_set else 0.0
    if gold_set:
        recall = len(cited_set & gold_set) / len(gold_set)
    else:
        recall = 1.0 if not cited_set else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    faithfulness = 1 if (not cited_set or cited_set <= set(retr_arts)) else 0

    gt = case["ground_truth_answer"]
    rl = rouge_l(answer, gt)
    bl = bleu(answer, gt)
    tf1 = token_f1(answer, gt)

    step = 1.0
    if "PROCEDURE" in case["notes"]:
        gt_steps = count_numbered_steps(gt)
        an_steps = count_numbered_steps(answer)
        step = min(1.0, an_steps / gt_steps) if gt_steps > 0 else 1.0

    case_score = min(f1, 1.0) * 5 + min(rl, 1.0) * 3 + min(step, 1.0) * 2

    return {
        "qid": qid, "tier": case["tier"], "cat": case["category"],
        "question": case["question"], "expected": case["expected_behavior"],
        "answer": answer, "gt": gt, "error": error, "latency_ms": latency_ms,
        "retr_ms": retr_ms, "retr_arts": retr_arts, "retr_laws": retr_laws,
        "hit1": hit1, "hit3": hit3, "hit5": hit5, "mrr": mrr, "ndcg5": ndcg5,
        "cross_count": cross_count, "cited": cited, "gold_arts": gold_arts,
        "gold_laws": gold_laws, "precision": precision, "recall": recall,
        "f1": f1, "rl": rl, "bl": bl, "tf1": tf1,
        "faithfulness": faithfulness, "step": step, "score": case_score,
        "notes": case["notes"],
    }


def classify_error(m, actual_refusal_ok):
    if m["error"]:
        return "timeout"
    if m["expected"] == "refuse" and not actual_refusal_ok:
        return "refusal_wrong"
    if m["expected"] == "answer":
        if len(m["gold_laws"]) >= 2 and (m["cross_count"] == "" or m["cross_count"] < 2):
            return "E3_cross_law_miss"
        if m["step"] < 1.0 and "PROCEDURE" in m["notes"]:
            return "E4_incomplete_steps"
        if m["recall"] < 1.0 and not m["cited"]:
            return "E1_missing_citation"
        if m["precision"] < 1.0:
            return "E2_wrong_article"
        if m["faithfulness"] == 0:
            return "hallucination"
        return "low_quality" if m["score"] < 6.0 else ""
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["1", "full"], default="full")
    ap.add_argument("--tier", choices=["1", "2", "3", "4", "5", "all"], default="all")
    ap.add_argument("--config", choices=["baseline", "improved"], default="improved")
    args = ap.parse_args()

    cases = load_cases(args.phase, args.tier)
    print(f"[run_eval] config={args.config} phase={args.phase} tier={args.tier} cases={len(cases)}", flush=True)

    print("[run_eval] loading resources...", flush=True)
    t0 = time.time()
    retriever, llm, all_docs, sorted_laws, reference_map = A.load_resources()
    print(f"[run_eval] resources loaded in {time.time()-t0:.1f}s", flush=True)

    def qa(question):
        res = A.answer_question(
            question, retriever, llm, all_docs, reference_map,
            filter_law="Tất cả", history=[], fast_mode=True, on_token=None,
        )
        return res

    # warmup
    try:
        wres = qa("Án treo là gì?")
        print(f"[run_eval] warmup ok ({wres['time']['total']:.1f}s)", flush=True)
    except Exception as e:
        print(f"[run_eval] warmup failed: {e}", flush=True)

    results = []
    for i, case in enumerate(cases, 1):
        q = case["question"]
        t_start = time.time()
        error = ""
        answer, sources, retr_ms = "", [], 0.0
        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(qa, q)
                try:
                    res = fut.result(timeout=CASE_TIMEOUT_S)
                except FutureTimeout:
                    error = "timeout"
                    try:
                        fut.result(timeout=300)  # drain de tranh dong thoi 2 gen
                    except Exception:
                        print(f"[run_eval] FATAL: gen thread still running after drain for {case['id']}", flush=True)
                        sys.exit(2)
        except Exception as e:
            error = f"exception:{type(e).__name__}"
            print(f"[run_eval] {case['id']} exception: {e}", flush=True)
        if not error:
            answer = res.get("answer") or ""
            sources = res.get("sources") or []
            retr_ms = (res.get("time") or {}).get("retrieve", 0.0) * 1000
        latency_ms = (time.time() - t_start) * 1000
        m = compute_metrics(case, answer, sources, latency_ms, retr_ms, error)
        results.append(m)
        print(f"[{i}/{len(cases)}] {m['qid']} t{m['tier']} {m['cat']:<28} score={m['score']:.2f} "
              f"f1={m['f1']:.2f} rl={m['rl']:.2f} step={m['step']:.1f} hit5={m['hit5']} "
              f"cross={m['cross_count']} lat={latency_ms/1000:.1f}s {('ERR:'+error) if error else ''}", flush=True)

    write_outputs(results, args)


def write_outputs(results, args):
    cfg = args.config
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")

    # ----- eval_retrieval.csv -----
    with open(BASE / "eval_retrieval.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "tier", "category", "question", "retrieved_articles",
                    "retrieved_laws", "retrieved_scores", "top_k", "hit_at_1", "hit_at_3",
                    "hit_at_5", "mrr", "ndcg_at_5", "cross_law_coverage", "latency_ms", "error"])
        for m in results:
            w.writerow([m["qid"], m["tier"], m["cat"], m["question"],
                        ";".join(m["retr_arts"]), ";".join(m["retr_laws"]), "N/A", TOP_K,
                        m["hit1"], m["hit3"], m["hit5"], f"{m['mrr']:.4f}", f"{m['ndcg5']:.4f}",
                        m["cross_count"], f"{m['retr_ms']:.0f}", m["error"]])

    # ----- eval_generation.csv -----
    with open(BASE / "eval_generation.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "tier", "category", "question", "answer", "ground_truth",
                    "cited_articles", "ground_truth_articles", "citation_precision",
                    "citation_recall", "citation_f1", "rouge_l", "bleu", "token_f1",
                    "faithfulness_proxy", "step_completeness", "latency_ms", "error", "note"])
        for m in results:
            w.writerow([m["qid"], m["tier"], m["cat"], m["question"], m["answer"],
                        m["gt"], ";".join(m["cited"]), ";".join(m["gold_arts"]),
                        f"{m['precision']:.4f}", f"{m['recall']:.4f}", f"{m['f1']:.4f}",
                        f"{m['rl']:.4f}", f"{m['bl']:.4f}", f"{m['tf1']:.4f}",
                        m["faithfulness"], f"{m['step']:.4f}", f"{m['latency_ms']:.0f}",
                        m["error"], m["notes"]])

    # ----- eval_tier_scores.csv -----
    tiers = {}
    for m in results:
        tiers.setdefault(m["tier"], []).append(m)
    with open(BASE / "eval_tier_scores.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tier", "avg_score", "max_score", "pass_threshold", "status", "common_errors", "note"])
        for t in sorted(tiers, key=int):
            ms = tiers[t]
            avg = sum(m["score"] for m in ms) / len(ms)
            mx = max(m["score"] for m in ms)
            errs = [classify_error(m, is_refusal_text(m["answer"])) for m in ms]
            errs = [e for e in errs if e]
            common = ";".join(f"{k}:{v}" for k, v in Counter(errs).most_common(3)) or "none"
            w.writerow([t, f"{avg:.4f}", f"{mx:.4f}", 6.0,
                        "PASS" if avg >= 6.0 else "FAIL", common,
                        "score = avg(citation_f1*5 + rouge_l*3 + step_completeness*2), gom ca case refuse"])

    # ----- eval_refusal.csv -----
    with open(BASE / "eval_refusal.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "tier", "category", "question", "expected_behavior",
                    "actual_behavior", "answer", "is_correct_refusal", "reason"])
        for m in results:
            if m["expected"] != "refuse":
                continue
            if m["error"]:
                beh, ok, reason = "timeout", 0, "timeout"
            elif is_refusal_text(m["answer"]):
                beh, ok, reason = "refuse", 1, "co cum tu tu choi"
            else:
                beh, ok, reason = "answer", 0, "tra loi thay vi tu choi"
            w.writerow([m["qid"], m["tier"], m["cat"], m["question"], "refuse",
                        beh, m["answer"], ok, reason])

    # ----- error_log.csv (append) -----
    elog = BASE / "error_log.csv"
    if not elog.exists():
        with open(elog, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["timestamp", "round", "stage", "query_id", "error_type", "message", "fix_applied"])
    with open(elog, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for m in results:
            if m["error"] == "timeout":
                w.writerow([stamp, cfg, "generation", m["qid"], "timeout",
                            f"vuot {CASE_TIMEOUT_S}s", "ghi nhan, khong retry trong run nay"])
            elif m["error"]:
                w.writerow([stamp, cfg, "generation", m["qid"], "exception", m["error"], ""])

    # ----- regression_tests.csv -----
    regr_path = BASE / "regression_tests.csv"
    regr_cases = {"TC001": "T1 di chuc hop phap (Dieu 630)",
                  "TC002": "T2 thu tuc cong chung UBND xa (58;630;635)",
                  "TC003": "T3 nguoi thua ke khong phu thuoc di chuc (644)",
                  "TC004": "T4 cross-law BLDS 630/635 + LC 58/61",
                  "TC005": "T5 cong bo di chuc + 644 (647;644)"}
    prev = {}
    if regr_path.exists():
        for r in csv.DictReader(open(regr_path, encoding="utf-8")):
            if r.get("round") == "baseline":
                prev[r["test_name"]] = float(r["before"])
    scores = {m["qid"]: m["score"] for m in results}
    qids = {"TC001": "TC1-00", "TC002": "TC2-16", "TC003": "TC3-32", "TC004": "TC4-48", "TC005": "TC5-63"}
    with open(regr_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "test_name", "metric", "before", "after", "delta", "status", "note"])
        for name, note in regr_cases.items():
            qid = qids[name]
            if qid not in scores:
                continue
            after = scores[qid]
            before = prev.get(name, after)
            delta = after - before
            status = "baseline" if cfg == "baseline" else ("FIXED" if after >= 6.0 else "STILL_FAIL")
            w.writerow([cfg, name, "case_score", f"{before:.4f}", f"{after:.4f}",
                        f"{delta:+.4f}", status, note])

    # ----- model_comparison.csv (upsert) -----
    mc_path = BASE / "model_comparison.csv"
    cols = ["config_id", "model_llm", "top_k", "use_cross_law", "use_citation_enforce",
            "use_procedure_prompt", "tier1_avg", "tier2_avg", "tier3_avg", "tier4_avg",
            "tier5_avg", "citation_f1", "cross_law_coverage_avg", "refusal_accuracy",
            "avg_latency_ms", "note"]
    rows_mc = {}
    if mc_path.exists():
        for r in csv.DictReader(open(mc_path, encoding="utf-8")):
            rows_mc[r["config_id"]] = r
    tier_avg = {}
    for t in sorted(tiers, key=int):
        ms = tiers[t]
        tier_avg[f"tier{t}_avg"] = f"{sum(m['score'] for m in ms) / len(ms):.4f}"

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    f1s = [m["f1"] for m in results if m["gold_arts"]]
    cross_vals = [m["cross_count"] / 2 for m in results if m["cross_count"] != ""]
    refuse_ms = [m for m in results if m["expected"] == "refuse"]
    refuse_ok = sum(1 for m in refuse_ms if not m["error"] and is_refusal_text(m["answer"]))
    lats = [m["latency_ms"] for m in results]

    row = {
        "config_id": cfg,
        "model_llm": "qwen2.5-3b",
        "top_k": TOP_K,
        "use_cross_law": "no" if cfg == "baseline" else "yes",
        "use_citation_enforce": "no" if cfg == "baseline" else "yes",
        "use_procedure_prompt": "no" if cfg == "baseline" else "yes",
        "tier1_avg": tier_avg.get("tier1_avg", ""), "tier2_avg": tier_avg.get("tier2_avg", ""),
        "tier3_avg": tier_avg.get("tier3_avg", ""), "tier4_avg": tier_avg.get("tier4_avg", ""),
        "tier5_avg": tier_avg.get("tier5_avg", ""),
        "citation_f1": f"{avg(f1s):.4f}",
        "cross_law_coverage_avg": f"{avg(cross_vals):.4f}",
        "refusal_accuracy": f"{refuse_ok / len(refuse_ms):.4f}" if refuse_ms else "",
        "avg_latency_ms": f"{avg(lats):.0f}",
        "note": ("app.py nguyen ban (bak_round0), max_tokens=512, khong guardrail" if cfg == "baseline"
                 else "E3 cross-law + E1 enforce + E2 rerank + E4 procedure prompt + guardrail, max_tokens=256"),
    }
    rows_mc[cfg] = row
    with open(mc_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for c in ["baseline", "improved"]:
            if c in rows_mc:
                w.writerow(rows_mc[c])

    # ----- failure_cases.csv -----
    with open(BASE / "failure_cases.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "tier", "category", "question", "predicted_answer",
                    "cited_articles", "ground_truth_articles", "predicted_score",
                    "error_type", "severity", "note"])
        for m in results:
            is_ok = (m["score"] >= 6.0) and (m["expected"] != "refuse" or is_refusal_text(m["answer"]))
            if is_ok and not m["error"]:
                continue
            et = classify_error(m, is_refusal_text(m["answer"]) if m["expected"] == "refuse" else True)
            sev = "high" if m["score"] < 3 else ("medium" if m["score"] < 6 else "low")
            w.writerow([m["qid"], m["tier"], m["cat"], m["question"],
                        (m["answer"] or "")[:200], ";".join(m["cited"]), ";".join(m["gold_arts"]),
                        f"{m['score']:.4f}", et, sev, m["error"] or m["notes"]])

    # ----- eval_summary.csv -----
    def status_of(value, target, higher_better):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "FAIL"
        return "PASS" if ((v >= target) if higher_better else (v <= target)) else "FAIL"

    in_scope = [m for m in results if m["gold_arts"]]
    hits5 = [m["hit5"] for m in in_scope]
    mrrs = [m["mrr"] for m in in_scope]
    ndcgs = [m["ndcg5"] for m in in_scope]
    recalls = [m["recall"] for m in in_scope]
    precs = [m["precision"] for m in in_scope]
    f1s = [m["f1"] for m in in_scope]
    rls = [m["rl"] for m in results]
    bls = [m["bl"] for m in results]
    tf1s = [m["tf1"] for m in results]
    faiths = [m["faithfulness"] for m in results]
    steps = [m["step"] for m in results if "PROCEDURE" in m["notes"]]
    scores = [m["score"] for m in results]
    hallu = [1 for m in results if m["faithfulness"] == 0]
    p95 = sorted(lats)[int(math.ceil(0.95 * len(lats))) - 1] if lats else 0.0

    summary = [
        ("retrieval_hit_at_5", f"{avg(hits5):.4f}", "ratio", 0.80, True, "hit@5 trung binh tren case in-scope (top-8 retrieved)"),
        ("retrieval_mrr", f"{avg(mrrs):.4f}", "ratio", 0.60, True, "mean reciprocal rank cua article vang trong top-8"),
        ("ndcg_at_5", f"{avg(ndcgs):.4f}", "ratio", 0.65, True, "NDCG binary gain top-5 retrieved"),
        ("cross_law_coverage_avg", f"{avg(cross_vals):.4f}", "ratio", 0.85, True, "so luat vang co mat trong retrieved / 2, trung binh tren case cross-law"),
        ("citation_precision", f"{avg(precs):.4f}", "ratio", 0.75, True, "cham E2: |cited giao gold| / |cited|"),
        ("citation_recall", f"{avg(recalls):.4f}", "ratio", 0.80, True, "cham E1: |cited giao gold| / |gold|"),
        ("citation_f1", f"{avg(f1s):.4f}", "ratio", 0.75, True, "F1 cua precision/recall citation"),
        ("rouge_l", f"{avg(rls):.4f}", "score", 0.30, True, "ROUGE-L F-measure (native LCS, token hoa tieng Viet)"),
        ("bleu", f"{avg(bls):.4f}", "score", 0.15, True, "BLEU 1-4 gram + brevity penalty"),
        ("token_f1", f"{avg(tf1s):.4f}", "score", 0.40, True, "F1 overlap token voi ground truth"),
        ("faithfulness_proxy", f"{avg(faiths):.4f}", "ratio", 0.80, True, "1 neu moi article cite nam trong retrieved docs"),
        ("step_completeness", f"{avg(steps):.4f}", "ratio", 0.80, True, "cham E4: so buoc danh so trong answer / so buoc gold (case PROCEDURE)"),
        ("tier1_avg", tier_avg.get("tier1_avg", ""), "score/10", 6.0, True, "diem trung binh tang 1"),
        ("tier2_avg", tier_avg.get("tier2_avg", ""), "score/10", 6.0, True, "diem trung binh tang 2"),
        ("tier3_avg", tier_avg.get("tier3_avg", ""), "score/10", 6.0, True, "diem trung binh tang 3"),
        ("tier4_avg", tier_avg.get("tier4_avg", ""), "score/10", 6.0, True, "diem trung binh tang 4"),
        ("tier5_avg", tier_avg.get("tier5_avg", ""), "score/10", 6.0, True, "diem trung binh tang 5"),
        ("refusal_accuracy", f"{refuse_ok / len(refuse_ms):.4f}" if refuse_ms else "", "ratio", 0.85, True, "so case tu choi dung / tong case refuse"),
        ("hallucination_rate", f"{len(hallu) / len(results):.4f}", "ratio", 0.15, False, "case cite article ngoai retrieved / tong case (validator chan bia so dieu)"),
        ("avg_latency_ms", f"{avg(lats):.0f}", "ms", 15000, False, "trung binh tong latency moi case"),
        ("p95_latency_ms", f"{p95:.0f}", "ms", 25000, False, "percentile 95 tong latency"),
        ("total_test_cases", len(results), "count", 80, True, "so case trong run nay"),
        ("pass_rate", f"{sum(1 for s in scores if s >= 6.0) / len(scores):.4f}", "ratio", 0.70, True, "so case co case_score >= 6.0 / tong case"),
    ]
    with open(BASE / "eval_summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric_name", "value", "unit", "target", "status", "note"])
        for name, val, unit, tgt, higher, note in summary:
            w.writerow([name, val, unit, tgt, status_of(val, tgt, higher), note])

    # ----- in ket qua -----
    print("\n[run_eval] ===== TOM TAT =====", flush=True)
    for t in sorted(tiers, key=int):
        ms = tiers[t]
        avg_t = sum(m["score"] for m in ms) / len(ms)
        print(f"  tier{t}: avg={avg_t:.3f} ({'PASS' if avg_t >= 6.0 else 'FAIL'})", flush=True)
    print(f"  citation_f1={avg(f1s):.3f}  cross_law_coverage_avg={avg(cross_vals):.3f}  "
          f"refusal_accuracy={refuse_ok / len(refuse_ms) if refuse_ms else 0:.3f}  "
          f"pass_rate={sum(1 for s in scores if s >= 6.0) / len(scores):.3f}", flush=True)


if __name__ == "__main__":
    main()
