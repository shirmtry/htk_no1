# -*- coding: utf-8 -*-
"""Vẽ sơ đồ pipeline RAG của Chatbot Pháp luật VN v2.0 bằng Graphviz."""
from graphviz import Digraph

FONT = "DejaVu Sans"
g = Digraph("pipeline", format="png")
g.attr(newrank="true")
g.attr(rankdir="TB", dpi="200", bgcolor="white", fontname=FONT,
       nodesep="0.28", ranksep="0.34", pad="0.12")
g.attr("node", shape="box", style="rounded,filled", fontname=FONT,
       fontsize="11", margin="0.14,0.06", penwidth="1.2", color="#334155")
g.attr("edge", fontname=FONT, fontsize="9", color="#475569", arrowsize="0.8")


def node(name, label, fill, fontcolor="#0f172a"):
    g.node(name, label=label, fillcolor=fill, fontcolor=fontcolor)


node("ui", "Giao diện Streamlit\nnhập câu hỏi · hiển thị nguồn", "#1e293b", "#ffffff")
node("guard", "0. GUARDRAIL\nshould_refuse(q)\nngoài phạm vi · injection", "#fde68a")
node("refuse", "Từ chối\nRU_TEXT / RP_TEXT · ~0 ms", "#fecaca")
node("rw", "1–2. REWRITE + EXPAND\nkhử coreference · alias luật", "#e0e7ff")

with g.subgraph(name="cluster_ret") as c:
    c.attr(label="3. RETRIEVAL — hybrid + cross-law", fontname=FONT, fontsize="11",
           style="rounded", color="#93c5fd", bgcolor="#f0f7ff", margin="10")
    c.node("bm25", "BM25 0.4 · k=15", fillcolor="#dbeafe", fontname=FONT)
    c.node("faiss", "FAISS 0.6 · k=15", fillcolor="#dbeafe", fontname=FONT)
    c.node("ens", "EnsembleRetriever · safe_invoke", fillcolor="#bfdbfe", fontname=FONT)
    c.node("cross", "Cross-law BM25 · 19 alias", fillcolor="#bfdbfe", fontname=FONT)
    c.node("rank", "Rerank · dedupe · truncate", fillcolor="#bfdbfe", fontname=FONT)
    c.edge("bm25", "ens")
    c.edge("faiss", "ens")
    c.edge("ens", "cross")
    c.edge("cross", "rank")

node("rule", "4. RULE LAYER\n55 nhánh deterministic", "#bbf7d0")
node("gen", "5. GENERATE\nQwen2.5-3B Q4_K_M", "#e9d5ff")
node("post", "6. POST-CHECK\nchặn số điều bịa · ép trích dẫn", "#fef08a")
node("answer", "Trả lời + danh sách nguồn", "#065f46", "#ffffff")

g.edge("ui", "guard")
g.edge("guard", "refuse", label="vi phạm")
g.edge("guard", "rw", label="hợp lệ")
g.edge("rw", "bm25")
g.edge("rw", "faiss")
g.edge("rank", "rule", label="ngữ cảnh top-k", constraint="false")
g.edge("rule", "answer", label="khớp rule")
g.edge("rule", "gen", label="không khớp")
g.edge("gen", "post")
g.edge("post", "answer")

with g.subgraph() as s:
    s.attr(rank="same")
    s.node("ui")
    s.node("rule")

g.render("pipeline", cleanup=True)
print("pipeline.png OK")
