

# ---------------------------------------------------------------------------------------------
# Additional baselines (round 18): head / middle truncation, RECOMP extractive, Selective Context.
# All fill the same context slot under the same exact downstream-token budget via _emit.
# ---------------------------------------------------------------------------------------------
import os as _os
_AUX = _os.environ.get("BP_AUX", "/root/autodl-tmp/aux")


def _room(tok, q, budget):
    return budget - len(tok(etp._prefix(q, False)).input_ids) - 4


def head_trunc(s, tok, budget=1024):
    """Keep the beginning of the context."""
    q = s["question"]
    ids = tok(s["context"]).input_ids[:max(0, _room(tok, q, budget))]
    return _emit(tok, q, [tok.decode(ids)], budget)


def mid_trunc(s, tok, budget=1024):
    """LongBench-style middle truncation: keep the first and the last half of the room."""
    q = s["question"]
    room = max(0, _room(tok, q, budget))
    ids = tok(s["context"]).input_ids
    if len(ids) > room:
        h = room // 2
        text = tok.decode(ids[:h]) + "\n\n" + tok.decode(ids[len(ids) - (room - h):])
    else:
        text = tok.decode(ids)
    return _emit(tok, q, [text], budget)


import re as _re
_SPLIT = _re.compile(r"(?<=[.!?])\s+|\n+")


def _sentences(text):
    return [x.strip() for x in _SPLIT.split(text) if x and len(x.strip()) > 1]


_RC = {}


def _recomp_embed(texts):
    import torch
    from transformers import AutoModel, AutoTokenizer
    if not _RC:
        torch.set_num_threads(int(_os.environ.get("RC_THREADS", "1")))
        p = _os.path.join(_AUX, "hotpotqa_extractive_compressor")
        _RC["tok"] = AutoTokenizer.from_pretrained(p); _RC["m"] = AutoModel.from_pretrained(p).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = _RC["tok"](texts[i:i + 64], padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = _RC["m"](**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            out.append((h * m).sum(1) / m.sum(1).clamp(min=1e-9))  # mean pooling (sentence-transformers config)
    return torch.cat(out)


def recomp_ext(s, tok, budget=1024):
    """RECOMP extractive compressor (Xu et al., 2024; HotpotQA checkpoint): sentences ranked by the
    dot product of their embedding with the question embedding, added in rank order while they fit."""
    q = s["question"]
    sents = _sentences(s["context"])
    if not sents:
        return _emit(tok, q, [], budget)
    E = _recomp_embed([q] + sents)
    sc = (E[1:] @ E[0]).tolist()
    order = sorted(range(len(sents)), key=lambda i: -sc[i])
    prefix = etp._prefix(q, False)
    chosen, used = [], len(tok(prefix).input_ids)
    for i in order:
        n = len(tok(sents[i]).input_ids) + 1
        if used + n <= budget:
            chosen.append(sents[i]); used += n
        if used >= budget - 2:
            break
    return _emit(tok, q, [" ".join(chosen)], budget)


_SC = {}


def _sc():
    if not _SC:
        import sys
        sys.path.insert(0, _os.path.join(_AUX, "sc_pkgs"))
        import torch, spacy
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        import selective_context as scm
        torch.set_num_threads(int(_os.environ.get("SC_THREADS", "1")))

        class LocalSC(scm.SelectiveContext):
            def _prepare_phrase_tokenizer(self):
                self.nlp = spacy.load(_os.path.join(_AUX, "en_core_web_sm_hf"), disable=["ner"])
                self.nlp.add_pipe("merge_noun_chunks")

            def _prepare_model(self):
                p = _os.path.join(_AUX, "gpt2")
                self.tokenizer = GPT2TokenizerFast.from_pretrained(p)
                self.model = GPT2LMHeadModel.from_pretrained(p).to(self.device).eval()
                self.max_token_length = self.model.config.n_positions
                self.get_self_information = self._get_self_info_via_gpt2

            def _get_self_info_via_gpt2(self, text):
                # official computation, applied in windows of 1,023 tokens for the rare over-long sentence
                with torch.no_grad():
                    ids = self.tokenizer(text, add_special_tokens=False).input_ids
                    toks, info, W = [], [], 1023
                    for a in range(0, max(1, len(ids)), W):
                        seg = [self.tokenizer.eos_token_id] + ids[a:a + W]
                        x = torch.tensor([seg], device=self.device)
                        lp = torch.log_softmax(self.model(input_ids=x).logits, dim=-1)
                        si = -lp[0, :-1].gather(-1, x[0, 1:].unsqueeze(-1)).squeeze(-1)
                        toks += [self.tokenizer.decode([t]) for t in seg[1:]]; info += si.tolist()
                return toks, info

        sc = LocalSC.__new__(LocalSC)
        sc.model_type, sc.lang, sc.device = "gpt2", "en", torch.device("cpu")
        sc.sent_level_self_info = True
        sc._prepare_phrase_tokenizer()
        sc.sent_tokenize_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s"
        sc.phrase_mask_token, sc.sent_mask_token, sc.keep_leading_word, sc.mask_token = "", "<...some content omitted.>", False, ""
        sc._prepare_model()
        _SC["sc"] = sc
    return _SC["sc"]


def sel_ctx(s, tok, budget=1024):
    """Selective Context (Li et al., 2023), official phrase-level masking with GPT-2 self-information.
    The reduce ratio is set so the kept text fits the room; any overflow is cut at the end by _emit."""
    q = s["question"]
    room = max(1, _room(tok, q, budget))
    n = len(tok(s["context"]).input_ids)
    if n <= room:
        return _emit(tok, q, [s["context"]], budget)
    text, _ = _sc()(s["context"], reduce_ratio=min(0.99, 1 - room / n), reduce_level="phrase")
    return _emit(tok, q, [text], budget)
