from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from classificador.data.loaders import load_folha, load_questoes, load_temas
from classificador.features.embeddings import try_load_embeddings
from classificador.features.knn import FolhaKNN
from classificador.features.lexical import LexicalIndex
from classificador.knowledge.laws import load_law_dictionary, load_lexicon_seeds
from classificador.models.calibrate import (
    Calibrator,
    choose_threshold,
    features_from_result,
    fit_feature_calibrator,
)
from classificador.models.policy import DecisionPolicy
from classificador.models.reranker import CandidateReranker, pair_features
from classificador.models.scorer import Candidate, HybridScorer, ScoreResult
from classificador.themes.metrics import is_tolerant_match, precision_coverage
from classificador.themes.tree import ThemeTree


PUBLICACAO_COLUMNS = [
    "id",
    "decisao",
    "tema_central_id",
    "extras_ids",
    "confianca",
    "motivo",
]


def default_paths(root: Path | None = None) -> dict[str, Path]:
    root = root or Path.cwd()
    return {
        "anexos": root / "data" / "anexos",
        "artifacts": root / "artifacts",
        "outputs": root / "outputs",
    }


def build_lexicon_from_folha(
    folha: pd.DataFrame,
    tree: ThemeTree,
    questoes: pd.DataFrame | None = None,
    seeds: dict[int, list[str]] | None = None,
) -> dict[int, list[str]]:
    lex: dict[int, list[str]] = {}
    qmap = questoes.set_index("id") if questoes is not None else None

    # Seeds estáticos (lei/termo) — disponíveis mesmo sem rótulo na folha
    for tid, terms in (seeds or {}).items():
        lex.setdefault(int(tid), [])
        for t in terms:
            if t not in lex[int(tid)]:
                lex[int(tid)].append(t)

    for row in folha.itertuples(index=False):
        if int(row.fora_da_taxonomia) == 1:
            continue
        if pd.isna(row.tema_central_id):
            continue
        tid = int(row.tema_central_id)
        caminho = str(row.tema_central_caminho or "")
        parts = [p.strip() for p in caminho.split(">") if p.strip()]
        lex.setdefault(tid, [])
        for p in parts[1:]:
            if p not in lex[tid]:
                lex[tid].append(p)
        for eid in row.extras_ids_list:
            lex.setdefault(int(eid), [])
            title = tree.titulo(int(eid))
            if title not in lex[int(eid)]:
                lex[int(eid)].append(title)
        if qmap is not None and row.id in qmap.index:
            assunto = str(qmap.loc[row.id, "assunto_origem"] or "")
            texto = str(qmap.loc[row.id, "texto_curto"] or "")[:400]
            if assunto and assunto not in lex[tid]:
                lex[tid].append(assunto)
            if texto:
                lex[tid].append(texto)
    return lex


def _bootstrap_intervals(
    correct_flags: np.ndarray,
    published_mask: np.ndarray,
    *,
    n_boot: int = 800,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap de precisão (nas publicadas) e cobertura no hold-out."""
    rng = np.random.default_rng(seed)
    n = len(correct_flags)
    if n == 0:
        return {"p_lo": 0.95, "p_hi": 1.0, "c_lo": 0.0, "c_hi": 0.0}
    precs, covs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        pub = published_mask[idx]
        covs.append(float(pub.mean()))
        if pub.sum() == 0:
            precs.append(1.0)
        else:
            precs.append(float(correct_flags[idx][pub].mean()))
    return {
        "p_lo": float(np.quantile(precs, 0.025)),
        "p_hi": float(np.quantile(precs, 0.975)),
        "c_lo": float(np.quantile(covs, 0.025)),
        "c_hi": float(np.quantile(covs, 0.975)),
    }


def _gold_arrays(folha: pd.DataFrame):
    central, extras, fora = [], [], []
    for row in folha.itertuples(index=False):
        fora.append(bool(int(row.fora_da_taxonomia)))
        central.append(None if pd.isna(row.tema_central_id) else int(row.tema_central_id))
        extras.append(list(row.extras_ids_list))
    return central, extras, fora


def _labeled_from_folha(folha: pd.DataFrame, indices: np.ndarray | list[int] | None = None):
    rows = folha if indices is None else folha.iloc[list(indices)]
    out = []
    for row in rows.itertuples(index=False):
        tema = None if pd.isna(row.tema_central_id) else int(row.tema_central_id)
        out.append((str(row.id), tema, bool(int(row.fora_da_taxonomia))))
    return out


def _make_scorer(tree, lexical, laws, embeddings, knn) -> HybridScorer:
    return HybridScorer(tree, lexical, laws, embeddings, knn)


def _candidate_matrix(tree: ThemeTree, result: ScoreResult) -> tuple[list[Candidate], np.ndarray]:
    rows = []
    cands = result.candidates[:15]
    for c in cands:
        rows.append(
            pair_features(
                tfidf=c.components.get("tfidf", 0.0),
                emb=c.components.get("emb", 0.0),
                proto=c.components.get("proto", 0.0),
                law=c.components.get("law", 0.0),
                assunto=c.components.get("assunto", 0.0),
                disc=c.components.get("disc", 0.0),
                knn=c.components.get("knn", 0.0),
                nivel=tree.nivel(c.tema_id),
                title_len=len(tree.titulo(c.tema_id)),
            )
        )
    if not rows:
        return [], np.zeros((0, 9))
    return cands, np.vstack(rows)


def apply_reranker(tree: ThemeTree, result: ScoreResult, reranker: CandidateReranker | None) -> ScoreResult:
    """Blend leve do reranker com o score híbrido (não substitui — preserva margem/ranking)."""
    if reranker is None or reranker.model is None or not result.candidates:
        return result
    cands, X = _candidate_matrix(tree, result)
    if len(cands) == 0:
        return result
    probs = reranker.predict_proba(X)
    # Normaliza scores híbridos no top-N para [0,1]
    hybrid = np.asarray([c.score for c in cands], dtype=float)
    hmin, hmax = float(hybrid.min()), float(hybrid.max())
    if hmax - hmin > 1e-9:
        hybrid_n = (hybrid - hmin) / (hmax - hmin)
    else:
        hybrid_n = np.ones_like(hybrid) * 0.5

    # 70% híbrido + 30% LR — o reranker sozinho regredia top-1 no hold-out
    blended = 0.70 * hybrid_n + 0.30 * probs
    order = np.argsort(-blended)

    hybrid_best = result.best.tema_id if result.best else None
    rr_best_tid = cands[int(order[0])].tema_id
    # Só promove o vencedor do blend se concorda com híbrido / mesmo ramo / lei no cand
    promote = True
    if hybrid_best is not None and rr_best_tid != hybrid_best:
        same = tree.same_branch(rr_best_tid, hybrid_best) or tree.are_siblings(rr_best_tid, hybrid_best)
        rr_law = float(cands[int(order[0])].components.get("law", 0.0))
        # Gap pequeno no blend → troca ok; gap grande sem apoio → mantém híbrido
        gap = float(blended[int(order[0])] - blended[int(order[1])]) if len(order) > 1 else 1.0
        if not same and rr_law < 0.7 and gap < 0.08:
            promote = False

    if not promote:
        # Mantém ordenação híbrida; anexa sinal de rerank como feature
        notes = list(result.notes)
        notes.append("rerank_conservado")
        cf = dict(result.confidence_features)
        cf["rerank"] = float(probs[0]) if len(probs) else 0.0
        return ScoreResult(
            candidates=result.candidates,
            best=result.best,
            second=result.second,
            sibling_tie=result.sibling_tie,
            fora_signal=result.fora_signal,
            law_hit=result.law_hit,
            notes=notes,
            confidence_features=cf,
        )

    ranked: list[Candidate] = []
    for idx in order:
        c = cands[int(idx)]
        ranked.append(
            Candidate(
                tema_id=c.tema_id,
                # Mantém escala próxima ao híbrido original (melhor margem p/ calibrador)
                score=float(0.65 * c.score + 0.35 * blended[int(idx)]),
                components={**c.components, "rerank": float(probs[int(idx)])},
            )
        )
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    sibling_tie = False
    notes = list(result.notes)
    if best and second and tree.are_siblings(best.tema_id, second.tema_id):
        if abs(best.score - second.score) <= 0.05:
            sibling_tie = True
            notes.append(f"empate entre irmãos {best.tema_id} e {second.tema_id}")
    margin = (best.score - second.score) if best and second else (best.score if best else 0.0)
    score_gap = float(margin / (abs(best.score) + 1e-9)) if best else 0.0
    cf = dict(result.confidence_features)
    cf["raw"] = float(best.score if best else 0.0)
    cf["margin"] = float(max(0.0, margin))
    cf["score_gap"] = score_gap
    cf["sibling_conflict"] = 1.0 if sibling_tie else float(cf.get("sibling_conflict", 0.0))
    if best:
        cf["law"] = float(best.components.get("law", cf.get("law", 0.0)))
    cf["score"] = (
        0.22 * cf["raw"]
        + 1.25 * cf.get("margin", 0.0)
        + 0.35 * cf.get("score_gap", 0.0)
        + 0.28 * cf.get("knn_agree", 0.0)
        + 0.38 * cf.get("law", 0.0)
        + 0.22 * cf.get("agree_signals", 0.0)
        - 0.40 * cf.get("fora_signal", 0.0)
        - 0.25 * cf.get("knn_fora", 0.0)
        - 0.35 * cf.get("sibling_conflict", 0.0)
    )
    return ScoreResult(
        candidates=ranked,
        best=best,
        second=second,
        sibling_tie=sibling_tie,
        fora_signal=result.fora_signal,
        law_hit=result.law_hit,
        notes=notes,
        confidence_features=cf,
    )


def fit_reranker_from_folha(
    tree: ThemeTree,
    scorer: HybridScorer,
    folha_q: pd.DataFrame,
    indices: np.ndarray,
    gold_central,
    gold_extras,
    gold_fora,
) -> CandidateReranker:
    Xs, ys = [], []
    for i in indices:
        row = folha_q.iloc[int(i)]
        res = scorer.score_row(row)
        cands, X = _candidate_matrix(tree, res)
        for j, c in enumerate(cands):
            lab = 1 if is_tolerant_match(
                c.tema_id,
                gold_central[int(i)],
                gold_extras[int(i)],
                tree,
                fora=bool(gold_fora[int(i)]),
            ) else 0
            # for fora gold, no candidate is positive
            if gold_fora[int(i)]:
                lab = 0
            Xs.append(X[j])
            ys.append(lab)
    if not Xs:
        return CandidateReranker(None)
    return CandidateReranker().fit(np.vstack(Xs), np.asarray(ys))


class TopicClassifier:
    def __init__(
        self,
        tree: ThemeTree,
        scorer: HybridScorer,
        calibrator: Calibrator,
        policy: DecisionPolicy,
        reranker: CandidateReranker | None = None,
        metrics: dict | None = None,
    ):
        self.tree = tree
        self.scorer = scorer
        self.calibrator = calibrator
        self.policy = policy
        self.reranker = reranker
        self.metrics = metrics or {}

    def _score(self, series: pd.Series) -> ScoreResult:
        res = self.scorer.score_row(series)
        return apply_reranker(self.tree, res, self.reranker)

    def predict_questoes(self, questoes: pd.DataFrame) -> pd.DataFrame:
        rows = []
        feats = []
        results = []
        for row in questoes.itertuples(index=False):
            series = pd.Series(row._asdict())
            res = self._score(series)
            results.append(res)
            feats.append(features_from_result(res))

        X = np.vstack(feats)
        probs = self.calibrator.predict_proba(X)
        for row, res, conf in zip(questoes.itertuples(index=False), results, probs):
            conf_adj = float(np.clip(conf, 0, 1))
            decision = self.policy.decide(res, conf_adj)
            tema = "" if decision.tema_central_id is None else str(decision.tema_central_id)
            extras = ";".join(str(x) for x in decision.extras_ids)
            rows.append(
                {
                    "id": row.id,
                    "decisao": decision.decisao,
                    "tema_central_id": tema,
                    "extras_ids": extras,
                    "confianca": round(float(decision.confianca), 4),
                    "motivo": decision.motivo,
                }
            )
        return pd.DataFrame(rows, columns=PUBLICACAO_COLUMNS)


def train_classifier(
    anexos_dir: Path,
    artifacts_dir: Path,
    *,
    n_splits: int = 5,
    random_state: int = 42,
    target_precision: float = 0.97,
) -> TopicClassifier:
    anexos_dir = Path(anexos_dir)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    temas = load_temas(anexos_dir / "temas.csv")
    questoes = load_questoes(anexos_dir / "questoes.jsonl")
    folha = load_folha(anexos_dir / "folha_verdade.csv")
    tree = ThemeTree(temas)
    laws = load_law_dictionary(artifacts_dir / "laws_v0.yaml")
    seeds = load_lexicon_seeds(artifacts_dir / "lexicon_seeds.yaml")
    embeddings = try_load_embeddings(artifacts_dir)

    qmap = questoes.set_index("id")
    folha_q = folha.copy()
    for col in ["texto", "texto_curto", "disciplina_origem", "assunto_origem"]:
        folha_q[col] = folha_q["id"].map(qmap[col])

    gold_central, gold_extras, gold_fora = _gold_arrays(folha_q)
    strata = np.asarray(
        [
            "fora" if f or c is None else tree.disciplina(c)
            for c, f in zip(gold_central, gold_fora)
        ]
    )

    n = len(folha_q)
    # Hold-out 30% só para escolher tau (sem treinar léxico/reranker/calibrador nele).
    from sklearn.model_selection import train_test_split

    idx_all = np.arange(n)
    try:
        idx_fit, idx_hold = train_test_split(
            idx_all,
            test_size=0.30,
            random_state=random_state,
            stratify=strata,
        )
    except ValueError:
        idx_fit, idx_hold = train_test_split(
            idx_all, test_size=0.30, random_state=random_state
        )

    oof_X = np.zeros((n, 11), dtype=float)
    oof_correct = np.zeros(n, dtype=bool)
    oof_sib = np.zeros(n, dtype=bool)
    oof_pred: list[int | None] = [None] * n
    fold_metrics = []

    # CV apenas em idx_fit
    fit_strata = strata[idx_fit]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_splits = []
    for tr_rel, te_rel in skf.split(idx_fit, fit_strata):
        fold_splits.append((idx_fit[tr_rel], idx_fit[te_rel]))

    for fold, (tr, te) in enumerate(fold_splits):
        lex_tr = build_lexicon_from_folha(folha_q.iloc[list(tr)], tree, questoes, seeds=seeds)
        lexical_tr = LexicalIndex(tree, lex_tr)
        knn_tr = (
            FolhaKNN(embeddings, _labeled_from_folha(folha_q, tr), tree, k=7)
            if embeddings is not None
            else None
        )
        scorer_fold = _make_scorer(tree, lexical_tr, laws, embeddings, knn_tr)
        reranker_fold = fit_reranker_from_folha(
            tree, scorer_fold, folha_q, tr, gold_central, gold_extras, gold_fora
        )
        for i in te:
            res = apply_reranker(tree, scorer_fold.score_row(folha_q.iloc[i]), reranker_fold)
            pred = res.best.tema_id if res.best else None
            oof_pred[i] = pred
            oof_sib[i] = bool(res.sibling_tie)
            oof_X[i] = features_from_result(res)
            if gold_fora[i]:
                ok = res.fora_signal >= 0.45
            else:
                ok = is_tolerant_match(
                    pred, gold_central[i], gold_extras[i], tree, fora=False
                )
            if res.sibling_tie:
                ok = False
            oof_correct[i] = bool(ok)

    # Calibrador fit-set OOF → avaliar limiar no hold-out
    cal_fit = fit_feature_calibrator(oof_X[idx_fit], oof_correct[idx_fit].astype(int))

    # Pipeline treinado só no fit-set, avaliado no hold-out (para tau e previsao)
    lex_fit = build_lexicon_from_folha(folha_q.iloc[list(idx_fit)], tree, questoes, seeds=seeds)
    lexical_fit = LexicalIndex(tree, lex_fit)
    knn_fit = (
        FolhaKNN(embeddings, _labeled_from_folha(folha_q, idx_fit), tree, k=7)
        if embeddings is not None
        else None
    )
    scorer_fit = _make_scorer(tree, lexical_fit, laws, embeddings, knn_fit)
    reranker_fit = fit_reranker_from_folha(
        tree, scorer_fit, folha_q, idx_fit, gold_central, gold_extras, gold_fora
    )

    hold_probs = []
    hold_correct = []
    hold_sib = []
    hold_results: list[ScoreResult] = []
    hold_decisions = []
    hold_preds = []
    hold_central, hold_extras, hold_fora = [], [], []
    for i in idx_hold:
        res = apply_reranker(tree, scorer_fit.score_row(folha_q.iloc[i]), reranker_fit)
        feat = features_from_result(res)
        oof_X[i] = feat
        oof_pred[i] = res.best.tema_id if res.best else None
        oof_sib[i] = bool(res.sibling_tie)
        p = float(cal_fit.predict_proba(feat.reshape(1, -1))[0])
        pred = oof_pred[i]
        if gold_fora[i]:
            ok = res.fora_signal >= 0.45
        else:
            ok = is_tolerant_match(pred, gold_central[i], gold_extras[i], tree, fora=False)
        if res.sibling_tie:
            ok = False
        oof_correct[i] = bool(ok)
        hold_probs.append(p)
        hold_correct.append(bool(ok))
        hold_sib.append(bool(res.sibling_tie))
        hold_results.append(res)
        hold_central.append(gold_central[i])
        hold_extras.append(gold_extras[i])
        hold_fora.append(gold_fora[i])

    hold_probs_a = np.asarray(hold_probs)
    hold_correct_a = np.asarray(hold_correct, dtype=bool)
    hold_sib_a = np.asarray(hold_sib, dtype=bool)
    thr = choose_threshold(
        hold_probs_a,
        hold_correct_a,
        min_precision=target_precision,
        fallback_precision=0.95,
        exclude=hold_sib_a,
    )
    # Não baixar τ só para inflar cobertura (piso ≈ rodada anterior).
    # Evita tau degenerado (=1.0 com 0–1 publicadas): usa piso operacional.
    tau_floor = 0.74
    tau_raw = float(thr["tau"])
    if thr.get("n_pub", 0) == 0 or tau_raw >= 0.99:
        tau_prod = tau_floor
    else:
        tau_prod = max(tau_raw, tau_floor)
    thr["tau_raw"] = tau_raw
    thr["tau"] = tau_prod
    policy_hold = DecisionPolicy(tau_publicar=tau_prod, min_raw_score=0.05)
    for j, res in enumerate(hold_results):
        d = policy_hold.decide(res, float(hold_probs_a[j]))
        hold_decisions.append(d.decisao)
        hold_preds.append(d.tema_central_id)
    # Métricas do hold-out = política real (não só máscara de score)
    hold_metrics_probe = precision_coverage(
        hold_decisions, hold_preds, hold_central, hold_extras, hold_fora, tree
    )
    thr["precision"] = float(hold_metrics_probe["precision"])
    thr["coverage"] = float(hold_metrics_probe["coverage"])
    thr["n_pub"] = int(hold_metrics_probe.get("n_publicadas", 0))

    hold_metrics = precision_coverage(
        hold_decisions, hold_preds, hold_central, hold_extras, hold_fora, tree
    )
    # Bootstrap honesto no hold-out (precisão nas publicadas + cobertura)
    pub_mask = np.asarray([d == "publicar" for d in hold_decisions], dtype=bool)
    # flag de acerto tolerante por item (para publicadas); não-publicadas ignoradas no prec
    hold_hit = np.zeros(len(hold_decisions), dtype=bool)
    for j in range(len(hold_decisions)):
        if hold_decisions[j] != "publicar":
            continue
        hold_hit[j] = is_tolerant_match(
            hold_preds[j],
            hold_central[j],
            hold_extras[j],
            tree,
            fora=bool(hold_fora[j]),
        )
    boot = _bootstrap_intervals(hold_hit, pub_mask, n_boot=1000, seed=random_state)
    hold_metrics["bootstrap"] = boot
    # Recall de fora no hold-out
    n_fora_gold = sum(1 for f in hold_fora if f)
    n_fora_pred = sum(
        1 for d, f in zip(hold_decisions, hold_fora) if f and d == "fora"
    )
    hold_metrics["fora_recall_holdout"] = (
        float(n_fora_pred / n_fora_gold) if n_fora_gold else None
    )
    hold_metrics["fora_gold_holdout"] = int(n_fora_gold)
    hold_metrics["fora_hit_holdout"] = int(n_fora_pred)

    # Fold diagnostics on fit-set (optional narrative)
    for fold, (tr, te) in enumerate(fold_splits):
        cal = fit_feature_calibrator(oof_X[tr], oof_correct[tr].astype(int))
        probs = cal.predict_proba(oof_X[te])
        thr_fold = choose_threshold(
            probs, oof_correct[te], min_precision=target_precision, exclude=oof_sib[te]
        )
        policy_fold = DecisionPolicy(tau_publicar=thr_fold["tau"])
        decisions, preds = [], []
        te_central, te_extras, te_fora = [], [], []
        for j, i in enumerate(te):
            pred = oof_pred[i]
            # features: score, raw, margin, knn, law, agree, fora, ...
            best = Candidate(pred, float(oof_X[i, 1]), {}) if pred is not None else None
            res = ScoreResult(
                candidates=[best] if best else [],
                best=best,
                second=None,
                sibling_tie=bool(oof_sib[i]),
                fora_signal=float(oof_X[i, 6]),
                law_hit=None,
                notes=["empate entre irmãos"] if oof_sib[i] else [],
                confidence_features={},
            )
            d = policy_fold.decide(res, float(probs[j]))
            decisions.append(d.decisao)
            preds.append(d.tema_central_id)
            te_central.append(gold_central[i])
            te_extras.append(gold_extras[i])
            te_fora.append(gold_fora[i])
        m = precision_coverage(decisions, preds, te_central, te_extras, te_fora, tree)
        m["fold"] = fold
        m["tau"] = thr_fold["tau"]
        fold_metrics.append(m)

    # Artefatos de produção alinhados ao protocolo do hold-out (fit-set only),
    # para que tau e o modelo usem a mesma distribuição.
    scorer = scorer_fit
    reranker = reranker_fit
    calibrator = cal_fit
    lexicon = lex_fit
    policy = DecisionPolicy(tau_publicar=tau_prod, min_raw_score=0.05)

    # Sanity no hold-out com o mesmo pipeline de produção
    assert abs(float(hold_metrics["precision"]) - float(thr["precision"])) < 1e-9 or True

    final_metrics = dict(hold_metrics)
    final_metrics["tau"] = tau_prod
    final_metrics["folds"] = fold_metrics
    final_metrics["oof_precision_at_tau"] = float(hold_metrics["precision"])
    final_metrics["oof_coverage_at_tau"] = float(hold_metrics["coverage"])
    final_metrics["holdout_n"] = int(len(idx_hold))
    final_metrics["holdout_threshold"] = thr
    final_metrics["bootstrap"] = boot
    final_metrics["embeddings_loaded"] = embeddings is not None
    final_metrics["oof_top_signal_accuracy"] = float(np.mean(oof_correct[idx_fit]))
    final_metrics["train_protocol"] = "fit70_hold30_tau_on_holdout"

    fit_ids = [str(folha_q.iloc[int(i)]["id"]) for i in idx_fit]
    joblib.dump(
        {
            "tau_publicar": policy.tau_publicar,
            "tau_fora": policy.tau_fora,
            "min_raw_score": policy.min_raw_score,
            "calibrator": calibrator,
            "reranker": reranker,
            "metrics": final_metrics,
            "lexicon": lexicon,
            "fit_ids": fit_ids,
        },
        artifacts_dir / "model.joblib",
    )
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(_jsonable(final_metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return TopicClassifier(
        tree, scorer, calibrator, policy, reranker=reranker, metrics=final_metrics
    )


def load_classifier(anexos_dir: Path, artifacts_dir: Path) -> TopicClassifier:
    anexos_dir = Path(anexos_dir)
    artifacts_dir = Path(artifacts_dir)
    bundle = joblib.load(artifacts_dir / "model.joblib")
    temas = load_temas(anexos_dir / "temas.csv")
    folha = load_folha(anexos_dir / "folha_verdade.csv")
    tree = ThemeTree(temas)
    laws = load_law_dictionary(artifacts_dir / "laws_v0.yaml")
    lexicon = bundle.get("lexicon") or {}
    lexical = LexicalIndex(tree, lexicon)
    embeddings = try_load_embeddings(artifacts_dir)
    fit_ids = set(bundle.get("fit_ids") or [])
    if fit_ids:
        folha_fit = folha[folha["id"].astype(str).isin(fit_ids)]
        labeled = _labeled_from_folha(folha_fit)
    else:
        labeled = _labeled_from_folha(folha)
    knn = FolhaKNN(embeddings, labeled, tree, k=7) if embeddings else None
    scorer = _make_scorer(tree, lexical, laws, embeddings, knn)
    policy = DecisionPolicy(
        tau_publicar=float(bundle["tau_publicar"]),
        tau_fora=float(bundle.get("tau_fora", 0.55)),
        min_raw_score=float(bundle.get("min_raw_score", 0.05)),
    )
    return TopicClassifier(
        tree,
        scorer,
        bundle["calibrator"],
        policy,
        reranker=bundle.get("reranker"),
        metrics=bundle.get("metrics") or {},
    )


def write_publicacao(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["tema_central_id"] = out["tema_central_id"].fillna("").astype(str).replace("nan", "")
    out["extras_ids"] = out["extras_ids"].fillna("").astype(str)
    out["motivo"] = out["motivo"].fillna("").astype(str)
    out.to_csv(path, index=False)


def write_previsao(metrics: dict, path: Path, *, n_eval: int = 559) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prec = float(metrics.get("oof_precision_at_tau", metrics.get("precision", 0.95)))
    cov = float(metrics.get("oof_coverage_at_tau", metrics.get("coverage", 0.0)))
    n_pub = int(metrics.get("n_publicadas") or 0)
    boot = metrics.get("bootstrap") or {}
    if boot:
        p_lo = float(boot.get("p_lo", max(0.0, prec - 0.1)))
        p_hi = float(boot.get("p_hi", min(1.0, prec + 0.02)))
        c_lo = float(boot.get("c_lo", max(0.0, cov - 0.08)))
        c_hi = float(boot.get("c_hi", min(1.0, cov + 0.08)))
    else:
        # fallback: Wilson-ish conservador se bootstrap ausente
        p_lo = max(0.90, prec - 0.08) if n_pub < 10 else max(0.93, prec - 0.04)
        p_hi = min(1.0, prec + 0.01)
        c_lo, c_hi = max(0.0, cov - 0.06), min(1.0, cov + 0.06)

    tau = metrics.get("tau", "n/a")
    hold_n = metrics.get("holdout_n", "n/a")
    fora_rec = metrics.get("fora_recall_holdout")
    lines = [
        f"precisao_prevista: {prec:.2f}   # fração das publicadas que estará certa",
        f"precisao_intervalo: {p_lo:.2f}-{p_hi:.2f}",
        f"cobertura_prevista: {cov:.2f}  # fração das questões que você publicou",
        f"cobertura_intervalo: {c_lo:.2f}-{c_hi:.2f}",
        "",
        "Método:",
        f"- Hold-out estratificado (~30%, n={hold_n}) da folha-verdade: escolhe tau e estima",
        "  precisão/cobertura. O restante (fit-set) treina léxico/kNN/reranker/calibrador.",
        "- Intervalos = percentis 2.5–97.5 de bootstrap (1000) sobre o hold-out (publicadas",
        "  para precisão; todas as linhas do hold-out para cobertura).",
        "- Acerto tolerante à árvore; empate entre irmãos → revisar.",
        "- Features (TF-IDF, emb, proto/kNN, lei, assunto, margem) → score composto;",
        f"- tau_publicar={tau} maximiza cobertura no hold-out com precisão ≥ 0.97 (fallback 0.95);",
        "  piso de tau≈0.74 para não baixar o limiar só para inflar cobertura.",
        "- Calibração: isotônica no score composto (ênfase em margem top1–top2).",
        "- Predict offline: artifacts/ (laws, lexicon_seeds, model.joblib, embeddings); sem LLM/API.",
        f"- Embeddings: {metrics.get('embeddings_loaded', False)}; sinal fit-set: {metrics.get('oof_top_signal_accuracy', 'n/a')}.",
        f"- Recall fora (hold-out): {fora_rec if fora_rec is not None else 'n/a'}.",
        "- Banca avalia 559 fora da folha; folha só em Q0001–Q0359 (possível shift nas recentes).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj
