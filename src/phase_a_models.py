# -*- coding: utf-8 -*-
"""
Módulo de Modelagem, Alinhamento e Limiares da Fase A (Seções 3, 4, 5 e 6 do Prompt 3).
Otimizado com checkpoints incrementais e barra de progresso (tqdm).
"""

import os
import json
import hashlib
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score
from joblib import Parallel, delayed
from tqdm.auto import tqdm

EPS = 1e-12

def extrair_conjunto_trabalho_sintetico(
    df_sint: pd.DataFrame, 
    n_por_classe: int, 
    cols_feat: List[str]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extrai as primeiras n amostras de cada classe na ordem de meta.csv."""
    dfs = [df_sint[df_sint["classe"] == c].iloc[:n_por_classe] for c in sorted(df_sint["classe"].unique())]
    df_sub = pd.concat(dfs, axis=0).reset_index(drop=True)
    
    X = df_sub[cols_feat].values.astype(np.float64)
    y = df_sub["classe"].values.astype(np.int64)
    groups = df_sub["realizacao_id"].values
    return X, y, groups


def computar_alinhamentos(
    X_s: np.ndarray, 
    y_s: np.ndarray, 
    X_c: np.ndarray, 
    X_l: np.ndarray
) -> Dict[str, Any]:
    """Seção 3: Calcula No-DA, NCA e CORAL usando estritamente dados íntegros."""
    idx_s0 = (y_s == 0)
    assert np.any(idx_s0), "Nenhuma amostra de classe 0 no domínio fonte!"
    
    # 1. No-DA
    mu_s0 = np.mean(X_s[idx_s0], axis=0)
    std_s0 = np.maximum(np.std(X_s[idx_s0], axis=0), EPS)
    
    X_s_noda = (X_s - mu_s0) / std_s0
    X_c_noda = (X_c - mu_s0) / std_s0
    X_l_noda = (X_l - mu_s0) / std_s0

    # 2. NCA
    mu_t = np.mean(X_c, axis=0)
    std_t = np.maximum(np.std(X_c, axis=0), EPS)
    
    X_s_nca = (X_s - mu_s0) / std_s0
    X_c_nca = (X_c - mu_t) / std_t
    X_l_nca = (X_l - mu_t) / std_t

    # 3. CORAL
    S0_nca = X_s_nca[idx_s0]
    C_nca = X_c_nca
    d = X_s.shape[1]
    
    Cs = np.cov(S0_nca, rowvar=False) + np.eye(d)
    Ct = np.cov(C_nca, rowvar=False) + np.eye(d)
    
    evals_s, evecs_s = np.linalg.eigh(Cs)
    evals_s = np.maximum(evals_s, EPS)
    Cs_inv_sqrt = evecs_s @ np.diag(1.0 / np.sqrt(evals_s)) @ evecs_s.T
    
    evals_t, evecs_t = np.linalg.eigh(Ct)
    evals_t = np.maximum(evals_t, EPS)
    Ct_sqrt = evecs_t @ np.diag(np.sqrt(evals_t)) @ evecs_t.T
    
    A_coral = Cs_inv_sqrt @ Ct_sqrt
    X_s_coral = X_s_nca @ A_coral
    X_c_coral = X_c_nca
    X_l_coral = X_l_nca

    return {
        "stats": {
            "mu_s0": mu_s0, "std_s0": std_s0,
            "mu_t": mu_t, "std_t": std_t,
            "A_coral": A_coral
        },
        "No-DA": {"Xs": X_s_noda, "Xc": X_c_noda, "Xl": X_l_noda},
        "NCA":   {"Xs": X_s_nca,  "Xc": X_c_nca,  "Xl": X_l_nca},
        "CORAL": {"Xs": X_s_coral,"Xc": X_c_coral,"Xl": X_l_coral}
    }


def _avaliar_dobra_svm(c, g, tr_idx, val_idx, X_train, y_train, random_state):
    clf = SVC(C=c, gamma=g, kernel="rbf", probability=False, random_state=random_state, tol=1e-3)
    clf.fit(X_train[tr_idx], y_train[tr_idx])
    preds = clf.predict(X_train[val_idx])
    return balanced_accuracy_score(y_train[val_idx], preds)


def otimizar_svm(
    X_train: np.ndarray, 
    y_train: np.ndarray, 
    groups: np.ndarray, 
    nome_espaco: str = "",
    checkpoint_dir: str = None,
    random_state: int = 42
) -> Tuple[SVC, Dict[str, Any]]:
    """
    Seção 4: Otimização com verificação de checkpoints em disco e barra de progresso.
    """
    # 1. Verificar se o modelo deste espaço já foi salvo em checkpoint
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        caminho_chk = os.path.join(checkpoint_dir, f"svm_{nome_espaco}.pkl")
        if os.path.exists(caminho_chk):
            print(f" -> [CHECKPOINT ENCONTRADO] Carregando modelo pré-computado para {nome_espaco}...")
            with open(caminho_chk, "rb") as f:
                chk_data = pickle.load(f)
            return chk_data["modelo"], chk_data["info"]

    grade_C = [1.0, 10.0, 100.0]
    grade_gamma = [1e-4, 1e-3, "scale"]
    sgkf = StratifiedGroupKFold(n_splits=5)
    splits = list(sgkf.split(X_train, y_train, groups=groups))
    
    melhor_score = -1.0
    melhor_params = {}
    registros_grade = []
    
    grade_combinacoes = [(c, g) for c in grade_C for g in grade_gamma]
    
    print(f"\n[Ajuste SVM: {nome_espaco}] Avaliando 9 combinações de hiperparâmetros (5-fold CV agrupado)...")
    pbar = tqdm(grade_combinacoes, desc=f"Grade {nome_espaco}", unit="comb")
    
    for c, g in pbar:
        # Paraleliza as 5 dobras nos cores disponíveis da CPU
        scores_cv = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_avaliar_dobra_svm)(c, g, tr_idx, val_idx, X_train, y_train, random_state)
            for tr_idx, val_idx in splits
        )
        media_score = float(np.mean(scores_cv))
        registros_grade.append({"C": c, "gamma": str(g), "acc_cv": media_score})
        
        pbar.set_postfix({"C": c, "gamma": g, "acc": f"{media_score*100:.2f}%"})
        
        if media_score > melhor_score:
            melhor_score = media_score
            melhor_params = {"C": c, "gamma": g}

    print(f" -> Reajustando modelo final ({nome_espaco}) com C={melhor_params['C']}, gamma={melhor_params['gamma']}...")
    modelo_final = SVC(
        C=melhor_params["C"], 
        gamma=melhor_params["gamma"], 
        kernel="rbf", 
        probability=True, 
        random_state=random_state,
        tol=1e-3
    )
    modelo_final.fit(X_train, y_train)
    
    info_resultado = {"melhor_score": melhor_score, "params": melhor_params, "grade": registros_grade}
    
    # 2. Salvar checkpoint no Drive imediatamente
    if checkpoint_dir:
        with open(caminho_chk, "wb") as f:
            pickle.dump({"modelo": modelo_final, "info": info_resultado}, f)
        print(f" -> [OK] Checkpoint gravado: {caminho_chk}")
        
    return modelo_final, info_resultado


def ajustar_target_only(X_c_nca: np.ndarray) -> Dict[str, Any]:
    """Seção 5: PCA (>= 95% variância) + MSD com pinv em C."""
    pca = PCA(n_components=0.95, svd_solver="full")
    Z_c = pca.fit_transform(X_c_nca)
    
    mu_z = np.mean(Z_c, axis=0)
    cov_z = np.cov(Z_c, rowvar=False)
    
    cond_num = float(np.linalg.cond(cov_z))
    inv_cov_z = np.linalg.pinv(cov_z)
    
    return {
        "pca": pca,
        "mu_z": mu_z,
        "inv_cov_z": inv_cov_z,
        "cond_num": cond_num,
        "n_componentes": pca.n_components_
    }


def calcular_escore_msd(X_nca: np.ndarray, mod_to: Dict[str, Any]) -> np.ndarray:
    """Calcula a Distância Quadrática de Mahalanobis no subespaço PCA."""
    Z = mod_to["pca"].transform(X_nca)
    diff = Z - mod_to["mu_z"]
    return np.sum((diff @ mod_to["inv_cov_z"]) * diff, axis=1)


def computar_limiares_congelar(
    modelos_svm: Dict[str, SVC],
    modelo_target_only: Dict[str, Any],
    dados_alinhados: Dict[str, Any],
    subconjunto_meta: Dict[str, Any],
    diretorio_saida: str
) -> Dict[str, Any]:
    """Seção 6 e Seção 1: Limiares T (Percentil 99) e Congelamento com manifesto SHA-256."""
    os.makedirs(diretorio_saida, exist_ok=True)
    limiares = {}
    
    # 1. Escores no Limiar para as SVMs: s = 1 - P(classe 0)
    for espaco in ["No-DA", "NCA", "CORAL"]:
        clf = modelos_svm[espaco]
        X_l = dados_alinhados[espaco]["Xl"]
        idx_c0 = np.where(clf.classes_ == 0)[0][0]
        prob_c0 = clf.predict_proba(X_l)[:, idx_c0]
        escores = 1.0 - prob_c0
        
        T = float(np.percentile(escores, 99, method="linear"))
        limiares[f"{espaco}+SVM"] = {"T": T, "FPR_limiar": float(np.mean(escores > T))}
        
    # 2. Escore no Limiar para o Target-Only
    escores_to = calcular_escore_msd(dados_alinhados["NCA"]["Xl"], modelo_target_only)
    T_to = float(np.percentile(escores_to, 99, method="linear"))
    limiares["target-only"] = {"T": T_to, "FPR_limiar": float(np.mean(escores_to > T_to))}

    # 3. Empacotar Artefatos Congelados
    artefatos = {
        "stats": dados_alinhados["stats"],
        "modelos_svm": modelos_svm,
        "modelo_target_only": modelo_target_only,
        "limiares": limiares,
        "meta": subconjunto_meta
    }
    
    caminho_artefatos = os.path.join(diretorio_saida, "artefatos_congelados.pkl")
    with open(caminho_artefatos, "wb") as f:
        pickle.dump(artefatos, f)
        
    # 4. Manifesto SHA-256
    with open(caminho_artefatos, "rb") as f:
        sha256 = hashlib.sha256(f.read()).hexdigest()
        
    manifesto = {
        "arquivo": "artefatos_congelados.pkl",
        "sha256": sha256,
        "n_por_classe": subconjunto_meta["n_por_classe"],
        "limiares": limiares
    }
    
    with open(os.path.join(diretorio_saida, "manifesto.json"), "w", encoding="utf-8") as f:
        json.dump(manifesto, f, indent=4)
        
    print(f"\n[FASE A CONCLUÍDA] Artefatos congelados com sucesso em: {diretorio_saida}")
    print(f" -> SHA-256: {sha256}")
    return artefatos
