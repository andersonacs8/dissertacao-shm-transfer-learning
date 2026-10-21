# -*- coding: utf-8 -*-
"""
Módulo de Curva de Aprendizado e Isolamento da Fase A (Seções 1 e 2 do Prompt 3).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import balanced_accuracy_score

# Tenta importar o decodificador de classe do consistency
try:
    from src.consistency import inferir_classe_lumo
except ImportError:
    from consistency import inferir_classe_lumo

SUBCONJUNTOS_N = [100, 250, 500, 1000, 2000]

def isolar_dados_fase_a(df_real: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Seção 1: Garante o congelamento anti-leakage.
    Retorna apenas os conjuntos de Calibração e Limiar, sem acesso ao Teste.
    """
    df_temp = df_real.copy()
    if "classe" not in df_temp.columns:
        df_temp["classe"] = df_temp.apply(inferir_classe_lumo, axis=1)

    df_calib = df_temp[df_temp["conjunto"] == "calibracao"].copy().reset_index(drop=True)
    df_limiar = df_temp[df_temp["conjunto"] == "limiar"].copy().reset_index(drop=True)
    
    assert set(df_calib["classe"].unique()) == {0}, f"Calibração contém classes de dano: {set(df_calib['classe'].unique())}"
    assert set(df_limiar["classe"].unique()) == {0}, f"Limiar contém classes de dano: {set(df_limiar['classe'].unique())}"
    
    print("[FASE A: ANTI-LEAKAGE] Calibração (C) e Limiar (L) isolados com sucesso.")
    print(f" -> Calibração: {len(df_calib)} blocos ({df_calib['dia'].nunique()} dias)")
    print(f" -> Limiar:      {len(df_limiar)} blocos ({df_limiar['dia'].nunique()} dias)")
    print(" -> O conjunto de Teste permanece inacessível até a Fase B (Seção 7).")
    return df_calib, df_limiar


def obter_subconjunto_sintetico(df_sint: pd.DataFrame, n_por_classe: int, cols_feat: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extrai as primeiras n amostras de cada classe na ordem de meta.csv,
    preservando as realizações estocásticas completas.
    """
    dfs_sub = []
    classes = sorted(df_sint["classe"].unique())
    
    for c in classes:
        df_c = df_sint[df_sint["classe"] == c].iloc[:n_por_classe]
        if len(df_c) < n_por_classe:
            raise ValueError(f"Classe {c} possui apenas {len(df_c)} amostras, menor que {n_por_classe}.")
        dfs_sub.append(df_c)
        
    df_sub = pd.concat(dfs_sub, axis=0).reset_index(drop=True)
    
    X = df_sub[cols_feat].values.astype(np.float32)
    y = df_sub["classe"].values.astype(np.int64)
    groups = df_sub["realizacao_id"].values
    
    return X, y, groups


def padronizar_com_s0(X: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Padronização estrita usando apenas a média e desvio das amostras íntegras (classe 0).
    """
    idx_s0 = (y == 0)
    assert np.any(idx_s0), "Nenhuma amostra de classe 0 encontrada no subconjunto."
    
    mu_s0 = np.mean(X[idx_s0], axis=0)
    sigma_s0 = np.std(X[idx_s0], axis=0)
    sigma_s0 = np.maximum(sigma_s0, eps)
    
    return (X - mu_s0) / sigma_s0


def avaliar_curva_aprendizado(
    df_sint: pd.DataFrame, 
    cols_feat: List[str], 
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Seção 2: Executa a curva de aprendizado para n in {100, 250, 500, 1000, 2000}.
    """
    resultados = {
        "n_lista": SUBCONJUNTOS_N,
        "acc_media": [],
        "acc_std": [],
        "detalhes_dobras": []
    }
    
    print(f"\n[SEÇÃO 2] Iniciando avaliação da Curva de Aprendizado...")
    print(f" -> Grade de tamanhos por classe: {SUBCONJUNTOS_N}")
    print(f" -> Validador: StratifiedGroupKFold(5) agrupado por realizacao_id")
    print(f" -> Modelo: SVC(C=10.0, gamma='scale') com padronização via S0")
    
    for n in SUBCONJUNTOS_N:
        X_sub, y_sub, groups = obter_subconjunto_sintetico(df_sint, n, cols_feat)
        X_norm = padronizar_com_s0(X_sub, y_sub)
        
        sgkf = StratifiedGroupKFold(n_splits=5)
        scores_dobras = []
        
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X_norm, y_sub, groups=groups)):
            X_tr, y_tr = X_norm[train_idx], y_sub[train_idx]
            X_val, y_val = X_norm[val_idx], y_sub[val_idx]
            
            clf = SVC(C=10.0, gamma="scale", random_state=random_state)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_val)
            
            acc_bal = balanced_accuracy_score(y_val, y_pred)
            scores_dobras.append(acc_bal)
            
        media = float(np.mean(scores_dobras))
        desvio = float(np.std(scores_dobras))
        resultados["acc_media"].append(media)
        resultados["acc_std"].append(desvio)
        resultados["detalhes_dobras"].append(scores_dobras)
        
        print(f"  * n = {n:>4} por classe | Acurácia Balanceada: {media*100:6.2f}% (+/- {desvio*100:4.2f}%)")

    # Regra de decisão do Prompt 3: menor n tal que nenhum aumento subsequente mude a acurácia em > 1 p.p. (0.01)
    acc_medias = resultados["acc_media"]
    n_escolhido = SUBCONJUNTOS_N[-1]
    
    for i in range(len(SUBCONJUNTOS_N)):
        estavel = True
        for j in range(i + 1, len(SUBCONJUNTOS_N)):
            if abs(acc_medias[j] - acc_medias[i]) > 0.01:
                estavel = False
                break
        if estavel:
            n_escolhido = SUBCONJUNTOS_N[i]
            break
            
    resultados["n_escolhido"] = n_escolhido
    print(f"\n[DECISÃO METODOLÓGICA] Menor n estável selecionado: {n_escolhido} amostras por classe.")
    return resultados


def plotar_curva_aprendizado(resultados: Dict[str, Any], save_path: str = None) -> plt.Figure:
    """
    Gera a figura de controle da Curva de Aprendizado (Acurácia Balanceada x n).
    """
    n_arr = np.array(resultados["n_lista"])
    media_arr = np.array(resultados["acc_media"]) * 100
    std_arr = np.array(resultados["acc_std"]) * 100
    n_sel = resultados["n_escolhido"]
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.plot(n_arr, media_arr, "o-", color="#1f77b4", lw=2, label="Média (5-fold CV)")
    ax.fill_between(n_arr, media_arr - std_arr, media_arr + std_arr, color="#1f77b4", alpha=0.2, label=r"Intervalo $\pm 1\sigma$")
    
    ax.axvline(x=n_sel, color="#d62728", linestyle="--", lw=1.5, label=f"n Escolhido = {n_sel}")
    
    ax.set_title("Seção 2: Curva de Aprendizado no Domínio Fonte Sintético", fontsize=12, pad=10)
    ax.set_xlabel("Número de Amostras por Classe (n)", fontsize=10)
    ax.set_ylabel("Acurácia Balanceada (%)", fontsize=10)
    ax.set_xticks(n_arr)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300)
        
    return fig
