# -*- coding: utf-8 -*-
"""
Módulo de Verificações de Consistência (Seção 0 do Prompt 3).
Executa a validação de integridade estrutural, dimensional e metodológica dos dados.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

CLASSES_TESTE_ESPERADAS = {0, 5, 6, 7, 8, 11, 12}

def inferir_classe_lumo(row: pd.Series) -> int:
    """
    Decodifica a classe segundo a Metodologia (Prompt 3):
    classe 0 = íntegro
    classe = 1 + 2*(L-1) + (0 se 1 diagonal, 1 se 3 diagonais)
    onde L = 1 (DAM1) ... 6 (DAM6)
    """
    # 1. Se já existir classe válida preenchida (0 a 12)
    if 'classe' in row and pd.notnull(row['classe']):
        try:
            val_c = int(row['classe'])
            if 0 <= val_c <= 12:
                return val_c
        except (ValueError, TypeError):
            pass

    # 2. Casos Íntegros (classe 0)
    estado = str(row.get('estado', '')).strip().lower()
    subconjunto = str(row.get('subconjunto', '')).strip().lower()
    nivel_raw = str(row.get('nivel_dam', '')).strip().upper()

    if estado in ['healthy', 'integro', 'íntegro', 'original'] or 'integro' in subconjunto or nivel_raw in ['0', 'NONE', 'NAN']:
        return 0

    # 3. Extrair Nível L (1 a 6)
    # nivel_dam pode ser int (3, 4, 6) ou texto ('DAM3', 'DAM4', 'DAM6')
    L = None
    if nivel_raw in ['1', '2', '3', '4', '5', '6']:
        L = int(nivel_raw)
    elif 'DAM' in nivel_raw:
        for num in range(1, 7):
            if f'DAM{num}' in nivel_raw:
                L = num
                break
    elif 'DAM' in estado.upper():
        for num in range(1, 7):
            if f'DAM{num}' in estado.upper():
                L = num
                break
    elif 'DAM' in subconjunto.upper():
        for num in range(1, 7):
            if f'DAM{num}' in subconjunto.upper():
                L = num
                break

    if L is None:
        return -1

    # 4. Extrair Extensão (1 ou 3 diagonais)
    ext = row.get('extensao', None)
    if pd.notnull(ext):
        try:
            ext = int(ext)
        except (ValueError, TypeError):
            ext = None

    if ext not in [1, 3]:
        # Tenta obter pelo nome do subconjunto (ex: teste_dano_DAM4_1diag)
        if '1diag' in subconjunto or '1_diag' in subconjunto or '1diag' in estado:
            ext = 1
        elif '3diag' in subconjunto or '3_diag' in subconjunto or '3diag' in estado:
            ext = 3
        else:
            # Fallback cronológico (Wernitz et al., Tabela 3)
            dia_str = str(row.get('dia', row.get('data_hora', ''))).replace('-', '')
            if dia_str.startswith('2020') or '202103' in dia_str or '202104' in dia_str:
                ext = 3
            else:
                ext = 1

    # Fórmula do Prompt 3: classe = 1 + 2*(L - 1) + (0 se 1 diag, 1 se 3 diag)
    offset = 1 if ext == 3 else 0
    return 1 + 2 * (L - 1) + offset


def verificar_consistencia(
    df_sintetico: pd.DataFrame, 
    df_real: pd.DataFrame, 
    dimensao_esperada: int = 3620
) -> Tuple[bool, Dict[str, Any]]:
    """
    Realiza todas as verificações de consistência obrigatórias da Seção 0.
    """
    relatorio = {}
    erros = []

    # 1. Identificar colunas de atributos
    cols_feat_sint = [c for c in df_sintetico.columns if c.startswith("feat_")]
    cols_feat_real = [c for c in df_real.columns if c.startswith("feat_")]

    d_sint = len(cols_feat_sint)
    d_real = len(cols_feat_real)

    relatorio["dimensao_features"] = {"sintetico": d_sint, "real": d_real}
    if d_sint != dimensao_esperada or d_real != dimensao_esperada:
        erros.append(f"Dimensão incompatível: sintético={d_sint}, real={d_real}, esperado={dimensao_esperada}")

    # 2. Ausência de NaN e Inf
    X_sint = df_sintetico[cols_feat_sint].values
    X_real = df_real[cols_feat_real].values

    nan_sint = bool(np.isnan(X_sint).any())
    inf_sint = bool(np.isinf(X_sint).any())
    nan_real = bool(np.isnan(X_real).any())
    inf_real = bool(np.isinf(X_real).any())

    relatorio["integridade_numerica"] = {
        "sintetico_nan": nan_sint, "sintetico_inf": inf_sint,
        "real_nan": nan_real, "real_inf": inf_real
    }
    if nan_sint or inf_sint or nan_real or inf_real:
        erros.append("Detectados valores NaN ou Inf nas matrizes de atributos.")

    # 3. Disjunção de dias no conjunto real
    particoes = df_real["conjunto"].unique()
    dias_por_particao = {p: set(df_real[df_real["conjunto"] == p]["dia"].unique()) for p in particoes}
    
    dias_calib = dias_por_particao.get("calibracao", set())
    dias_limiar = dias_por_particao.get("limiar", set())
    dias_teste = dias_por_particao.get("teste", set())

    inter_calib_limiar = dias_calib.intersection(dias_limiar)
    inter_calib_teste = dias_calib.intersection(dias_teste)
    inter_limiar_teste = dias_limiar.intersection(dias_teste)

    relatorio["dias_disjuntos"] = {
        "inter_calib_limiar": list(inter_calib_limiar),
        "inter_calib_teste": list(inter_calib_teste),
        "inter_limiar_teste": list(inter_limiar_teste)
    }
    if inter_calib_limiar or inter_calib_teste or inter_limiar_teste:
        erros.append("Violação de disjunção temporal entre calibração, limiar e teste.")

    # 4. Classes de dano em Calibração, Limiar e Teste
    df_real_calc = df_real.copy()
    df_real_calc['classe'] = df_real_calc.apply(inferir_classe_lumo, axis=1)

    classes_calib = set(df_real_calc[df_real_calc["conjunto"] == "calibracao"]["classe"].unique())
    classes_limiar = set(df_real_calc[df_real_calc["conjunto"] == "limiar"]["classe"].unique())
    classes_teste = set(df_real_calc[df_real_calc["conjunto"] == "teste"]["classe"].unique())

    relatorio["classes_identificadas"] = {
        "calibracao": list(classes_calib),
        "limiar": list(classes_limiar),
        "teste": sorted(list(classes_teste))
    }

    if classes_calib != {0}:
        erros.append(f"Calibração contém classes de dano não íntegras: {classes_calib}")
    if classes_limiar != {0}:
        erros.append(f"Limiar contém classes de dano não íntegras: {classes_limiar}")
    if not classes_teste.issubset(CLASSES_TESTE_ESPERADAS):
        erros.append(f"Teste contém classes fora do conjunto esperado {CLASSES_TESTE_ESPERADAS}: {classes_teste}")

    # 5. Censo de blocos e dias por conjunto
    censo = df_real_calc.groupby(["conjunto", "estado", "classe"]).agg(
        n_blocos=("bloco_id", "count"),
        n_dias=("dia", "nunique")
    ).reset_index()
    relatorio["tabela_censo"] = censo

    sucesso = (len(erros) == 0)
    relatorio["status"] = "APROVADO" if sucesso else "FALHA"
    relatorio["erros"] = erros

    return sucesso, relatorio
