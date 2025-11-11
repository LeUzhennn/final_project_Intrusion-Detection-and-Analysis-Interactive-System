"""此模組負責根據不同策略進行特徵選擇，使用 DEAP 函式庫。"""
import random
import streamlit as st
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# --- DEAP 全域設定 ---
# 建立問題：最大化適應度（準確率）
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# 建立個體：一個列表，並帶有 FitnessMax 的屬性
creator.create("Individual", list, fitness=creator.FitnessMax)


@st.cache_data(show_spinner=False)
def run_genetic_selection(_X, _y):
    """
    使用 DEAP 函式庫執行基因演算法進行特徵選擇。
    使用 Streamlit 的快取來儲存結果。
    """
    n_features = _X.shape[1]

    # --- 工具箱設定 ---
    toolbox = base.Toolbox()
    # 定義如何產生一個基因（0 或 1）
    toolbox.register("attr_bool", random.randint, 0, 1)
    # 定義如何產生一個個體（由 n_features 個基因組成）
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features)
    # 定義如何產生一個族群（由多個個體組成）
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # --- 評估函式 ---
    def evaluate_features(individual):
        selected_indices = [i for i, bit in enumerate(individual) if bit == 1]
        if not selected_indices:
            return 0.0,

        X_subset = _X.iloc[:, selected_indices]
        estimator = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
        score = np.mean(cross_val_score(estimator, X_subset, _y, cv=3, scoring='accuracy'))
        return score,

    # --- 註冊遺傳算子 ---
    toolbox.register("evaluate", evaluate_features)
    toolbox.register("mate", tools.cxTwoPoint) # 交配
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) # 突變
    toolbox.register("select", tools.selTournament, tournsize=3) # 選擇

    # --- 執行演算法 ---
    with st.spinner("正在使用 DEAP 核心執行基因演算法...這可能需要幾分鐘。"):
        pop = toolbox.population(n=40) # 族群大小
        hof = tools.HallOfFame(1)      # 名人堂，儲存最佳個體
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("max", np.max)

        # 執行演算法
        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=15, 
                                       stats=stats, halloffame=hof, verbose=False)

        best_individual = hof[0]
        selected_features_mask = np.array(best_individual).astype(bool)
        selected_features = _X.columns[selected_features_mask].tolist()
        best_score = best_individual.fitness.values[0]

    return selected_features, best_score