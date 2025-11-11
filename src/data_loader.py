"""此模組負責資料的讀取、解析與初步清理。"""
import pandas as pd
import numpy as np
import streamlit as st
import os

@st.cache_data
def load_data(file_path):
    """
    從指定路徑載入 CSV 檔案。

    Args:
        file_path (str): CSV 檔案的路徑。

    Returns:
        pd.DataFrame: 載入的資料，如果檔案不存在則回傳 None。
    """
    if not os.path.exists(file_path):
        st.error(f"錯誤：找不到檔案 {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"讀取檔案時發生錯誤：{e}")
        return None

def clean_data(df):
    """
    清理 DataFrame，處理無窮值和遺失值。

    Args:
        df (pd.DataFrame): 原始 DataFrame。

    Returns:
        pd.DataFrame: 清理後的 DataFrame。
    """
    st.header("資料清理")
    original_rows = len(df)
    
    # 將無窮值替換為 NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 檢查並移除遺失值
    nan_rows = df.isnull().any(axis=1).sum()
    if nan_rows > 0:
        st.info(f"發現 {nan_rows} 筆包含無效值(NaN/Infinity)的資料，將予以移除。")
        df.dropna(inplace=True)
    else:
        st.success("資料完整，沒有發現無效值(NaN/Infinity)。")

    cleaned_rows = len(df)
    st.write(f"清理完成，共移除 {original_rows - cleaned_rows} 筆資料。剩餘 {cleaned_rows} 筆有效資料。")
    return df