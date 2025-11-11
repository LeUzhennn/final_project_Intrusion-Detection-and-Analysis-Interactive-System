from src.data_loader import load_data, clean_data
from src.feature_selector import run_genetic_selection
from src.model_trainer import train_and_evaluate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st
import pandas as pd
import io

st.set_page_config(
    page_title="入侵偵測互動式分析系統",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ 入侵偵測互動式分析系統")

DATA_PATH = "data/03-01-2018.csv"

# 載入資料
df_raw = load_data(DATA_PATH)

if df_raw is not None:
    # 在清理前，強制將所有特徵欄位轉換為數值，無法轉換的會變成 NaN
    feature_cols = df_raw.columns.drop(['Label', 'Timestamp'])
    for col in feature_cols:
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    st.success(f"成功載入原始資料集，共 {df_raw.shape[0]} 筆記錄，{df_raw.shape[1]} 個欄位。")
    
    # 清理資料（現在也能移除上面步驟產生的 NaN）
    df_cleaned = clean_data(df_raw.copy()) # 使用 copy 避免修改到快取中的原始資料

    st.write("---")
    st.header("**目標變數 (Label) 分析**")
    label_counts = df_cleaned['Label'].value_counts()
    st.write("各類別資料筆數：")
    st.write(label_counts)

    st.subheader("目標變數分佈圖")
    st.bar_chart(label_counts)
    
    st.info("從上圖可知，資料集存在嚴重的類別不平衡問題，'Benign' (正常) 流量遠多於各類攻擊流量。這在後續模型評估時需要特別注意。")

    st.write("---")
    st.header("特徵選擇 (使用基因演算法)")

    if st.button("🚀 開始特徵選擇"):
        # 1. 資料預處理
        with st.spinner("正在進行資料預處理..."):
            # 分離特徵和目標
            X = df_cleaned.drop(columns=['Label', 'Timestamp'])
            y = df_cleaned['Label']

            # 將目標變數進行編碼
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)

            # 對特徵進行標準化
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
            st.session_state['scaler'] = scaler # 儲存 scaler
        st.success("資料預處理完成！")

        # 2. 執行基因演算法
        selected_features, best_score = run_genetic_selection(X_scaled, y_encoded)

        # 3. 顯示結果
        st.subheader("基因演算法選擇結果")
        st.success(f"演算法執行完畢！最佳分數 (Accuracy): {best_score:.4f}")
        st.metric(label="選擇的特徵數量", value=f"{len(selected_features)} / {len(X.columns)}")
        
        st.write("**選擇的特徵列表：**")
        st.dataframe(selected_features)

        # 將結果儲存到 session state 以便後續使用
        st.session_state['selection_done'] = True
        st.session_state['selected_features'] = selected_features
        st.session_state['X_scaled'] = X_scaled
        st.session_state['y_encoded'] = y_encoded
        st.session_state['le'] = le

    st.write("---")

    # --- 模型訓練區塊 ---
    if st.session_state.get('selection_done', False):
        st.header("3. 模型訓練與評估")
        if st.button("🧠 使用選定特徵進行模型訓練"):
            with st.spinner("正在準備訓練資料..."):
                X_selected = st.session_state['X_scaled'][st.session_state['selected_features']]
                y_encoded = st.session_state['y_encoded']
                le = st.session_state['le']

                X_train, X_test, y_train, y_test = train_test_split(
                    X_selected, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
                )
            st.success("資料分割完成 (80% 訓練, 20% 測試)！")

            # 訓練並評估
            metrics, model = train_and_evaluate(X_train, X_test, y_train, y_test, le.classes_)
            st.session_state['trained_model'] = model # 儲存訓練好的模型

            # 顯示評估指標
            st.subheader("模型評估指標")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['precision']:.4f}")
            col3.metric("Recall", f"{metrics['recall']:.4f}")
            col4.metric("F1-Score", f"{metrics['f1_score']:.4f}")
    
    st.write("---")

    # --- 即時預測區塊 ---
    if st.session_state.get('trained_model'):
        st.header("4. 即時預測")
        st.write("請輸入以下特徵值，來模擬一筆新的網路流量數據：")

        selected_features = st.session_state['selected_features']
        
        with st.form(key='prediction_form'):
            # 建立多列輸入
            num_cols = 4
            cols = st.columns(num_cols)
            user_inputs = {}
            for i, feature in enumerate(selected_features):
                with cols[i % num_cols]:
                    user_inputs[feature] = st.number_input(label=feature, value=0.0, format="%.4f")
            
            submit_button = st.form_submit_button(label='⚡ 執行預測')

        if submit_button:
            # 收集資料
            input_df = pd.DataFrame([user_inputs])

            # 縮放資料
            scaler = st.session_state['scaler']
            input_scaled = scaler.transform(input_df)

            # 預測
            model = st.session_state['trained_model']
            prediction = model.predict(input_scaled)

            # 解碼結果
            le = st.session_state['le']
            predicted_label = le.inverse_transform(prediction)[0]

            st.subheader("預測結果")
            if predicted_label == 'Benign':
                st.success(f"✅ 預測結果： **{predicted_label}** (正常)")
            else:
                st.error(f"🚨 預測結果： **{predicted_label}** (攻擊!)")

    st.write("---")

    # --- 批次預測區塊 (含欄位映射) ---
    if st.session_state.get('trained_model'):
        st.header("5. 批次預測 (上傳 CSV 檔案並映射欄位)")
        st.write("請上傳您的 CSV 檔案，並將其欄位映射到模型所需的特徵。")

        selected_features = st.session_state['selected_features']
        template_df = pd.DataFrame(columns=selected_features)
        csv_template = template_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="下載批次預測範例 CSV 檔案",
            data=csv_template,
            file_name="prediction_template.csv",
            mime="text/csv",
            help="下載一個包含所有選定特徵欄位的空白 CSV 檔案，您可以填入數據後上傳。"
        )

        uploaded_file = st.file_uploader("上傳 CSV 檔案", type=["csv"])

        if uploaded_file is not None:
            try:
                batch_df_raw = pd.read_csv(uploaded_file)
                st.write("上傳檔案預覽：")
                st.write(batch_df_raw.head())

                selected_features = st.session_state['selected_features']
                uploaded_columns = batch_df_raw.columns.tolist()

                st.subheader("欄位映射設定")
                st.info("請將模型所需的特徵，映射到您上傳檔案中對應的欄位。如果某個特徵在您的檔案中不存在，請選擇 '未映射'，系統將使用預設值 0 填充。")

                column_mapping = {}
                mapping_cols = st.columns(4)
                for i, feature in enumerate(selected_features):
                    with mapping_cols[i % 4]:
                        default_index = uploaded_columns.index(feature) if feature in uploaded_columns else 0
                        column_mapping[feature] = st.selectbox(
                            f"模型特徵: {feature}",
                            ['未映射'] + uploaded_columns,
                            index=default_index + 1 if feature in uploaded_columns else 0,
                            key=f"map_{feature}"
                        )
                
                if st.button("執行批次預測 (已映射)"):
                    with st.spinner("正在根據映射設定處理資料並進行預測..."):
                        # 建立用於預測的 DataFrame
                        batch_X_mapped = pd.DataFrame(0.0, index=batch_df_raw.index, columns=selected_features)

                        for model_feature, uploaded_col in column_mapping.items():
                            if uploaded_col != '未映射':
                                batch_X_mapped[model_feature] = pd.to_numeric(batch_df_raw[uploaded_col], errors='coerce')
                            # 如果是 '未映射'，則保持為 0.0 (預設值)
                        
                        # 處理 NaN 值 (可能來自 to_numeric 或未映射的特徵)
                        batch_X_mapped.dropna(inplace=True)

                        if batch_X_mapped.empty:
                            st.warning("預處理後，上傳檔案中沒有有效資料可供預測。請檢查您的映射和數據。")
                        else:
                            # 縮放資料
                            scaler = st.session_state['scaler']
                            batch_scaled = scaler.transform(batch_X_mapped)

                            # 預測
                            model = st.session_state['trained_model']
                            batch_predictions_encoded = model.predict(batch_scaled)

                            # 解碼結果
                            le = st.session_state['le']
                            batch_predictions_label = le.inverse_transform(batch_predictions_encoded)

                            # 將預測結果加入原始資料框 (只針對成功預測的行)
                            batch_df_results = batch_df_raw.loc[batch_X_mapped.index].copy() # 確保索引匹配
                            batch_df_results['Predicted_Label'] = batch_predictions_label

                            st.subheader("批次預測結果摘要")
                            prediction_counts = pd.Series(batch_predictions_label).value_counts()
                            st.write(prediction_counts)
                            st.bar_chart(prediction_counts)

                            st.subheader("帶有預測結果的資料")
                            st.dataframe(batch_df_results)

            except Exception as e:
                st.error(f"處理上傳檔案時發生錯誤：{e}")

    st.write("---")
    if st.checkbox("顯示清理後的資料摘要"):
        st.subheader("資料預覽 (前 5 筆)")
        st.write(df_cleaned.head())

        st.subheader("資料基本資訊")
        buffer = io.StringIO()
        df_cleaned.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)

        st.subheader("數值特徵統計摘要")
        st.write(df_cleaned.describe())
else:
    st.warning("請確認 `03-01-2018.csv` 已放置在 `data` 資料夾中。")