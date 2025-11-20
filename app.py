from src.data_loader import load_data, clean_data
from src.feature_selector import run_genetic_selection
from src.model_trainer import train_and_evaluate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt
import seaborn as sns
import shap

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
            metrics, model, cm_df = train_and_evaluate(X_train, X_test, y_train, y_test, le.classes_)
            st.session_state['trained_model'] = model # 儲存訓練好的模型

            with st.spinner("建立 SHAP 解釋器..."):
                explainer = shap.TreeExplainer(model)
                st.session_state['shap_explainer'] = explainer

            # 顯示評估指標
            st.subheader("模型評估指標")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['precision']:.4f}")
            col3.metric("Recall", f"{metrics['recall']:.4f}")
            col4.metric("F1-Score", f"{metrics['f1_score']:.4f}")

            st.subheader("混淆矩陣 (Confusion Matrix)")
            st.info("混淆矩陣顯示模型在各類別上的預測表現。對角線上的數字代表正確預測的數量，非對角線則代表錯誤預測的數量。")
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)
    
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
            # 收集使用者輸入的資料 (只包含 selected_features)
            input_df_user = pd.DataFrame([user_inputs])

            # 取得 scaler, model, le
            scaler = st.session_state['scaler']
            model = st.session_state['trained_model']
            le = st.session_state['le']
            
            # 取得 scaler 訓練時所需的所有特徵欄位
            required_features_for_scaler = scaler.feature_names_in_
            
            # 建立一個符合 scaler 輸入要求的完整 DataFrame，預設值為 0
            input_df_full = pd.DataFrame(0.0, index=[0], columns=required_features_for_scaler)
            
            # 將使用者的輸入填入對應的欄位
            for col in input_df_user.columns:
                if col in input_df_full.columns:
                    input_df_full[col] = input_df_user[col].values

            # 使用 scaler 對完整的資料進行縮放
            input_scaled_full = scaler.transform(input_df_full)
            
            # 將縮放後的 numpy array 轉回 DataFrame，並加上欄位名稱
            input_scaled_df = pd.DataFrame(input_scaled_full, columns=required_features_for_scaler)
            
            # 從縮放後的完整資料中，篩選出模型訓練時使用的特徵
            final_input_for_model = input_scaled_df[st.session_state['selected_features']]

            # 預測
            prediction = model.predict(final_input_for_model)

            # 解碼結果
            predicted_label = le.inverse_transform(prediction)[0]

            st.subheader("預測結果")
            if predicted_label == 'Benign':
                st.success(f"✅ 預測結果： **{predicted_label}** (正常)")
            else:
                st.error(f"🚨 預測結果： **{predicted_label}** (攻擊!)")

            # --- SHAP 解釋 ---
            st.subheader("模型預測解釋 (SHAP Analysis)")
            st.info("下圖顯示了各個特徵如何將預測結果從基準值（Base value）推向最終的預測值。紅色特徵增加了預測為該類別的機率，藍色特徵則降低了機率。")

            with st.spinner("正在計算 SHAP 值..."):
                try:
                    explainer = st.session_state['shap_explainer']
                    shap_values = explainer.shap_values(final_input_for_model)
                    predicted_class_index = prediction[0]

                    # Handle both multi-class and binary classification outputs from SHAP
                    if isinstance(explainer.expected_value, (list, np.ndarray)):
                        # Multi-class case
                        shap_base_value = explainer.expected_value[predicted_class_index]
                        shap_values_for_class = shap_values[predicted_class_index][0]
                    else:
                        # Binary case
                        shap_base_value = explainer.expected_value
                        shap_values_for_class = shap_values[0]
                    
                    # 繪製 SHAP force plot
                    st.write(f"**對於類別 `{predicted_label}` 的解釋：**")
                    fig, ax = plt.subplots(figsize=(20, 4))
                    shap.force_plot(
                        shap_base_value,
                        shap_values_for_class,
                        final_input_for_model.iloc[0],
                        matplotlib=True,
                        show=False,
                        text_rotation=15
                    )
                    plt.tight_layout()
                    st.pyplot(fig, bbox_inches='tight')
                    plt.close(fig)
                except Exception as e:
                    st.warning(f"無法產生 SHAP 分析圖：{e}")

    st.write("---")

    # --- 批次預測區塊 (含欄位映射) ---
    if st.session_state.get('trained_model'):
        st.header("5. 批次流量分析 (上傳 CSV)")
        st.write("上傳包含多筆網路流量的 CSV 檔，系統將逐筆分析並判斷是否為攻擊。")

        selected_features = st.session_state['selected_features']
        template_df = pd.DataFrame(columns=selected_features)
        csv_template = template_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="下載分析範例 CSV 檔案",
            data=csv_template,
            file_name="prediction_template.csv",
            mime="text/csv",
            help="下載一個包含所有選定特徵欄位的空白 CSV 檔案，您可以填入數據後上傳。"
        )

        uploaded_file = st.file_uploader("上傳待分析的 CSV 檔案", type=["csv"])

        if uploaded_file is not None:
            # --- STATE MANAGEMENT ---
            # If a new file is uploaded, clear old analysis results
            if 'current_file_name' not in st.session_state or st.session_state.current_file_name != uploaded_file.name:
                st.session_state.current_file_name = uploaded_file.name
                if 'batch_results_df' in st.session_state:
                    del st.session_state['batch_results_df']
            # --- END STATE MANAGEMENT ---

            try:
                batch_df_raw = pd.read_csv(uploaded_file)
                # Replace infinite values with NaN to prevent scaler errors
                batch_df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
                
                with st.expander("點此查看上傳的原始資料 (前 5 筆)"):
                    st.dataframe(batch_df_raw.head())

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
                
                if st.button("🚀 開始分析流量"):
                    with st.spinner("正在根據映射設定處理資料並進行分析..."):
                        # 建立一個只包含使用者映射欄位的 DataFrame
                        batch_X_mapped_user = pd.DataFrame(0.0, index=batch_df_raw.index, columns=selected_features)

                        for model_feature, uploaded_col in column_mapping.items():
                            if uploaded_col != '未映射':
                                batch_X_mapped_user[model_feature] = pd.to_numeric(batch_df_raw[uploaded_col], errors='coerce')
                        
                        # 處理 NaN 值
                        batch_X_mapped_user.dropna(inplace=True)

                        if batch_X_mapped_user.empty:
                            st.warning("預處理後，上傳檔案中沒有有效資料可供分析。請檢查您的映射和數據。")
                            # If empty, make sure we don't show old results
                            if 'batch_results_df' in st.session_state:
                                del st.session_state['batch_results_df']
                        else:
                            # --- START OF PREDICTION LOGIC ---
                            scaler = st.session_state['scaler']
                            model = st.session_state['trained_model']
                            le = st.session_state['le']
                            
                            required_features_for_scaler = scaler.feature_names_in_
                            
                            batch_df_full = pd.DataFrame(0.0, index=batch_X_mapped_user.index, columns=required_features_for_scaler)
                            
                            for col in batch_X_mapped_user.columns:
                                if col in batch_df_full.columns:
                                    batch_df_full[col] = batch_X_mapped_user[col]

                            batch_scaled_full = scaler.transform(batch_df_full)
                            
                            batch_scaled_df = pd.DataFrame(batch_scaled_full, index=batch_df_full.index, columns=required_features_for_scaler)
                            
                            final_batch_for_model = batch_scaled_df[st.session_state['selected_features']]
                            
                            # --- STORE SCALED DATA FOR SHAP ---
                            st.session_state['final_batch_for_model'] = final_batch_for_model
                            # --- END ---

                            batch_predictions_encoded = model.predict(final_batch_for_model)

                            batch_predictions_label = le.inverse_transform(batch_predictions_encoded)

                            batch_df_results = batch_df_raw.loc[final_batch_for_model.index].copy()
                            batch_df_results['Predicted_Label'] = batch_predictions_label
                            
                            batch_df_results['分析結果'] = batch_df_results['Predicted_Label'].apply(lambda x: '攻擊' if x != 'Benign' else '正常')
                            
                            # --- STORE RESULTS IN SESSION STATE ---
                            st.session_state['batch_results_df'] = batch_df_results

            except Exception as e:
                st.error(f"處理上傳檔案時發生錯誤：{e}")
                if 'batch_results_df' in st.session_state:
                    del st.session_state['batch_results_df']

            # --- DISPLAY RESULTS (MOVED OUTSIDE THE BUTTON LOGIC) ---
            if 'batch_results_df' in st.session_state:
                batch_df_results = st.session_state['batch_results_df']
                
                st.subheader("📊 分析結果總覽")
                prediction_counts = batch_df_results['分析結果'].value_counts()
                st.bar_chart(prediction_counts)

                st.subheader("📄 詳細分析結果")
                filter_option = st.radio(
                    "篩選顯示結果：",
                    ('顯示全部', '僅顯示攻擊', '僅顯示正常'),
                    horizontal=True,
                    key='filter_radio'
                )

                if filter_option == '僅顯示攻擊':
                    filtered_df = batch_df_results[batch_df_results['分析結果'] == '攻擊']
                elif filter_option == '僅顯示正常':
                    filtered_df = batch_df_results[batch_df_results['分析結果'] == '正常']
                else:
                    filtered_df = batch_df_results

                if filtered_df.empty:
                    st.info("在目前的篩選條件下，沒有可顯示的資料。")
                else:
                    final_cols = ['分析結果', 'Predicted_Label'] + [col for col in batch_df_raw.columns if col not in ['分析結果', 'Predicted_Label']]
                    st.dataframe(filtered_df[final_cols])
                    csv_results = filtered_df[final_cols].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 下載目前的分析結果",
                        data=csv_results,
                        file_name="traffic_analysis_results.csv",
                        mime="text/csv"
                    )

                # --- SHAP Drill-down Analysis ---
                st.subheader("🔬 深入分析單筆攻擊流量 (SHAP Drill-down)")
                
                attack_df = batch_df_results[batch_df_results['分析結果'] == '攻擊']
                if attack_df.empty:
                    st.info("在目前的分析結果中，沒有偵測到攻擊流量可供深入分析。")
                else:
                    st.write("從被標記為「攻擊」的流量中選擇一筆，查看模型的判斷依據。")
                    selected_index = st.selectbox(
                        "選擇一筆攻擊流量的索引 (Index) 進行分析：",
                        options=attack_df.index
                    )

                    if selected_index is not None:
                        with st.spinner("正在為您選擇的流量產生 SHAP 分析圖..."):
                            try:
                                explainer = st.session_state['shap_explainer']
                                final_batch_for_model = st.session_state['final_batch_for_model']
                                le = st.session_state['le']

                                # 取得該筆流量的資料與預測結果
                                single_instance = final_batch_for_model.loc[[selected_index]]
                                single_prediction_label = batch_df_results.loc[selected_index, 'Predicted_Label']
                                single_prediction_index = list(le.classes_).index(single_prediction_label)

                                # 計算 SHAP 值
                                shap_values = explainer.shap_values(single_instance)

                                # Handle both multi-class and binary classification outputs from SHAP
                                if isinstance(explainer.expected_value, (list, np.ndarray)):
                                    # Multi-class case
                                    shap_base_value = explainer.expected_value[single_prediction_index]
                                    shap_values_for_class = shap_values[single_prediction_index][0]
                                else:
                                    # Binary case
                                    shap_base_value = explainer.expected_value
                                    shap_values_for_class = shap_values[0]

                                # 繪製 SHAP force plot
                                st.write(f"**對於索引 `{selected_index}`，類別 `{single_prediction_label}` 的解釋：**")
                                fig, ax = plt.subplots(figsize=(20, 4))
                                shap.force_plot(
                                    shap_base_value,
                                    shap_values_for_class,
                                    single_instance.iloc[0],
                                    matplotlib=True,
                                    show=False,
                                    text_rotation=15
                                )
                                plt.tight_layout()
                                st.pyplot(fig, bbox_inches='tight')
                                plt.close(fig)

                            except KeyError:
                                st.error(f"發生錯誤：無法在已處理的資料中找到索引 {selected_index}。這可能是因為該筆資料在上傳後因包含無效值而被移除。請嘗試選擇另一筆流量。")
                            except Exception as e:
                                st.warning(f"無法產生 SHAP 分析圖：{e}")

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