import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import json
import textwrap
from datetime import datetime, timedelta

# --- 基础库检查 ---
try:
    import yfinance as yf
    import numpy as np
    import s3fs # 新增：引入 S3 文件系统库
except ImportError as e:
    st.error(f"缺少必要库，请先安装: {e}\n请确保已执行: pip install s3fs yfinance plotly pandas")
    st.stop()

# --- 页面设置 ---
st.set_page_config(page_title="股价复盘 (云端同步版)", layout="wide")

# --- 云端连接初始化 ---
# 尝试从 secrets 获取 AWS 配置
if "aws" in st.secrets:
    try:
        # 初始化 S3 文件系统
        fs = s3fs.S3FileSystem(
            key=st.secrets["aws"]["aws_access_key_id"],
            secret=st.secrets["aws"]["aws_secret_access_key"]
        )
        BUCKET_NAME = st.secrets["aws"]["bucket_name"]
        HISTORY_DIR = f"{BUCKET_NAME}/history_charts"
        
        # 确保云端目录存在 (S3其实是平铺的，这步主要是检查权限)
        if not fs.exists(HISTORY_DIR):
            fs.makedirs(HISTORY_DIR)
            
        USE_CLOUD = True
    except Exception as e:
        st.error(f"AWS S3 连接失败，将无法保存到云端: {e}")
        USE_CLOUD = False
else:
    st.warning("⚠️ 未检测到 [.streamlit/secrets.toml] 配置，请配置 AWS 密钥以启用云同步。")
    USE_CLOUD = False


# --- 辅助函数定义 ---
def process_text_smart(text, wrap_width):
    if not isinstance(text, str): return str(text)
    lines = text.split('\n')
    processed_lines = []
    for line in lines:
        line = line.strip()
        if not line: continue
        line = line.replace("<br>", "\n")
        sub_lines = line.split("\n")
        for sl in sub_lines:
            wrapped = textwrap.wrap(sl, width=wrap_width)
            processed_lines.extend(wrapped)
    return "<br>".join(processed_lines)

def generate_mock_data(start, end):
    dates = pd.date_range(start=start, end=end, freq='B')
    n = len(dates)
    if n == 0: return None
    np.random.seed(42)
    returns = np.random.normal(loc=0.0003, scale=0.015, size=n)
    price = 3000 * np.cumprod(1 + returns)
    df = pd.DataFrame(index=dates)
    df['Close'] = price
    df['Open'] = df['Close'].shift(1).fillna(price[0]) * (1 + np.random.randn(n)*0.005)
    return df.round(0)

def load_data_from_excel(file):
    try:
        df = pd.read_excel(file, sheet_name='Prices')
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        return df
    except: 
        return None

def get_stock_data(source, ticker, start, end, uploaded_file):
    if source == "Yahoo Finance (实盘数据)":
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        try:
            with st.spinner("正在连接 Yahoo..."):
                dat = yf.Ticker(ticker)
                df = dat.history(start=start_str, end=end_str, auto_adjust=True)
            if df.empty:
                st.error("❌ Yahoo 返回空数据")
                return None
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            st.error(f"连接失败: {e}")
            return None
    elif source == "Excel文件 (Prices表)":
        return load_data_from_excel(uploaded_file) if uploaded_file else None
    else:
        return generate_mock_data(start, end)

def find_col_in_list(columns, keywords, exclude_keywords=None):
    for col in columns:
        col_str = str(col)
        if exclude_keywords and any(ex in col_str for ex in exclude_keywords):
            continue
        for kw in keywords:
            if kw in col_str:
                return col
    return None

def extract_table_dynamically(df, required_keywords, name="Table"):
    def check_columns(cols):
        found_cols = {}
        for key, (kws, ex_kws) in required_keywords.items():
            found = find_col_in_list(cols, kws, ex_kws)
            if found:
                found_cols[key] = found
            else:
                return None
        return found_cols

    found_cols = check_columns(df.columns)
    if found_cols:
        return df, found_cols

    max_scan = min(len(df), 100)
    for i in range(max_scan):
        row_values = df.iloc[i].astype(str).tolist()
        is_header_row = True
        for key, (kws, ex_kws) in required_keywords.items():
            if not any(kw in cell for cell in row_values for kw in kws):
                is_header_row = False
                break
        
        if is_header_row:
            new_df = df.iloc[i+1:].copy()
            new_df.columns = df.iloc[i]
            new_found_cols = check_columns(new_df.columns)
            if new_found_cols:
                return new_df, new_found_cols
    
    return None, None

def aggregate_details(df, group_keys, detail_col, output_detail_name="Detail"):
    if not detail_col: return df
    for k in group_keys:
        df[k] = df[k].ffill()
    
    def join_text(series):
        texts = [str(s).strip() for s in series if pd.notna(s) and str(s).strip() != '']
        if not texts: return None
        if len(texts) == 1: return texts[0]
        return "<br>".join([f"• {t}" for t in texts])

    agg_dict = {detail_col: join_text}
    temp = df.groupby(group_keys, as_index=False).agg(agg_dict)
    temp = temp.rename(columns={detail_col: output_detail_name})
    return temp

def parse_uploaded_excel(file):
    try:
        all_sheets = pd.read_excel(file, sheet_name=None)
        events_list = []
        phases_list = []
        
        event_rules = {
            'event': (['主要驱动', 'Event'], None),
            'date': (['日期', 'Date', '时间'], ['起始', '开始', 'Start', '结束', 'End'])
        }
        
        phase_rules = {
            'phase': (['阶段概述', 'Phase'], None),
            'start': (['起始日期', '开始日期', 'Start'], None),
            'end': (['结束日期', 'End'], None)
        }

        for sheet_name, df in all_sheets.items():
            df.columns = df.columns.astype(str).str.strip()
            
            # 1. 提取事件表
            e_df, e_cols = extract_table_dynamically(df, event_rules, "Events")
            if e_df is not None:
                hover_col = find_col_in_list(e_df.columns, ['详细解释', '因果链', 'Detailed'])
                cols_to_keep = [e_cols['date'], e_cols['event']]
                if hover_col: cols_to_keep.append(hover_col)
                temp = e_df[cols_to_keep].copy()
                
                if hover_col:
                    temp = aggregate_details(temp, group_keys=[e_cols['date'], e_cols['event']], detail_col=hover_col, output_detail_name='详细解释')
                
                temp = temp.rename(columns={e_cols['date']: 'Date', e_cols['event']: '主要驱动'})
                temp['Date'] = pd.to_datetime(temp['Date'], errors='coerce')
                temp = temp.dropna(subset=['Date'])
                if not temp.empty:
                    events_list.append(temp)
            
            # 2. 提取阶段表
            p_df, p_cols = extract_table_dynamically(df, phase_rules, "Phases")
            if p_df is not None:
                hover_col = find_col_in_list(p_df.columns, ['关键因素', '要点', 'Key Factors'])
                cols_to_keep = [p_cols['start'], p_cols['end'], p_cols['phase']]
                if hover_col: cols_to_keep.append(hover_col)
                temp = p_df[cols_to_keep].copy()
                
                if hover_col:
                    temp = aggregate_details(temp, group_keys=[p_cols['start'], p_cols['end'], p_cols['phase']], detail_col=hover_col, output_detail_name='关键因素')
                
                temp = temp.rename(columns={p_cols['start']: 'Start date', p_cols['end']: 'End date', p_cols['phase']: '阶段概述'})
                temp['Start date'] = pd.to_datetime(temp['Start date'], errors='coerce')
                temp['End date'] = pd.to_datetime(temp['End date'], errors='coerce')
                temp = temp.dropna(subset=['Start date'])
                if not temp.empty:
                    phases_list.append(temp)

        events_df = pd.concat(events_list, ignore_index=True) if events_list else None
        phases_df = pd.concat(phases_list, ignore_index=True) if phases_list else None
        return events_df, phases_df

    except Exception as e:
        import traceback
        st.error(f"解析 Excel 出错: {e}")
        st.text(traceback.format_exc())
        return None, None

# ==============================================================================
# 主程序入口
# ==============================================================================

st.sidebar.title("🎛️ 系统模式")
app_mode = st.sidebar.radio("选择功能", ["🚀 生成新图表", "📂 浏览历史记录 (云端)"])

if app_mode == "🚀 生成新图表":
    st.title("📈 2025 股价复盘系统：云端智能版")
    st.markdown("---")

    # --- 0. 代理设置 ---
    st.sidebar.header("0. 网络代理设置")
    enable_proxy = st.sidebar.checkbox("开启代理连接", value=True)
    proxy_address = st.sidebar.text_input("代理地址", value="http://127.0.0.1:17890")
    if enable_proxy:
        os.environ["HTTP_PROXY"] = proxy_address
        os.environ["HTTPS_PROXY"] = proxy_address
    else:
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)

    # --- 1. 数据来源 ---
    st.sidebar.header("1. 数据来源")
    data_source = st.sidebar.radio("选择模式", ["Yahoo Finance (实盘数据)", "Excel文件 (Prices表)", "生成模拟数据 (测试用)"])

    # --- 2. 绘图参数 ---
    st.sidebar.header("2. 绘图参数")
    default_start = pd.to_datetime("2024-12-23")
    default_end = min(pd.to_datetime("2025-12-23"), datetime.today())
    ticker = st.sidebar.text_input("股票代码", value="6324.T")
    start_date = st.sidebar.date_input("开始日期", value=default_start)
    end_date_input = st.sidebar.date_input("结束日期", value=default_end, max_value=datetime.today())
    end_date_final = end_date_input + timedelta(days=1)

    # --- 3. 视觉与排版微调 ---
    st.sidebar.header("3. 视觉与排版微调")
    export_scale = st.sidebar.radio("导出清晰度/倍率", [1, 2, 3], index=0, format_func=lambda x: f"{x}倍", horizontal=True)
    phase_font_size = st.sidebar.slider("顶部阶段字体大小", 10, 80, 20)
    event_font_size = st.sidebar.slider("下方事件字体大小", 8, 60, 16)
    phase_label_y = st.sidebar.slider("阶段标签基础高度", 1.0, 1.3, 1.02, 0.01)
    phase_stagger = st.sidebar.checkbox("开启顶部标签错落", value=True)
    phase_stagger_gap = st.sidebar.slider("顶部错落高度差", 0.01, 0.15, 0.05)
    label_wrap_width = st.sidebar.slider("标签换行字数", 5, 30, 10)
    hover_wrap_width = st.sidebar.slider("悬浮文字换行字数", 20, 80, 40)
    arrow_len_base = st.sidebar.slider("引线基础长度", 20, 150, 50)
    stagger_steps = st.sidebar.slider("下方防重叠阶梯数", 3, 10, 6)
    stagger_gap = st.sidebar.slider("下方阶梯垂直间距", 10, 100, 50)
    y_headroom = st.sidebar.slider("顶部强制留白 (%)", 0, 100, 7)
    bg_opacity = st.sidebar.slider("标签背景透明度", 0.1, 1.0, 0.8)
    bottom_margin = st.sidebar.slider("底部留白高度", 50, 150, 80)
    top_margin = st.sidebar.slider("顶部留白高度", 100, 300, 150)

    # --- 4. 上传文件 ---
    st.sidebar.header("4. 上传文件")
    uploaded_file = st.sidebar.file_uploader("上传 Excel (中文版)", type=["xlsx"])

    # --- 核心处理逻辑 ---
    if uploaded_file or data_source == "生成模拟数据 (测试用)":
        stock_df = get_stock_data(data_source, ticker, start_date, end_date_final, uploaded_file)
        
        if stock_df is not None and not stock_df.empty:
            events_df, phases_df = None, None
            if uploaded_file:
                events_df, phases_df = parse_uploaded_excel(uploaded_file)
            
            if uploaded_file and events_df is None and phases_df is None:
                st.warning("⚠️ 未能识别Excel内容。")
            else:
                try:
                    fig = go.Figure()
                    # 1. 绘制股价
                    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Close'], mode='lines', name=f"{ticker} 收盘价", line=dict(color='#1976D2', width=2.5), line_shape='spline'))
                    data_start, data_end = stock_df.index.min(), stock_df.index.max()

                    # 2. 绘制阶段
                    if phases_df is not None and not phases_df.empty:
                        phase_colors = ["rgba(255,99,132,0.12)", "rgba(54,162,235,0.12)", "rgba(255,206,86,0.15)", "rgba(75,192,192,0.12)"]
                        target_col = find_col_in_list(phases_df.columns, ['阶段概述'])
                        for i, row in phases_df.iterrows():
                            p_start = max(row['Start date'], data_start)
                            p_end = min(row['End date'], data_end)
                            if p_start < p_end:
                                mid_point = p_start + (p_end - p_start) / 2
                                fig.add_vrect(x0=p_start, x1=p_end, fillcolor=phase_colors[i % 4], layer="below", line_width=0)
                                raw_text = str(row.get(target_col, ''))
                                wrapped_text = process_text_smart(raw_text, label_wrap_width)
                                hover_col = find_col_in_list(phases_df.columns, ['关键因素', '要点', 'Key Factors'])
                                hover_text_raw = str(row.get(hover_col, '')) if hover_col else raw_text
                                hover_text = process_text_smart(hover_text_raw, hover_wrap_width)
                                current_phase_y = phase_label_y
                                if phase_stagger: current_phase_y += (i % 2) * phase_stagger_gap
                                fig.add_annotation(x=mid_point, y=current_phase_y, yref="paper", text=f"<b>{wrapped_text}</b>", hovertext=hover_text, showarrow=False, font=dict(size=phase_font_size, color="#555"), bgcolor="rgba(255,255,255,0.8)", borderpad=3, captureevents=True)

                    # 3. 绘制事件
                    if events_df is not None and not events_df.empty:
                        events_df = events_df.sort_values('Date').reset_index(drop=True)
                        label_col = find_col_in_list(events_df.columns, ['主要驱动'])
                        for i, row in events_df.iterrows():
                            event_date = row['Date']
                            if data_start <= event_date <= data_end:
                                try:
                                    idx = stock_df.index.get_indexer([event_date], method='nearest')[0]
                                    curr = stock_df.index[idx]
                                    vals = stock_df.loc[curr]
                                    close_p = vals['Close'].iloc[0] if isinstance(vals['Close'], pd.Series) else vals['Close']
                                    open_p = vals['Open'].iloc[0] if isinstance(vals['Open'], pd.Series) else vals['Open']
                                    y_anchor = close_p
                                    is_rising = close_p >= open_p
                                    ay_dir = 1 if is_rising else -1
                                    color = "#D32F2F" if is_rising else "#00796B"
                                    stagger_level = i % stagger_steps 
                                    current_arrow_len = arrow_len_base + (stagger_level * stagger_gap)
                                    txt = str(row.get(label_col, ''))
                                    formatted = process_text_smart(txt, label_wrap_width)
                                    hover_col = find_col_in_list(events_df.columns, ['详细解释', '因果链', 'Detailed'])
                                    hover_text_raw = str(row.get(hover_col, '')) if hover_col else txt
                                    hover_formatted = process_text_smart(hover_text_raw, hover_wrap_width)
                                    fig.add_annotation(x=curr, y=y_anchor, text=f"<b>{formatted}</b>", hovertext=hover_formatted, showarrow=True, arrowhead=2, arrowwidth=1.5, arrowcolor=color, ax=0, ay=current_arrow_len * ay_dir, font=dict(size=event_font_size, color="#333"), bgcolor=f"rgba(255,255,255,{bg_opacity})", bordercolor=color, borderwidth=1, borderpad=3, hoverlabel=dict(bgcolor="white", font=dict(size=event_font_size)), captureevents=True)
                                except: pass

                    # 4. 布局
                    y_max = stock_df['Close'].max()
                    y_min = stock_df['Close'].min()
                    range_max = y_max * (1 + y_headroom / 100)
                    range_min = y_min * 0.95
                    fig.update_layout(title=dict(text=f"{ticker} 收盘价趋势复盘", x=0.5, font=dict(size=22)), yaxis_title="收盘价 (JPY)", height=950, xaxis_rangeslider_visible=False, template="plotly_white", margin=dict(t=top_margin, r=50, b=bottom_margin), plot_bgcolor='rgba(250,250,250,1)', hovermode="x unified", dragmode="pan")
                    fig.update_xaxes(tickformat="%y年%-m月", dtick="M1", showgrid=True, gridcolor='rgba(0,0,0,0.05)')
                    fig.update_yaxes(range=[range_min, range_max], showgrid=True, gridcolor='rgba(0,0,0,0.05)')

                    st.plotly_chart(fig, use_container_width=True, config={'editable': True, 'scrollZoom': True, 'toImageButtonOptions': {'format': 'png', 'filename': f'{ticker}_复盘分析', 'height': 950 * export_scale, 'width': 1600 * export_scale, 'scale': 1}})

                    # === 新增：云端保存功能 ===
                    st.markdown("### 💾 保存到云端")
                    col_save_1, col_save_2 = st.columns([3, 1])
                    with col_save_1:
                        save_name = st.text_input("输入保存名称", placeholder="例如：特斯拉2024复盘_V1")
                    with col_save_2:
                        st.write("") 
                        st.write("") 
                        if st.button("☁️ 同步到云端", type="primary"):
                            if not USE_CLOUD:
                                st.error("❌ 未配置 AWS，无法上传。请检查 secrets.toml")
                            else:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                safe_name = "".join([c for c in save_name if c.isalnum() or c in (' ', '_', '-')]).strip()
                                filename = f"{timestamp}_{ticker}_{safe_name}.json" if safe_name else f"{timestamp}_{ticker}.json"
                                cloud_path = f"{HISTORY_DIR}/{filename}"
                                
                                try:
                                    with st.spinner("🚀 正在上传到 AWS S3..."):
                                        json_str = fig.to_json()
                                        with fs.open(cloud_path, "w") as f:
                                            f.write(json_str)
                                    st.success(f"✅ 云端同步成功: {filename}")
                                except Exception as e:
                                    st.error(f"上传失败: {e}")

                except Exception as e:
                    import traceback
                    st.error(f"绘图报错: {e}")
                    st.text(traceback.format_exc())
        else:
            if data_source != "Yahoo Finance (实盘数据)": st.warning("⚠️ 数据为空")
    else:
        st.info("👈 请上传 Excel 文件或选择模拟数据模式")

# ==============================================================================
# 历史记录查看模式 (云端版)
# ==============================================================================
elif app_mode == "📂 浏览历史记录 (云端)":
    st.title("☁️ 云端图表档案馆")
    st.markdown("---")
    
    if not USE_CLOUD:
        st.error("❌ 未连接 AWS S3。请在 .streamlit/secrets.toml 中配置密钥。")
        st.stop()

    st.sidebar.header("🔍 查找与筛选")
    
    # 1. 获取所有云端文件
    try:
        # fs.glob 返回的是完整路径 list
        raw_files = fs.glob(f"{HISTORY_DIR}/*.json")
        
        # 获取文件详细信息以便按时间排序
        file_details = []
        for f_path in raw_files:
            info = fs.info(f_path)
            # S3 info 包含 LastModified
            file_details.append({
                'path': f_path,
                'name': os.path.basename(f_path),
                'time': info.get('LastModified', datetime.now())
            })
        
        # 按时间倒序排列
        file_details.sort(key=lambda x: x['time'], reverse=True)
        
        if not file_details:
            st.info("📭 云端暂无记录。请在“生成新图表”模式下保存。")
        else:
            # 2. 搜索功能
            search_term = st.sidebar.text_input("搜索文件名/股票代码", "")
            
            # 过滤
            filtered_files = [f for f in file_details if search_term.lower() in f['name'].lower()]
            
            if not filtered_files:
                st.warning("没有找到匹配的文件。")
            else:
                # 3. 选择文件
                # 制作下拉菜单选项：包含更友好的时间显示
                options_map = {f['name']: f for f in filtered_files}
                selected_name = st.sidebar.selectbox("选择要查看的图表", list(options_map.keys()))
                
                if selected_name:
                    selected_obj = options_map[selected_name]
                    full_path = selected_obj['path']
                    
                    # 友好的标题显示
                    st.caption(f"📅 上次修改: {selected_obj['time']} | 📄 文件: {selected_name}")
                    
                    # 4. 加载并展示
                    try:
                        with st.spinner("正在从云端下载..."):
                            with fs.open(full_path, 'r') as f:
                                fig_json = json.load(f)
                        
                        loaded_fig = go.Figure(fig_json)
                        st.plotly_chart(loaded_fig, use_container_width=True, config={
                            'scrollZoom': True,
                            'toImageButtonOptions': {'format': 'png', 'filename': selected_name.replace('.json', '')}
                        })
                        
                        st.markdown("---")
                        # 5. 删除功能
                        if st.button("🗑️ 从云端删除此记录"):
                            try:
                                fs.rm(full_path)
                                st.success("✅ 已删除，请刷新页面。")
                                st.rerun()
                            except Exception as e:
                                st.error(f"删除失败: {e}")
                                
                    except Exception as e:
                        st.error(f"无法读取文件: {e}")
                        
    except Exception as e:
        st.error(f"读取文件列表失败: {e}")