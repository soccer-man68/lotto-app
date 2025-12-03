import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import time

# -----------------------------------------------------------------------------
# [0] 페이지 설정
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AI 로또 분석기 (Web)", layout="wide")
st.title("🎱 AI 로또 번호 추출기 (Pro)")

# -----------------------------------------------------------------------------
# [1] 데이터 로드 및 AI 학습 (캐싱 적용으로 속도 향상)
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_preprocess_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file, header=None)
        raw_data = df.iloc[:, 1:] # 1회차부터 데이터가 있다고 가정
        numeric_data = raw_data.apply(pd.to_numeric, errors='coerce')
        # 1~45 사이 숫자만 필터링
        df_clean = numeric_data.where(numeric_data.ge(1) & numeric_data.le(45))
        
        if len(df_clean) < 50:
            return None, "데이터 부족 (최소 50회차 이상 필요)"
            
        all_draws = [row.dropna().astype(int).tolist() for _, row in df_clean.iterrows()]
        # 궁합수 계산
        co_occurrence = Counter(pair for draw in all_draws for pair in combinations(sorted(draw), 2))
        
        return (df_clean, co_occurrence), "성공"
    except Exception as e:
        return None, f"오류 발생: {e}"

@st.cache_resource
def train_ai_model(df):
    # 피처 엔지니어링
    features = []
    for index, row in df.iterrows():
        draw = row.dropna().astype(int).tolist()
        if len(draw) < 6: continue
        features.append({
            'sum': sum(draw),
            'mean': sum(draw)/6,
            'std': pd.Series(draw).std(),
            'odd_count': sum(1 for n in draw if n % 2 != 0),
            'low_count': sum(1 for n in draw if 1 <= n <= 22),
            'ends_unique': len({n % 10 for n in draw})
        })
    
    if not features: return None
    
    X = pd.DataFrame(features)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(X)
    
    # KMeans 클러스터링
    kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_features)
    
    # 패턴 전이 학습
    transitions = {i: Counter() for i in range(7)}
    for i in range(len(labels) - 1):
        transitions[labels[i]][labels[i+1]] += 1
        
    return kmeans, scaler, labels, transitions

# -----------------------------------------------------------------------------
# [2] 예측 로직 함수 모음
# -----------------------------------------------------------------------------
def predict_by_total_frequency(df, count=15):
    return pd.Series(df.dropna().values.flatten().astype(int)).value_counts().head(count).index.tolist()

def predict_by_recent_frequency(df, weeks=10, count=15):
    return pd.Series(df.tail(weeks).dropna().values.flatten().astype(int)).value_counts().head(count).index.tolist()

def predict_by_weighted_recent(df, span=20, count=15):
    scores = defaultdict(float)
    weights = np.exp(np.linspace(0, 1, len(df.tail(span)))) 
    recent_data = df.tail(span)
    for i, (_, row) in enumerate(recent_data.iterrows()):
        w = weights[i]
        for num in row.dropna().astype(int):
            scores[num] += w
    return [num for num, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:count]]

def predict_by_neighbors(df, count=15):
    last_draw = df.iloc[-1].dropna().astype(int).tolist()
    neighbors = set()
    for n in last_draw:
        if n > 1: neighbors.add(n - 1)
        if n < 45: neighbors.add(n + 1)
    neighbors = neighbors - set(last_draw)
    return list(neighbors)

def predict_by_long_term_unappeared(df, count=15):
    last_appeared = {num: -1 for num in range(1, 46)}
    for index, row in df.iterrows():
        for num in row.dropna().astype(int): last_appeared[num] = index
    return [num for num, idx in sorted(last_appeared.items(), key=lambda item: item[1])[:count]]

def predict_by_good_compatibility(df, co_counts, count=15):
    last_draw = df.iloc[-1].dropna().astype(int).tolist()
    scores = {n: sum(co_counts.get(tuple(sorted((n, wn))), 0) for wn in last_draw) for n in range(1, 46) if n not in last_draw}
    return [num for num, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:count]]

def predict_by_strongest_pairs(co_counts, count=5):
    return sorted({num for pair, freq in co_counts.most_common(count) for num in pair})

def predict_by_number_temperature(df, count=15):
    all_nums = pd.Series(df.dropna().values.flatten().astype(int))
    recent_nums = pd.Series(df.tail(10).dropna().values.flatten().astype(int))
    total_freq = all_nums.value_counts(normalize=True)
    recent_freq = recent_nums.value_counts(normalize=True)
    last_appeared = {num: len(df) for num in range(1, 46)}
    for i, row in df.iterrows():
        for num in row.dropna().astype(int): last_appeared[num] = i
    unappeared_period = pd.Series({num: len(df) - idx for num, idx in last_appeared.items()})
    if unappeared_period.max() > 0: unappeared_period /= unappeared_period.max()
    temp_scores = pd.Series({n: 0 for n in range(1, 46)})
    temp_scores = temp_scores.add(total_freq * 0.5, fill_value=0).add(recent_freq * 2.0, fill_value=0).add(unappeared_period * 0.8, fill_value=0)
    return temp_scores.sort_values(ascending=False).head(count).index.tolist()

def predict_by_positional_frequency(df, count_per_pos=5):
    sorted_draws = [sorted(r.dropna().astype(int).tolist()) for _, r in df.iterrows() if len(r.dropna()) == 6]
    if not sorted_draws: return []
    pos_counters = [Counter(col) for col in zip(*sorted_draws)]
    return sorted(list({num for c in pos_counters for num, _ in c.most_common(count_per_pos)}))

def predict_by_volatility_vector(df, anchor_count=5):
    all_vectors = [ [d[i+1]-d[i] for i in range(5)] for d in [sorted(r.dropna().astype(int).tolist()) for _, r in df.iterrows()] if len(d) == 6]
    if not all_vectors: return []
    avg_vector = [round(sum(col) / len(all_vectors)) for col in zip(*all_vectors)]
    anchor_points = predict_by_long_term_unappeared(df, anchor_count)
    predictions = set()
    for anchor in anchor_points:
        combo, num, is_valid = [anchor], anchor, True
        for interval in avg_vector:
            num += interval
            if num > 45: is_valid = False; break
            combo.append(num)
        if is_valid: predictions.update(combo)
    return sorted(list(predictions))

def predict_by_consecutive_pattern(df, weeks=3):
    last_draw = df.iloc[-1].dropna().astype(int).tolist()
    candidates = set()
    for n in last_draw:
        candidates.add(n-1); candidates.add(n+1)
    return sorted(list({n for n in candidates if 1 <= n <= 45} - set(last_draw)))

def predict_by_reappearing_number(df):
    return df.iloc[-1].dropna().astype(int).tolist()

def predict_by_prime_numbers(df, weeks=5):
    primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43}
    return sorted(list(primes))

def predict_by_number_zones(df, weeks=5):
    zones = {0: list(range(1, 11)), 1: list(range(11, 21)), 2: list(range(21, 31)), 3: list(range(31, 41)), 4: list(range(41, 46))}
    zone_counts = {i: 0 for i in range(5)}
    for num in df.tail(weeks).dropna().values.flatten().astype(int):
        for zid, nums in zones.items():
            if num in nums: zone_counts[zid] += 1; break
    min_count = min(zone_counts.values())
    return sorted([num for zid, cnt in zone_counts.items() if cnt == min_count for num in zones[zid]])

def predict_by_regression_cycle(df):
    predictions = []
    total_draws = len(df)
    for num in range(1, 46):
        appearances = df[df.isin([num])].count().max()
        if appearances < 2: continue
        avg_cycle = total_draws / appearances
        last_appeared_index = df.where(df == num).last_valid_index()
        if last_appeared_index is None: continue
        if total_draws - last_appeared_index > avg_cycle: predictions.append(num)
    return predictions

def predict_by_cluster_transition(df, model, scaler, labels, transitions, count=15):
    if any(x is None for x in [df, model, scaler, labels, transitions]): return []
    try:
        # 최근 데이터를 피처로 변환
        features = []
        draw = df.tail(1).dropna().values.flatten().astype(int).tolist()
        features.append({
            'sum': sum(draw), 'mean': sum(draw)/6, 'std': pd.Series(draw).std(),
            'odd_count': sum(1 for n in draw if n % 2 != 0),
            'low_count': sum(1 for n in draw if 1 <= n <= 22),
            'ends_unique': len({n % 10 for n in draw})
        })
        last_cluster = model.predict(scaler.transform(pd.DataFrame(features)))[0]
        
        if not transitions[last_cluster]: return []
        next_cluster = transitions[last_cluster].most_common(1)[0][0]
        member_indices = [i for i, label in enumerate(labels) if label == next_cluster]
        if not member_indices: return []
        return pd.Series(df.iloc[member_indices].dropna().values.flatten().astype(int)).value_counts().head(count).index.tolist()
    except:
        return []

# -----------------------------------------------------------------------------
# [3] 유틸리티 및 조합 탐색
# -----------------------------------------------------------------------------
LOGIC_WEIGHTS = {
    1: 1.0, 2: 1.5, 3: 1.0, 4: 1.2, 5: 0.8, 6: 2.0, 7: 1.0, 8: 1.2,
    9: 1.0, 10: 1.0, 11: 0.5, 12: 0.8, 13: 1.0, 14: 2.5, 15: 1.5, 16: 1.5,
}

logic_info = [
    (1, "전체 기간 빈도 상위"), (2, "최근 10주 빈도 상위"), (3, "장기 미출수"), (4, "궁합수 (vs 직전회차)"),
    (5, "궁합수 (최강 조합)"), (6, "[핵심] 숫자 온도 분석"), (7, "[신규] 위치별 빈도"), (8, "[획기적] 변동성 벡터"),
    (9, "[신규] 연속수 패턴"), (10, "이월수 (직전 번호)"), (11, "[신규] 소수(Prime) 패턴"), (12, "[신규] 번호대(Zone) 분석"),
    (13, "회귀 주기 분석"), (14, "[강력] 가중치 최근 빈도"), (15, "[강력] 이웃수 분석"), (16, "[AI] 군집 전환 패턴")
]

def select_final_numbers(score_board, count):
    sorted_scores = sorted(score_board.items(), key=lambda x: x[1], reverse=True)
    all_candidates = [num for num, s in sorted_scores]
    if len(all_candidates) <= count: return all_candidates
    
    n_hot = int(count * 0.6)
    n_warm = int(count * 0.2)
    n_cold = count - n_hot - n_warm
    
    final_set = set()
    final_set.update(all_candidates[:n_hot])
    
    mid_start = n_hot
    mid_end = mid_start + n_warm + 5
    warm_pool = all_candidates[mid_start:mid_end]
    final_set.update(warm_pool[:n_warm])
    
    if n_cold > 0:
        cold_start = mid_end
        cold_pool = all_candidates[cold_start:cold_start+10]
        final_set.update(cold_pool[:n_cold])
        
    return sorted(list(final_set))

# -----------------------------------------------------------------------------
# [4] 메인 UI
# -----------------------------------------------------------------------------
def main():
   # === [비밀번호 보안 기능 시작] ===
    # 여기에 원하는 비밀번호를 적으세요 (예: "1234")
    my_password = "4938"
    
    # 사이드바에 비밀번호 입력창 만들기
    input_pw = st.sidebar.text_input("🔒 비밀번호를 입력하세요", type="password")
    
    if input_pw != my_password:
        st.sidebar.warning("비밀번호가 틀렸거나 입력되지 않았습니다.")
        st.stop()  # 비밀번호가 틀리면 여기서 프로그램 중단 (아래 내용 안 보여줌)
    # === [비밀번호 보안 기능 끝] ===
    st.sidebar.header("📁 데이터 및 설정")
    uploaded_file = st.sidebar.file_uploader("로또 엑셀 파일 업로드", type=['xlsx', 'xls'])
    
    # 템플릿 파일 다운로드 제공 (선택사항)
    # st.sidebar.download_button("엑셀 양식 다운로드", ...) 

    if uploaded_file is not None:
        with st.spinner("데이터 분석 및 AI 학습 중..."):
            data_result, msg = load_and_preprocess_data(uploaded_file)
            
            if data_result is None:
                st.error(msg)
                return
            
            df, co_occurrence = data_result
            model_pack = train_ai_model(df)
            
            st.sidebar.success(f"데이터 로드 완료! (총 {len(df)}회차)")
            st.sidebar.info("AI 모델 학습 완료 (K-Means)")
    else:
        st.info("왼쪽 사이드바에서 로또 당첨번호 엑셀 파일을 업로드해주세요.")
        return

    # 탭 구성
    tab1, tab2 = st.tabs(["🤖 AI 자동 리포트 (추천)", "🎲 수동 예측 생성"])
    
    # --- [탭 1] AI 자동 리포트 ---
    with tab1:
        st.subheader("📊 AI 최적/최악 조합 분석 리포트")
        st.caption("최근 50회차(약 1년) 트렌드를 기반으로 가장 성적이 좋은 로직 조합을 자동으로 찾습니다.")
        
        target_count = st.slider("추출할 번호 개수 (최적화 기준)", 3, 15, 10, key="slider_auto")
        
        if st.button("🚀 AI 분석 시작 (시간 소요됨)"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 1. 모든 로직 결과 미리 계산 (속도 최적화)
            precalculated = {}
            model, scaler, labels, transitions = model_pack
            
            current_logics = {
                1: (predict_by_total_frequency, (df,)),
                2: (predict_by_recent_frequency, (df,)),
                3: (predict_by_long_term_unappeared, (df,)),
                4: (predict_by_good_compatibility, (df, co_occurrence)),
                5: (predict_by_strongest_pairs, (co_occurrence,)),
                6: (predict_by_number_temperature, (df,)),
                7: (predict_by_positional_frequency, (df,)),
                8: (predict_by_volatility_vector, (df,)),
                9: (predict_by_consecutive_pattern, (df,)),
                10: (predict_by_reappearing_number, (df,)),
                11: (predict_by_prime_numbers, (df,)),
                12: (predict_by_number_zones, (df,)),
                13: (predict_by_regression_cycle, (df,)),
                14: (predict_by_weighted_recent, (df,)),
                15: (predict_by_neighbors, (df,)),
                16: (predict_by_cluster_transition, (df, model, scaler, labels, transitions)),
            }
            
            for i in range(1, 17):
                func, args = current_logics[i]
                precalculated[i] = func(*args)
            
            # 2. 조합 탐색 (Streamlit Timeout 방지를 위해 탐색 범위 축소 적용)
            past_draws = [set(row.dropna().astype(int)) for _, row in df.tail(50).iterrows()]
            all_logic_indices = list(range(1, 17))
            
            best_score = -1
            best_combo = []
            worst_score = float('inf')
            worst_combo = []
            
            # 랜덤 샘플링 방식으로 최적화 (서버 부하 방지)
            import random
            random.seed(42)
            
            # 전체 조합 중 1000개만 샘플링하여 테스트 (속도 vs 정확도 타협)
            # 클라우드 무료 서버는 30초 이상 걸리면 멈추므로 이 방식이 안전함
            status_text.text("조합 시뮬레이션 중... (서버 최적화 모드)")
            
            sample_combos = []
            for r in range(3, 8): # 로직 3개~7개 조합만 봄 (너무 많으면 과적합)
                 combos = list(combinations(all_logic_indices, r))
                 if len(combos) > 200:
                     sample_combos.extend(random.sample(combos, 200))
                 else:
                     sample_combos.extend(combos)
            
            total_steps = len(sample_combos)
            
            for idx, combo_indices in enumerate(sample_combos):
                score_board = defaultdict(float)
                for l_idx in combo_indices:
                    candidates = precalculated[l_idx]
                    weight = LOGIC_WEIGHTS.get(l_idx, 1.0)
                    for rank, num in enumerate(candidates):
                        score_board[num] += (weight + (0.5 if rank < 5 else 0))
                
                final_numbers = set(select_final_numbers(score_board, target_count))
                
                score = 0
                for draw in past_draws:
                    matches = len(final_numbers.intersection(draw))
                    if matches >= 3: score += (10 ** (matches - 2)) # 3개:10, 4개:100, 5개:1000...
                
                if score > best_score:
                    best_score = score
                    best_combo = combo_indices
                if score < worst_score:
                    worst_score = score
                    worst_combo = combo_indices
                
                if idx % 100 == 0:
                    progress_bar.progress(idx / total_steps)
            
            progress_bar.progress(1.0)
            status_text.success("분석 완료!")
            
            # 3. 결과 출력
            st.divider()
            
            # BEST 결과
            best_names = [next(name for i, name in logic_info if i == idx) for idx in best_combo]
            st.write(f"### 🏆 최적(Best) 조합")
            st.info(f"**사용된 로직:** {', '.join(best_names)}")
            
            best_score_board = defaultdict(float)
            for l_idx in best_combo:
                candidates = precalculated[l_idx]
                for rank, num in enumerate(candidates):
                    best_score_board[num] += (LOGIC_WEIGHTS.get(l_idx,1) + (0.5 if rank < 5 else 0))
            
            prediction = select_final_numbers(best_score_board, target_count)
            st.success(f"**추천 번호:** {sorted(prediction)}")
            
            # WORST 결과
            worst_names = [next(name for i, name in logic_info if i == idx) for idx in worst_combo]
            st.write(f"### ☠️ 최악(Worst) 조합 (제외수 추천)")
            st.warning(f"**사용된 로직:** {', '.join(worst_names)}")
            
            worst_score_board = defaultdict(float)
            for l_idx in worst_combo:
                candidates = precalculated[l_idx]
                for rank, num in enumerate(candidates):
                    worst_score_board[num] += (LOGIC_WEIGHTS.get(l_idx,1) + (0.5 if rank < 5 else 0))
            
            exclusion = select_final_numbers(worst_score_board, target_count)
            st.error(f"**제외 추천 번호:** {sorted(exclusion)}")

    # --- [탭 2] 수동 예측 ---
    with tab2:
        st.subheader("🛠️ 로직 직접 선택")
        
        cols = st.columns(2)
        selected_logics = []
        for i, (idx, name) in enumerate(logic_info):
            col = cols[0] if i < 8 else cols[1]
            if col.checkbox(name, value=(i in [0, 1, 5, 6, 7, 8, 10, 11, 13, 14, 15]), key=f"logic_{idx}"):
                selected_logics.append(idx)
        
        manual_count = st.slider("추출할 번호 개수", 3, 15, 6, key="slider_manual")
        
        if st.button("🎲 번호 생성"):
            if not selected_logics:
                st.warning("최소 1개 이상의 로직을 선택하세요.")
            else:
                model, scaler, labels, transitions = model_pack
                current_logics = {
                    1: (predict_by_total_frequency, (df,)),
                    2: (predict_by_recent_frequency, (df,)),
                    3: (predict_by_long_term_unappeared, (df,)),
                    4: (predict_by_good_compatibility, (df, co_occurrence)),
                    5: (predict_by_strongest_pairs, (co_occurrence,)),
                    6: (predict_by_number_temperature, (df,)),
                    7: (predict_by_positional_frequency, (df,)),
                    8: (predict_by_volatility_vector, (df,)),
                    9: (predict_by_consecutive_pattern, (df,)),
                    10: (predict_by_reappearing_number, (df,)),
                    11: (predict_by_prime_numbers, (df,)),
                    12: (predict_by_number_zones, (df,)),
                    13: (predict_by_regression_cycle, (df,)),
                    14: (predict_by_weighted_recent, (df,)),
                    15: (predict_by_neighbors, (df,)),
                    16: (predict_by_cluster_transition, (df, model, scaler, labels, transitions)),
                }
                
                score_board = defaultdict(float)
                for idx in selected_logics:
                    func, args = current_logics[idx]
                    candidates = func(*args)
                    weight = LOGIC_WEIGHTS.get(idx, 1.0)
                    for rank, num in enumerate(candidates):
                        score_board[num] += (weight + (0.5 if rank < 5 else 0))
                
                final_nums = select_final_numbers(score_board, manual_count)
                
                st.divider()
                st.write("### 🎱 생성 결과")
                
                # 공 모양으로 예쁘게 출력
                html_code = ""
                for n in sorted(final_nums):
                    color = "#fbc400" if n <= 10 else "#69c" if n <= 20 else "#f72" if n <= 30 else "#aaa" if n <= 40 else "#b0d"
                    html_code += f"<span style='display:inline-block;background-color:{color};color:white;padding:10px;border-radius:50%;width:40px;height:40px;text-align:center;font-weight:bold;margin:5px;line-height:20px;'>{n}</span>"
                
                st.markdown(html_code, unsafe_allow_html=True)
                st.write("")
                st.info("선택한 로직들을 종합하여 AI가 추천한 번호입니다.")

if __name__ == '__main__':

    main()
