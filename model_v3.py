import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 命令列參數 / Command-line arguments
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--train',  type=str, default='/Users/renyihuang/NLP/train_data.csv')
parser.add_argument('--test',   type=str, default='/Users/renyihuang/NLP/test_data.csv')
parser.add_argument('--output', type=str, default='/Users/renyihuang/NLP/submission_v3.csv')
args = parser.parse_args()

# ============================================================
# 1. 讀取資料 / Load Data
# ============================================================
train = pd.read_csv(args.train)
test  = pd.read_csv(args.test)

print("資料載入完成 / Data loaded: train={}, test={}".format(len(train), len(test)))

# ============================================================
# 2. 解析 word_id / Parse word_id
# ============================================================
def get_word_key(wid):
    parts = wid.split('_')
    try:
        page_idx = parts.index('page')
        text = '_'.join(parts[:page_idx - 1])
        page = parts[page_idx + 1]
        widx = parts[page_idx + 2]
        return '{}_page_{}_{}'.format(text, page, widx)
    except:
        return wid

train['word_key'] = train['word_id'].apply(get_word_key)
test['word_key']  = test['word_id'].apply(get_word_key)

# ============================================================
# 3. 清理單詞 / Clean words
# ============================================================
STRIP_CHARS = ".,;:!?()[]{}\"'-–—«»„\""
train['word_clean'] = train['word'].str.strip(STRIP_CHARS)
test['word_clean']  = test['word'].str.strip(STRIP_CHARS)
train['word_lower'] = train['word_clean'].str.lower()
test['word_lower']  = test['word_clean'].str.lower()
train['text_type']  = train['text'].str.split('_').str[0]
test['text_type']   = test['text'].str.split('_').str[0]

# ============================================================
# 4. 建立查找表 / Build lookup tables
# ============================================================

# 4a. 參與者統計 / Participant stats
participant_stats = train.groupby('participant_id')['answer'].agg(['mean','median','std'])
participant_stats.columns = ['part_mean','part_median','part_std']

# 4b. 單詞統計（區分大小寫）/ Word form stats
word_form_stats = train.groupby('word_clean')['answer'].agg(['mean','median','std','count'])
word_form_stats.columns = ['wf_mean','wf_median','wf_std','wf_count']

# 4c. 單詞統計（不分大小寫）/ Word lower stats
word_lower_stats = train.groupby('word_lower')['answer'].agg(['mean','median','count'])
word_lower_stats.columns = ['wl_mean','wl_median','wl_count']

# 4d. 文本類型統計 / Text type stats
text_type_stats = train.groupby('text_type')['answer'].agg(['mean','std'])
text_type_stats.columns = ['tt_mean','tt_std']

# 4e. word_key 統計（同一位置跨參與者）/ word_key stats
word_key_stats = train.groupby('word_key')['answer'].agg(['mean','median','std'])
word_key_stats.columns = ['wk_mean','wk_median','wk_std']

# 4f. 前後詞長度 / Neighbor word lengths
train['word_len_raw'] = train['word_clean'].str.len().fillna(0)
train_sorted = train.drop_duplicates('word_key').sort_values(['text','word_key'])
train_sorted['prev_word_len'] = train_sorted.groupby('text')['word_len_raw'].shift(1).fillna(0)
train_sorted['next_word_len'] = train_sorted.groupby('text')['word_len_raw'].shift(-1).fillna(0)
train_sorted['prev2_word_len'] = train_sorted.groupby('text')['word_len_raw'].shift(2).fillna(0)
train_sorted['next2_word_len'] = train_sorted.groupby('text')['word_len_raw'].shift(-2).fillna(0)
neighbor_stats = train_sorted[['word_key','prev_word_len','next_word_len','prev2_word_len','next2_word_len']]
train = train.merge(neighbor_stats, on='word_key', how='left')

# 4g. 【新】字符 n-gram 頻率（估計單詞罕見程度）
#     [NEW] Character n-gram frequency (estimates word rarity)
all_words = train['word_lower'].fillna('').tolist()
bigram_counter  = Counter()
trigram_counter = Counter()
for w in all_words:
    for i in range(len(w)-1):
        bigram_counter[w[i:i+2]]  += 1
    for i in range(len(w)-2):
        trigram_counter[w[i:i+3]] += 1

total_bigrams  = sum(bigram_counter.values())  + 1
total_trigrams = sum(trigram_counter.values()) + 1

def avg_bigram_freq(word):
    w = str(word).lower()
    if len(w) < 2:
        return 0.0
    freqs = [bigram_counter.get(w[i:i+2], 0) / total_bigrams for i in range(len(w)-1)]
    return float(np.mean(freqs))

def avg_trigram_freq(word):
    w = str(word).lower()
    if len(w) < 3:
        return 0.0
    freqs = [trigram_counter.get(w[i:i+3], 0) / total_trigrams for i in range(len(w)-2)]
    return float(np.mean(freqs))

# 4h. 零值比例（這個字被跳過的機率）/ Skip rate per word
word_skip_rate = train.groupby('word_lower').apply(lambda x: (x['answer'] == 0).mean()).rename('skip_rate')

# 4i. 非零TRT的平均值 / Non-zero TRT mean
nonzero_train = train[train['answer'] > 0]
word_nonzero_mean = nonzero_train.groupby('word_lower')['answer'].mean().rename('wl_nonzero_mean')

global_mean        = train['answer'].mean()
global_median      = train['answer'].median()
global_std         = train['answer'].std()
global_skip_rate   = (train['answer'] == 0).mean()
global_nonzero_mean = nonzero_train['answer'].mean()

print("查找表建立完成 / Lookup tables built")
print("全局跳過率 / Global skip rate: {:.1%}".format(global_skip_rate))

# ============================================================
# 5. 特徵工程 / Feature Engineering
# ============================================================
VOWELS    = set('aeiouAEIOUăâîĂÂÎ')
RO_DIAC   = set('ăâîșțĂÂÎȘȚ')
CONSONANT_CLUSTERS = ['str','ntr','mpr','ngr','ndr','spr','scr','zdr','zgr']

def engineer(df, is_train=True):
    df = df.copy()
    wc = df['word_clean'].fillna('').astype(str)
    wl = df['word_lower'].fillna('').astype(str)

    # === 基礎特徵 / Basic features ===
    df['word_len']        = wc.str.len().astype(float)
    df['n_vowels']        = wc.apply(lambda w: sum(1 for c in w if c in VOWELS))
    df['n_consonants']    = df['word_len'] - df['n_vowels']
    df['vowel_ratio']     = df['n_vowels'] / (df['word_len'] + 1)
    df['has_diacritics']  = wc.apply(lambda w: int(any(c in RO_DIAC for c in w)))
    df['n_diacritics']    = wc.apply(lambda w: sum(1 for c in w if c in RO_DIAC))
    df['is_numeric']      = wc.apply(lambda w: int(w.replace('.','').replace(',','').isdigit() and len(w) > 0))
    df['is_url']          = df['word'].fillna('').apply(lambda w: int('http' in w or 'www.' in w))
    df['is_capitalized']  = wc.apply(lambda w: int(len(w) > 0 and w[0].isupper()))
    df['is_all_caps']     = wc.apply(lambda w: int(len(w) > 1 and w.isupper()))
    df['word_len_sq']     = df['word_len'] ** 2
    df['word_len_log']    = np.log1p(df['word_len'])
    df['word_len_sqrt']   = np.sqrt(df['word_len'])
    df['has_hyphen']      = df['word'].fillna('').apply(lambda w: int('-' in w))

    # 羅馬尼亞語輔音群 / Romanian consonant clusters
    df['has_cluster']     = wl.apply(lambda w: int(any(c in w for c in CONSONANT_CLUSTERS)))

    # 字尾特徵（羅馬尼亞語字尾很重要）/ Word suffix features
    df['suffix_2']        = wl.apply(lambda w: w[-2:] if len(w) >= 2 else '')
    df['suffix_3']        = wl.apply(lambda w: w[-3:] if len(w) >= 3 else '')

    # 常見羅馬尼亞語停用詞 / Romanian stopwords (usually skipped)
    RO_STOPWORDS = {'de','la','în','și','cu','pe','că','ca','sa','se','nu','dar','sau','un','o','al','ale','lui','ei','ei','le','îi','ne','vă','mă','te'}
    df['is_stopword']     = wl.apply(lambda w: int(w in RO_STOPWORDS))

    # === n-gram 頻率特徵 / n-gram frequency features ===
    df['bigram_freq']     = wl.apply(avg_bigram_freq)
    df['trigram_freq']    = wl.apply(avg_trigram_freq)
    df['bigram_freq_log'] = np.log1p(df['bigram_freq'] * 1000)

    # === 位置特徵 / Position features ===
    df['word_pos']        = df['word_key'].str.split('_').str[-1].astype(float)
    df['page_num']        = df['word_key'].str.split('_').apply(lambda x: float(x[-2]) if len(x) >= 2 else 0.0)
    df['word_pos_log']    = np.log1p(df['word_pos'])
    df['word_pos_sq']     = df['word_pos'] ** 2

    # === 參與者特徵 / Participant features ===
    df = df.merge(participant_stats, on='participant_id', how='left')
    df['part_mean']       = df['part_mean'].fillna(global_mean)
    df['part_median']     = df['part_median'].fillna(global_median)
    df['part_std']        = df['part_std'].fillna(global_std)
    # 參與者速度相對於全局 / Relative participant speed
    df['part_speed_rel']  = df['part_mean'] / global_mean

    # === 單詞查找特徵 / Word lookup features ===
    df = df.merge(word_form_stats, on='word_clean', how='left')
    df['wf_mean']         = df['wf_mean'].fillna(global_mean)
    df['wf_median']       = df['wf_median'].fillna(global_median)
    df['wf_std']          = df['wf_std'].fillna(global_std)
    df['wf_count']        = df['wf_count'].fillna(0)

    df = df.merge(word_lower_stats, on='word_lower', how='left')
    df['wl_mean']         = df['wl_mean'].fillna(df['wf_mean'])
    df['wl_median']       = df['wl_median'].fillna(df['wf_median'])
    df['wl_count']        = df['wl_count'].fillna(0)

    df = df.merge(text_type_stats, on='text_type', how='left')
    df['tt_mean']         = df['tt_mean'].fillna(global_mean)
    df['tt_std']          = df['tt_std'].fillna(global_std)

    df = df.merge(word_key_stats, on='word_key', how='left')
    df['wk_mean']         = df['wk_mean'].fillna(df['wf_mean'])
    df['wk_median']       = df['wk_median'].fillna(df['wf_median'])
    df['wk_std']          = df['wk_std'].fillna(df['wf_std'])

    # 跳過率 / Skip rate
    df = df.merge(word_skip_rate, on='word_lower', how='left')
    df['skip_rate']       = df['skip_rate'].fillna(global_skip_rate)

    # 非零均值 / Non-zero mean
    df = df.merge(word_nonzero_mean, on='word_lower', how='left')
    df['wl_nonzero_mean'] = df['wl_nonzero_mean'].fillna(global_nonzero_mean)

    # === 前後詞特徵 / Neighbor features ===
    for col in ['prev_word_len','next_word_len','prev2_word_len','next2_word_len']:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = df[col].fillna(0)
    df['neighbor_len_sum']  = df['prev_word_len'] + df['next_word_len']
    df['neighbor_len_mean'] = df['neighbor_len_sum'] / 2

    # === 互動特徵 / Interaction features ===
    df['wf_x_part']       = df['wf_mean'] * df['part_mean'] / global_mean
    df['wk_x_part']       = df['wk_mean'] * df['part_mean'] / global_mean
    df['len_x_part']      = df['word_len'] * df['part_mean']
    df['wf_seen']         = (df['wf_count'] > 0).astype(float)
    df['part_wf_diff']    = df['part_mean'] - df['wf_mean']
    df['skip_x_part']     = df['skip_rate'] * df['part_mean']
    df['bigram_x_len']    = df['bigram_freq'] * df['word_len']
    # 預期TRT = 非零均值 × (1 - 跳過率)
    df['expected_trt']    = df['wl_nonzero_mean'] * (1 - df['skip_rate'])
    df['expected_x_part'] = df['expected_trt'] * df['part_speed_rel']

    return df

print("開始特徵工程... / Engineering features...")
train = engineer(train, is_train=True)
test  = engineer(test,  is_train=False)
print("特徵工程完成 / Feature engineering done")

# ============================================================
# 6. 特徵列表 / Feature list
# ============================================================
FEATURES = [
    # 基礎 / Basic
    'word_len','n_vowels','n_consonants','vowel_ratio',
    'has_diacritics','n_diacritics','is_numeric','is_url',
    'is_capitalized','is_all_caps','has_hyphen','has_cluster',
    'is_stopword',
    'word_len_sq','word_len_log','word_len_sqrt',
    # n-gram / n-gram
    'bigram_freq','trigram_freq','bigram_freq_log',
    # 位置 / Position
    'word_pos','page_num','word_pos_log','word_pos_sq',
    # 前後詞 / Neighbors
    'prev_word_len','next_word_len','prev2_word_len','next2_word_len',
    'neighbor_len_sum','neighbor_len_mean',
    # 參與者 / Participant
    'part_mean','part_median','part_std','part_speed_rel',
    # 單詞查找 / Word lookup
    'wf_mean','wf_median','wf_std','wf_count',
    'wl_mean','wl_median','wl_count',
    'wk_mean','wk_median','wk_std',
    'skip_rate','wl_nonzero_mean',
    # 文本 / Text
    'tt_mean','tt_std',
    # 互動 / Interactions
    'wf_x_part','wk_x_part','len_x_part','wf_seen',
    'part_wf_diff','skip_x_part','bigram_x_len',
    'expected_trt','expected_x_part',
]

X      = train[FEATURES].fillna(0).values
y      = train['answer'].values
X_test = test[FEATURES].fillna(0).values

print("特徵數量 / Number of features:", len(FEATURES))

# ============================================================
# 7. 評分函數 / Metric
# ============================================================
def eval_metric(y_true, preds):
    y_true = np.array(y_true, dtype=float)
    preds  = np.array(preds,  dtype=float)
    r2     = max(0.0, r2_score(y_true, preds))
    pears  = pearsonr(y_true, preds)[0]
    if np.isnan(pears): pears = 0.0
    return 100.0 * (abs(pears) + r2) / 2.0

# ============================================================
# 8. 模型定義 / Model definitions
# ============================================================
# 主力模型A：深度梯度提升 / Main model: deep gradient boosting
modelA = HistGradientBoostingRegressor(
    max_iter=1000, learning_rate=0.02, max_leaf_nodes=127,
    min_samples_leaf=15, l2_regularization=0.05,
    max_bins=255, random_state=42
)
# 模型B：極度隨機森林 / Model B: Extra Trees
modelB = ExtraTreesRegressor(
    n_estimators=300, max_depth=15, min_samples_leaf=10,
    n_jobs=-1, random_state=42
)
# 模型C：隨機森林 / Model C: Random Forest
modelC = RandomForestRegressor(
    n_estimators=300, max_depth=15, min_samples_leaf=10,
    n_jobs=-1, random_state=42
)
# 模型D：Huber（對異常值更robust）/ Model D: Huber (robust to outliers)
scaler = StandardScaler()
modelD = HuberRegressor(epsilon=1.35, max_iter=200, alpha=0.01)

# ============================================================
# 9. 交叉驗證 + Stacking / CV + Stacking
# ============================================================
groups = train['text'].values
gkf    = GroupKFold(n_splits=9)

# Stacking：收集OOF預測 / Collect Out-Of-Fold predictions for stacking
oof_A = np.zeros(len(X))
oof_B = np.zeros(len(X))
oof_C = np.zeros(len(X))
oof_D = np.zeros(len(X))

cv_scores = []

print("=" * 65)
print("交叉驗證 / Cross-Validation")
print("=" * 65)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    modelA.fit(X_tr, y_tr)
    modelB.fit(X_tr, y_tr)
    modelC.fit(X_tr, y_tr)
    X_tr_s  = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    modelD.fit(X_tr_s, y_tr)

    pA = modelA.predict(X_val)
    pB = modelB.predict(X_val)
    pC = modelC.predict(X_val)
    pD = modelD.predict(X_val_s)

    oof_A[val_idx] = pA
    oof_B[val_idx] = pB
    oof_C[val_idx] = pC
    oof_D[val_idx] = pD

    # 加權集成 / Weighted ensemble
    p_ens = 0.45 * pA + 0.25 * pB + 0.20 * pC + 0.10 * pD

    sA   = eval_metric(y_val, pA)
    sB   = eval_metric(y_val, pB)
    sC   = eval_metric(y_val, pC)
    sD   = eval_metric(y_val, pD)
    sEns = eval_metric(y_val, p_ens)
    cv_scores.append(sEns)

    g = groups[val_idx][0]
    print("Fold {:d} ({:20s}) | A:{:.1f} B:{:.1f} C:{:.1f} D:{:.1f} | Ens:{:.1f}".format(
        fold+1, g, sA, sB, sC, sD, sEns))

print("=" * 65)
print("平均 Ensemble / Mean Ensemble: {:.2f} +/- {:.2f}".format(
    np.mean(cv_scores), np.std(cv_scores)))
print("=" * 65)

# ============================================================
# 10. Meta-model Stacking
#     用OOF預測值訓練一個meta模型 / Train meta-model on OOF predictions
# ============================================================
print("\nMeta-model Stacking 訓練中...")
X_meta_train = np.column_stack([oof_A, oof_B, oof_C, oof_D])
meta_model    = Ridge(alpha=1.0)
meta_scaler   = StandardScaler()
X_meta_s      = meta_scaler.fit_transform(X_meta_train)
meta_model.fit(X_meta_s, y)

# Meta-model OOF 分數
meta_oof = meta_model.predict(X_meta_s)
print("Meta-model OOF score: {:.2f}".format(eval_metric(y, meta_oof)))

# ============================================================
# 11. 用全部資料重新訓練 / Retrain on full data
# ============================================================
print("\n全部資料重新訓練中... / Retraining on full data...")
modelA.fit(X, y)
modelB.fit(X, y)
modelC.fit(X, y)
X_s      = scaler.fit_transform(X)
X_test_s = scaler.transform(X_test)
modelD.fit(X_s, y)

# 預測測試集 / Predict test set
pA_test = modelA.predict(X_test)
pB_test = modelB.predict(X_test)
pC_test = modelC.predict(X_test)
pD_test = modelD.predict(X_test_s)

# Meta-model 預測 / Meta-model prediction
X_meta_test   = np.column_stack([pA_test, pB_test, pC_test, pD_test])
X_meta_test_s = meta_scaler.transform(X_meta_test)
meta_preds    = meta_model.predict(X_meta_test_s)

# 最終：加權集成 + Meta-model 混合 / Final: blend weighted ensemble + meta
p_ens_test  = 0.45 * pA_test + 0.25 * pB_test + 0.20 * pC_test + 0.10 * pD_test
final_preds = 0.6 * p_ens_test + 0.4 * meta_preds
final_preds = np.clip(final_preds, 0, None)

# ============================================================
# 12. 儲存結果 / Save submission
# ============================================================
out = test[['datapointID']].copy()
out['subtaskID'] = 1
out['answer']    = final_preds
out = out[['subtaskID', 'datapointID', 'answer']]
out.to_csv(args.output, index=False)

print("\n完成！/ Done!")
print("輸出前10行 / First 10 rows:")
print(out.head(10).to_string())
print("\n檔案已存到 / Saved to:", args.output)
