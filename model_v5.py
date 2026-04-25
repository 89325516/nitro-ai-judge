import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from collections import Counter
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--train',  type=str, default='/Users/renyihuang/NLP/train_data.csv')
parser.add_argument('--test',   type=str, default='/Users/renyihuang/NLP/test_data.csv')
parser.add_argument('--output', type=str, default='/Users/renyihuang/NLP/submission_v5.csv')
args = parser.parse_args()

train = pd.read_csv(args.train)
test  = pd.read_csv(args.test)
print("Data loaded: train={}, test={}".format(len(train), len(test)))

# ============================================================
# Parse word_id
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

STRIP_CHARS = ".,;:!?()[]{}\"'-–—«»„\""
train['word_clean'] = train['word'].str.strip(STRIP_CHARS)
test['word_clean']  = test['word'].str.strip(STRIP_CHARS)
train['word_lower'] = train['word_clean'].str.lower()
test['word_lower']  = test['word_clean'].str.lower()
train['text_type']  = train['text'].str.split('_').str[0]
test['text_type']   = test['text'].str.split('_').str[0]

# ============================================================
# Lookup tables
# ============================================================
participant_stats = train.groupby('participant_id')['answer'].agg(['mean','median','std'])
participant_stats.columns = ['part_mean','part_median','part_std']
part_skip = train.groupby('participant_id').apply(lambda x: (x['answer'] == 0).mean()).rename('part_skip_rate')
part_nonzero_mean = train[train['answer'] > 0].groupby('participant_id')['answer'].mean().rename('part_nonzero_mean')
participant_stats = participant_stats.join(part_skip).join(part_nonzero_mean)

part_texttype = train.groupby(['participant_id','text_type'])['answer'].mean().rename('part_tt_mean').reset_index()
part_tt_skip  = train.groupby(['participant_id','text_type']).apply(lambda x: (x['answer']==0).mean()).rename('part_tt_skip').reset_index()

word_form_stats = train.groupby('word_clean')['answer'].agg(['mean','median','std','count'])
word_form_stats.columns = ['wf_mean','wf_median','wf_std','wf_count']

word_lower_stats = train.groupby('word_lower')['answer'].agg(['mean','median','std','count'])
word_lower_stats.columns = ['wl_mean','wl_median','wl_std','wl_count']

text_stats = train.groupby('text')['answer'].agg(['mean','std','median'])
text_stats.columns = ['text_mean','text_std','text_median']

text_type_stats = train.groupby('text_type')['answer'].agg(['mean','std'])
text_type_stats.columns = ['tt_mean','tt_std']

word_key_stats = train.groupby('word_key')['answer'].agg(['mean','median','std','count'])
word_key_stats.columns = ['wk_mean','wk_median','wk_std','wk_count']

# Neighbor word lengths
train['word_len_raw'] = train['word_clean'].str.len().fillna(0)
train_sorted = train.drop_duplicates('word_key').sort_values(['text','word_key'])
for shift, name in [(1,'prev'),(- 1,'next'),(2,'prev2'),(-2,'next2')]:
    train_sorted[f'{name}_word_len'] = train_sorted.groupby('text')['word_len_raw'].shift(shift).fillna(0)
neighbor_cols = ['word_key','prev_word_len','next_word_len','prev2_word_len','next2_word_len']
train = train.merge(train_sorted[neighbor_cols], on='word_key', how='left')

# Text word counts
text_len_map = train.drop_duplicates('word_key').groupby('text')['word_key'].count().rename('text_n_words')
train = train.merge(text_len_map, on='text', how='left')
test  = test.merge(text_len_map, on='text', how='left')

# n-gram frequency
all_words = train['word_lower'].fillna('').tolist()
bigram_counter, trigram_counter = Counter(), Counter()
for w in all_words:
    for i in range(len(w)-1): bigram_counter[w[i:i+2]] += 1
    for i in range(len(w)-2): trigram_counter[w[i:i+3]] += 1
total_bigrams  = sum(bigram_counter.values())  + 1
total_trigrams = sum(trigram_counter.values()) + 1

def avg_bigram_freq(word):
    w = str(word).lower()
    if len(w) < 2: return 0.0
    return float(np.mean([bigram_counter.get(w[i:i+2], 0) / total_bigrams for i in range(len(w)-1)]))

def avg_trigram_freq(word):
    w = str(word).lower()
    if len(w) < 3: return 0.0
    return float(np.mean([trigram_counter.get(w[i:i+3], 0) / total_trigrams for i in range(len(w)-2)]))

word_skip_rate    = train.groupby('word_lower').apply(lambda x: (x['answer']==0).mean()).rename('skip_rate')
nonzero_train     = train[train['answer'] > 0]
word_nonzero_mean = nonzero_train.groupby('word_lower')['answer'].mean().rename('wl_nonzero_mean')
word_nonzero_std  = nonzero_train.groupby('word_lower')['answer'].std().rename('wl_nonzero_std')

global_mean         = train['answer'].mean()
global_median       = train['answer'].median()
global_std          = train['answer'].std()
global_skip_rate    = (train['answer'] == 0).mean()
global_nonzero_mean = nonzero_train['answer'].mean()
global_nonzero_std  = nonzero_train['answer'].std()

print("Lookup tables built | Global skip rate: {:.1%}".format(global_skip_rate))

# ============================================================
# Feature engineering
# ============================================================
VOWELS = set('aeiouAEIOUăâîĂÂÎ')
RO_DIAC = set('ăâîșțĂÂÎȘȚ')
CONSONANT_CLUSTERS = ['str','ntr','mpr','ngr','ndr','spr','scr','zdr','zgr']
RO_STOPWORDS = {'de','la','în','și','cu','pe','că','ca','sa','se','nu','dar','sau',
                'un','o','al','ale','lui','ei','le','îi','ne','vă','mă','te','este',
                'sunt','era','ai','am','avem','fi','fost','care','ce','din','prin',
                'mai','poate','când','cum','tot','toate','acest','această','aceste'}

def engineer(df, is_train=True):
    df = df.copy()
    wc = df['word_clean'].fillna('').astype(str)
    wl = df['word_lower'].fillna('').astype(str)

    # Basic features
    df['word_len']       = wc.str.len().astype(float)
    df['n_vowels']       = wc.apply(lambda w: sum(1 for c in w if c in VOWELS))
    df['n_consonants']   = df['word_len'] - df['n_vowels']
    df['vowel_ratio']    = df['n_vowels'] / (df['word_len'] + 1)
    df['has_diacritics'] = wc.apply(lambda w: int(any(c in RO_DIAC for c in w)))
    df['n_diacritics']   = wc.apply(lambda w: sum(1 for c in w if c in RO_DIAC))
    df['is_numeric']     = wc.apply(lambda w: int(w.replace('.','').replace(',','').isdigit() and len(w) > 0))
    df['is_url']         = df['word'].fillna('').apply(lambda w: int('http' in w or 'www.' in w))
    df['is_capitalized'] = wc.apply(lambda w: int(len(w) > 0 and w[0].isupper()))
    df['is_all_caps']    = wc.apply(lambda w: int(len(w) > 1 and w.isupper()))
    df['word_len_sq']    = df['word_len'] ** 2
    df['word_len_log']   = np.log1p(df['word_len'])
    df['word_len_sqrt']  = np.sqrt(df['word_len'])
    df['has_hyphen']     = df['word'].fillna('').apply(lambda w: int('-' in w))
    df['has_cluster']    = wl.apply(lambda w: int(any(c in w for c in CONSONANT_CLUSTERS)))
    df['is_stopword']    = wl.apply(lambda w: int(w in RO_STOPWORDS))
    df['n_unique_chars'] = wc.apply(lambda w: len(set(w.lower())))
    df['char_diversity'] = df['n_unique_chars'] / (df['word_len'] + 1)

    # n-gram
    df['bigram_freq']     = wl.apply(avg_bigram_freq)
    df['trigram_freq']    = wl.apply(avg_trigram_freq)
    df['bigram_freq_log'] = np.log1p(df['bigram_freq'] * 1000)
    df['trigram_freq_log']= np.log1p(df['trigram_freq'] * 1000)

    # Position features
    df['word_pos']       = df['word_key'].str.split('_').str[-1].astype(float)
    df['page_num']       = df['word_key'].str.split('_').apply(lambda x: float(x[-2]) if len(x) >= 2 else 0.0)
    df['word_pos_log']   = np.log1p(df['word_pos'])
    df['word_pos_sq']    = df['word_pos'] ** 2
    df['text_n_words']   = df['text_n_words'].fillna(df['word_pos'] + 1)
    df['pos_ratio']      = df['word_pos'] / (df['text_n_words'] + 1)
    df['is_last_10pct']  = (df['pos_ratio'] > 0.9).astype(float)
    df['is_first_10pct'] = (df['pos_ratio'] < 0.1).astype(float)
    df['is_mid']         = ((df['pos_ratio'] > 0.3) & (df['pos_ratio'] < 0.7)).astype(float)

    # Participant features
    df = df.merge(participant_stats, on='participant_id', how='left')
    df['part_mean']         = df['part_mean'].fillna(global_mean)
    df['part_median']       = df['part_median'].fillna(global_median)
    df['part_std']          = df['part_std'].fillna(global_std)
    df['part_skip_rate']    = df['part_skip_rate'].fillna(global_skip_rate)
    df['part_nonzero_mean'] = df['part_nonzero_mean'].fillna(global_nonzero_mean)
    df['part_speed_rel']    = df['part_mean'] / global_mean

    df = df.merge(part_texttype, on=['participant_id','text_type'], how='left')
    df['part_tt_mean'] = df['part_tt_mean'].fillna(df['part_mean'])
    df['part_tt_rel']  = df['part_tt_mean'] / (global_mean + 1e-9)

    df = df.merge(part_tt_skip, on=['participant_id','text_type'], how='left')
    df['part_tt_skip'] = df['part_tt_skip'].fillna(df['part_skip_rate'])

    # Word lookup
    df = df.merge(word_form_stats, on='word_clean', how='left')
    df['wf_mean']   = df['wf_mean'].fillna(global_mean)
    df['wf_median'] = df['wf_median'].fillna(global_median)
    df['wf_std']    = df['wf_std'].fillna(global_std)
    df['wf_count']  = df['wf_count'].fillna(0)

    df = df.merge(word_lower_stats, on='word_lower', how='left')
    df['wl_mean']   = df['wl_mean'].fillna(df['wf_mean'])
    df['wl_median'] = df['wl_median'].fillna(df['wf_median'])
    df['wl_std']    = df['wl_std'].fillna(df['wf_std'])
    df['wl_count']  = df['wl_count'].fillna(0)

    df = df.merge(text_stats, on='text', how='left')
    df['text_mean']   = df['text_mean'].fillna(global_mean)
    df['text_std']    = df['text_std'].fillna(global_std)
    df['text_median'] = df['text_median'].fillna(global_median)

    df = df.merge(text_type_stats, on='text_type', how='left')
    df['tt_mean'] = df['tt_mean'].fillna(global_mean)
    df['tt_std']  = df['tt_std'].fillna(global_std)

    df = df.merge(word_key_stats, on='word_key', how='left')
    df['wk_mean']   = df['wk_mean'].fillna(df['wf_mean'])
    df['wk_median'] = df['wk_median'].fillna(df['wf_median'])
    df['wk_std']    = df['wk_std'].fillna(df['wf_std'])
    df['wk_count']  = df['wk_count'].fillna(0)

    df = df.merge(word_skip_rate, on='word_lower', how='left')
    df['skip_rate'] = df['skip_rate'].fillna(global_skip_rate)

    df = df.merge(word_nonzero_mean, on='word_lower', how='left')
    df['wl_nonzero_mean'] = df['wl_nonzero_mean'].fillna(global_nonzero_mean)

    df = df.merge(word_nonzero_std, on='word_lower', how='left')
    df['wl_nonzero_std'] = df['wl_nonzero_std'].fillna(global_nonzero_std)

    # Neighbor words
    for col in ['prev_word_len','next_word_len','prev2_word_len','next2_word_len']:
        if col not in df.columns: df[col] = 0.0
        df[col] = df[col].fillna(0)
    df['neighbor_len_sum']  = df['prev_word_len'] + df['next_word_len']
    df['neighbor_len_mean'] = df['neighbor_len_sum'] / 2
    df['neighbor_len_max']  = df[['prev_word_len','next_word_len']].max(axis=1)

    # Interaction features
    df['wf_x_part']       = df['wf_mean'] * df['part_mean'] / global_mean
    df['wk_x_part']       = df['wk_mean'] * df['part_mean'] / global_mean
    df['len_x_part']      = df['word_len'] * df['part_mean']
    df['wf_seen']         = (df['wf_count'] > 0).astype(float)
    df['wk_seen']         = (df['wk_count'] > 0).astype(float)
    df['part_wf_diff']    = df['part_mean'] - df['wf_mean']
    df['skip_x_part']     = df['skip_rate'] * df['part_mean']
    df['bigram_x_len']    = df['bigram_freq'] * df['word_len']
    df['expected_trt']    = df['wl_nonzero_mean'] * (1 - df['skip_rate'])
    df['expected_x_part'] = df['expected_trt'] * df['part_speed_rel']
    df['skip_diff']       = df['part_skip_rate'] - df['skip_rate']
    df['len_x_tt']        = df['word_len'] * df['tt_mean']
    df['part_tt_x_wf']    = df['part_tt_mean'] * df['wf_mean'] / global_mean
    df['wk_vs_wf']        = df['wk_mean'] - df['wf_mean']
    df['part_vs_text']    = df['part_mean'] - df['text_mean']
    # Word difficulty estimate from length and low n-gram frequency
    df['word_difficulty'] = df['word_len'] * (1 - df['bigram_freq'] * 100)
    # Participant skip rate times word skip rate
    df['skip_x_skip']     = df['part_skip_rate'] * df['skip_rate']
    # Non-zero TRT relative to participant mean
    df['nz_vs_part']      = df['wl_nonzero_mean'] / (df['part_nonzero_mean'] + 1e-9)

    return df

print("Engineering features...")
train = engineer(train, is_train=True)
test  = engineer(test,  is_train=False)
print("Feature engineering done")

FEATURES = [
    'word_len','n_vowels','n_consonants','vowel_ratio',
    'has_diacritics','n_diacritics','is_numeric','is_url',
    'is_capitalized','is_all_caps','has_hyphen','has_cluster','is_stopword',
    'word_len_sq','word_len_log','word_len_sqrt',
    'n_unique_chars','char_diversity',
    'bigram_freq','trigram_freq','bigram_freq_log','trigram_freq_log',
    'word_pos','page_num','word_pos_log','word_pos_sq',
    'pos_ratio','is_last_10pct','is_first_10pct','is_mid','text_n_words',
    'prev_word_len','next_word_len','prev2_word_len','next2_word_len',
    'neighbor_len_sum','neighbor_len_mean','neighbor_len_max',
    'part_mean','part_median','part_std','part_speed_rel',
    'part_skip_rate','part_nonzero_mean','part_tt_mean','part_tt_rel','part_tt_skip',
    'wf_mean','wf_median','wf_std','wf_count',
    'wl_mean','wl_median','wl_std','wl_count',
    'wk_mean','wk_median','wk_std','wk_count',
    'skip_rate','wl_nonzero_mean','wl_nonzero_std',
    'text_mean','text_std','text_median',
    'tt_mean','tt_std',
    'wf_x_part','wk_x_part','len_x_part','wf_seen','wk_seen',
    'part_wf_diff','skip_x_part','bigram_x_len',
    'expected_trt','expected_x_part',
    'skip_diff','len_x_tt','part_tt_x_wf',
    'wk_vs_wf','part_vs_text','word_difficulty','skip_x_skip','nz_vs_part',
]

# Add key raw features to the meta-model
META_EXTRA_FEATURES = [
    'wk_mean','wk_count','part_mean','part_skip_rate',
    'skip_rate','wl_nonzero_mean','expected_trt',
    'word_len','pos_ratio','text_mean',
]

X      = train[FEATURES].fillna(0).values
y      = train['answer'].values
y_bin  = (y == 0).astype(int)          # 1 = skip, 0 = read
y_log  = np.where(y > 0, np.log1p(y), 0.0)  # log-transform for non-zero
X_test = test[FEATURES].fillna(0).values
X_meta_extra       = train[META_EXTRA_FEATURES].fillna(0).values
X_meta_extra_test  = test[META_EXTRA_FEATURES].fillna(0).values

print("Number of features:", len(FEATURES))

# ============================================================
# Metric
# ============================================================
def eval_metric(y_true, preds):
    y_true = np.array(y_true, dtype=float)
    preds  = np.array(preds,  dtype=float)
    r2     = max(0.0, r2_score(y_true, preds))
    pears  = pearsonr(y_true, preds)[0]
    if np.isnan(pears): pears = 0.0
    return 100.0 * (abs(pears) + r2) / 2.0

# ============================================================
# Model definitions
# ============================================================
lgb_reg_params = dict(
    n_estimators=2000, learning_rate=0.02, num_leaves=127,
    min_child_samples=15, reg_alpha=0.05, reg_lambda=0.05,
    subsample=0.8, colsample_bytree=0.8,
    n_jobs=-1, random_state=42, verbose=-1,
)
lgb_clf_params = dict(
    n_estimators=2000, learning_rate=0.02, num_leaves=63,
    min_child_samples=15, reg_alpha=0.1, reg_lambda=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective='binary', n_jobs=-1, random_state=42, verbose=-1,
)
lgb_log_params = dict(
    n_estimators=2000, learning_rate=0.02, num_leaves=127,
    min_child_samples=10, reg_alpha=0.05, reg_lambda=0.05,
    subsample=0.8, colsample_bytree=0.8,
    n_jobs=-1, random_state=43, verbose=-1,
)
xgb_params = dict(
    n_estimators=2000, learning_rate=0.02, max_depth=7,
    min_child_weight=10, reg_alpha=0.05, reg_lambda=1.0,
    subsample=0.8, colsample_bytree=0.8,
    tree_method='hist', n_jobs=-1, random_state=42, verbosity=0,
    early_stopping_rounds=50,
)
hgb_params = dict(
    max_iter=1000, learning_rate=0.02, max_leaf_nodes=127,
    min_samples_leaf=15, l2_regularization=0.05,
    max_bins=255, random_state=42
)

# ============================================================
# Cross-validation
# ============================================================
groups = train['text'].values
gkf    = GroupKFold(n_splits=9)

oof_lgb  = np.zeros(len(X))   # LightGBM regressor (raw target)
oof_xgb  = np.zeros(len(X))   # XGBoost regressor
oof_hgb  = np.zeros(len(X))   # HistGradientBoosting
oof_skip = np.zeros(len(X))   # P(skip) from classifier
oof_log  = np.zeros(len(X))   # LightGBM on log target
cv_scores = []

print("=" * 65)
print("Cross-validation")
print("=" * 65)

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    X_tr, X_val = X[tr_idx], X[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]
    y_bin_tr    = y_bin[tr_idx]
    y_log_tr    = y_log[tr_idx]

    cb_lgb = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]

    # A: LightGBM regressor
    mA = lgb.LGBMRegressor(**lgb_reg_params)
    mA.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=cb_lgb)
    pA = mA.predict(X_val)
    oof_lgb[val_idx] = pA

    # B: XGBoost regressor
    mB = xgb.XGBRegressor(**xgb_params)
    mB.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    pB = mB.predict(X_val)
    oof_xgb[val_idx] = pB

    # C: HistGradientBoosting
    mC = HistGradientBoostingRegressor(**hgb_params)
    mC.fit(X_tr, y_tr)
    pC = mC.predict(X_val)
    oof_hgb[val_idx] = pC

    # D: LightGBM binary classifier → P(skip)
    mD = lgb.LGBMClassifier(**lgb_clf_params)
    mD.fit(X_tr, y_bin_tr, eval_set=[(X_val, y_bin[val_idx])], callbacks=cb_lgb)
    p_skip = mD.predict_proba(X_val)[:, 1]
    oof_skip[val_idx] = p_skip

    # E: LightGBM on log(TRT) for better non-zero value fit
    mE = lgb.LGBMRegressor(**lgb_log_params)
    mE.fit(X_tr, y_log_tr, eval_set=[(X_val, y_log[val_idx])], callbacks=cb_lgb)
    p_log = np.expm1(mE.predict(X_val))
    oof_log[val_idx] = p_log

    # Hurdle blend: P(not_skip) * predicted_TRT
    p_hurdle = (1 - p_skip) * pA

    p_ens = 0.35*pA + 0.25*pB + 0.15*pC + 0.15*p_hurdle + 0.10*p_log
    p_ens = np.clip(p_ens, 0, None)

    sA    = eval_metric(y_val, pA)
    sB    = eval_metric(y_val, pB)
    sC    = eval_metric(y_val, pC)
    sHurd = eval_metric(y_val, p_hurdle)
    sEns  = eval_metric(y_val, p_ens)
    cv_scores.append(sEns)

    g = groups[val_idx][0]
    print("Fold {:d} ({:20s}) | LGB:{:.1f} XGB:{:.1f} HGB:{:.1f} Hrd:{:.1f} | Ens:{:.1f}".format(
        fold+1, g, sA, sB, sC, sHurd, sEns))

print("=" * 65)
print("Mean ensemble: {:.2f} +/- {:.2f}".format(np.mean(cv_scores), np.std(cv_scores)))
print("=" * 65)

# ============================================================
# Meta-model Stacking
# LightGBM meta input: four OOF predictions, hurdle features, and key raw features
# ============================================================
print("\nTraining meta-model stacker...")
oof_hurdle = (1 - oof_skip) * oof_lgb

X_meta_train = np.column_stack([
    oof_lgb, oof_xgb, oof_hgb, oof_log,
    oof_skip, oof_hurdle,
    X_meta_extra
])

meta_params = dict(
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
    subsample=0.8, colsample_bytree=0.8,
    n_jobs=-1, random_state=42, verbose=-1,
)
meta_model = lgb.LGBMRegressor(**meta_params)
meta_model.fit(X_meta_train, y)

meta_oof = meta_model.predict(X_meta_train)
print("Meta-model OOF score: {:.2f}".format(eval_metric(y, meta_oof)))

# ============================================================
# Retrain on full data
# ============================================================
print("\nRetraining on full data...")
cb = [lgb.log_evaluation(-1)]
fA = lgb.LGBMRegressor(**lgb_reg_params); fA.fit(X, y, callbacks=cb)
fB = xgb.XGBRegressor(**{k: v for k, v in xgb_params.items() if k != 'early_stopping_rounds'}); fB.fit(X, y, verbose=False)
fC = HistGradientBoostingRegressor(**hgb_params); fC.fit(X, y)
fD = lgb.LGBMClassifier(**lgb_clf_params); fD.fit(X, y_bin, callbacks=cb)
fE = lgb.LGBMRegressor(**lgb_log_params);  fE.fit(X, y_log, callbacks=cb)

pA_t = fA.predict(X_test)
pB_t = fB.predict(X_test)
pC_t = fC.predict(X_test)
skip_t = fD.predict_proba(X_test)[:, 1]
pE_t = np.expm1(fE.predict(X_test))
hurdle_t = (1 - skip_t) * pA_t

X_meta_test = np.column_stack([
    pA_t, pB_t, pC_t, pE_t,
    skip_t, hurdle_t,
    X_meta_extra_test
])
meta_preds = meta_model.predict(X_meta_test)

p_ens_test  = 0.35*pA_t + 0.25*pB_t + 0.15*pC_t + 0.15*hurdle_t + 0.10*pE_t
final_preds = 0.5 * np.clip(p_ens_test, 0, None) + 0.5 * np.clip(meta_preds, 0, None)
final_preds = np.clip(final_preds, 0, None)

# ============================================================
# Save submission
# ============================================================
out = test[['datapointID']].copy()
out['subtaskID'] = 1
out['answer']    = final_preds
out = out[['subtaskID', 'datapointID', 'answer']]
out.to_csv(args.output, index=False)

print("\nDone!")
print("First 10 rows:")
print(out.head(10).to_string())
print("\nSaved to:", args.output)
