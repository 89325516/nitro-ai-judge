import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('/Users/renyihuang/NLP/train_data.csv')
test = pd.read_csv('/Users/renyihuang/NLP/test_data.csv')

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
test['word_key'] = test['word_id'].apply(get_word_key)

STRIP_CHARS = ".,;:!?()[]{}\"'-"
train['word_clean'] = train['word'].str.strip(STRIP_CHARS)
test['word_clean'] = test['word'].str.strip(STRIP_CHARS)
train['word_lower'] = train['word_clean'].str.lower()
test['word_lower'] = test['word_clean'].str.lower()
train['text_type'] = train['text'].str.split('_').str[0]
test['text_type'] = test['text'].str.split('_').str[0]

participant_stats = train.groupby('participant_id')['answer'].agg(['mean','median'])
participant_stats.columns = ['part_mean', 'part_median']

word_form_stats = train.groupby('word_clean')['answer'].agg(['mean','median','count'])
word_form_stats.columns = ['wordform_mean_trt', 'wordform_median_trt', 'wordform_count']

word_lower_stats = train.groupby('word_lower')['answer'].agg(['mean','count'])
word_lower_stats.columns = ['wordlower_mean_trt', 'wordlower_count']

text_type_stats = train.groupby('text_type')['answer'].mean()
text_type_stats.name = 'text_type_mean_trt'

VOWELS = set('aeiouAEIOU')
global_mean = train['answer'].mean()
global_median = train['answer'].median()

def engineer(df):
    df = df.copy()
    wc = df['word_clean'].fillna('').astype(str)
    df['word_len'] = wc.str.len().astype(float)
    df['n_syllables'] = wc.apply(lambda w: sum(1 for c in w if c in VOWELS))
    df['is_numeric'] = wc.apply(lambda w: int(w.replace('.','').replace(',','').isdigit() and len(w) > 0))
    df['is_url'] = df['word'].fillna('').apply(lambda w: int('http' in w or 'www.' in w))
    df['word_len_sq'] = df['word_len'] ** 2
    df['word_len_log'] = np.log1p(df['word_len'])
    df['word_pos'] = df['word_key'].str.split('_').str[-1].astype(float)
    df['page_num'] = df['word_key'].str.split('_').apply(lambda x: float(x[-2]) if len(x) >= 2 else 0.0)
    df['word_pos_log'] = np.log1p(df['word_pos'])

    df = df.merge(participant_stats, on='participant_id', how='left')
    df['part_mean'] = df['part_mean'].fillna(global_mean)
    df['part_median'] = df['part_median'].fillna(global_median)

    df = df.merge(word_form_stats, on='word_clean', how='left')
    df['wordform_mean_trt'] = df['wordform_mean_trt'].fillna(global_mean)
    df['wordform_median_trt'] = df['wordform_median_trt'].fillna(global_median)
    df['wordform_count'] = df['wordform_count'].fillna(0)

    df = df.merge(word_lower_stats, on='word_lower', how='left')
    df['wordlower_mean_trt'] = df['wordlower_mean_trt'].fillna(df['wordform_mean_trt'])
    df['wordlower_count'] = df['wordlower_count'].fillna(0)

    df = df.merge(text_type_stats, on='text_type', how='left')
    df['text_type_mean_trt'] = df['text_type_mean_trt'].fillna(global_mean)

    df['wordform_x_partmean'] = df['wordform_mean_trt'] * df['part_mean'] / global_mean
    df['len_x_partmean'] = df['word_len'] * df['part_mean']

    return df

train = engineer(train)
test = engineer(test)

FEATURES = [
    'word_len', 'n_syllables', 'is_numeric', 'is_url',
    'word_len_sq', 'word_len_log',
    'word_pos', 'page_num', 'word_pos_log',
    'part_mean', 'part_median',
    'wordform_mean_trt', 'wordform_median_trt', 'wordform_count',
    'wordlower_mean_trt', 'wordlower_count',
    'text_type_mean_trt',
    'wordform_x_partmean', 'len_x_partmean'
]

X = train[FEATURES].fillna(0).values
y = train['answer'].values
X_test_arr = test[FEATURES].fillna(0).values

def eval_metric(y_true, preds):
    y_true = np.array(y_true, dtype=float)
    preds = np.array(preds, dtype=float)
    r2 = max(0.0, r2_score(y_true, preds))
    pears = pearsonr(y_true, preds)[0]
    if np.isnan(pears):
        pears = 0.0
    return 100.0 * (abs(pears) + r2) / 2.0

groups = train['text'].values
gkf = GroupKFold(n_splits=9)

model = HistGradientBoostingRegressor(
    max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
    min_samples_leaf=20, l2_regularization=0.1, random_state=42
)

cv_scores = []
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    model.fit(X[tr_idx], y[tr_idx])
    val_preds = model.predict(X[val_idx])
    score = eval_metric(y[val_idx], val_preds)
    cv_scores.append(score)
    g = groups[val_idx][0]
    print("Fold {} ({}): {:.2f}".format(fold+1, g, score))

print("\nMean CV: {:.2f} +/- {:.2f}".format(np.mean(cv_scores), np.std(cv_scores)))

# Train on full data and predict
model.fit(X, y)
test_preds = model.predict(X_test_arr)
test_preds = np.clip(test_preds, 0, None)

out = test[['datapointID']].copy()
out['subtaskID'] = 1
out['answer'] = test_preds
out = out[['subtaskID', 'datapointID', 'answer']]

out.to_csv('/Users/renyihuang/NLP/submission.csv', index=False)
print("\nSubmission saved.")
print(out.head(10).to_string())
print("Shape:", out.shape)