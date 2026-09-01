"""KuaiRand-Pure 数据加载 + 官方划分 + 特征编码。只依赖标准库和 numpy。"""
import csv, os, collections
import numpy as np

LABEL = 'long_view'
SPLITS = {'train': (20220408, 20220421),
          'valid': (20220422, 20220428),
          'test':  (20220429, 20220508)}
# 5 个基础特征域，始终存在。
FIELDS = ['user_id', 'video_id', 'author_id', 'tab', 'dur_bucket']

# 可选扩展域（物品侧 + 用户侧），供 encode(..., extra_fields=[...]) 按需拼接。
# 索引对应 load() 返回的行元组里紧跟在基础 7 个字段之后的位置，见 _EXTRA_FIELD_INDEX。
ITEM_EXTRA_FIELDS = ['music_id', 'video_type', 'upload_type']
USER_EXTRA_FIELDS = ['follow_user_num_range', 'register_days_range',
                      'fans_user_num_range', 'friend_user_num_range',
                      'user_active_degree']
EXTRA_FIELD_CHOICES = ITEM_EXTRA_FIELDS + USER_EXTRA_FIELDS
# 常用组合的快捷方式，对应 ablation_features.py 里验证过的三档。
FEATURE_SETS = {
    'base':  [],
    'item':  ITEM_EXTRA_FIELDS,
    'cwm13': ITEM_EXTRA_FIELDS + USER_EXTRA_FIELDS,
}

_EXTRA_FIELD_INDEX = {name: 7 + i for i, name in enumerate(EXTRA_FIELD_CHOICES)}

def load(data_dir):
    """读日志 + 视频/用户侧特征，返回按划分切好的 dict。
    每行元组的前 7 个字段（date..label）保持原有下标不变，向后兼容旧代码；
    扩展域一律追加在末尾，缺失值统一填 'UNK'。"""
    vid_meta = {}
    with open(os.path.join(data_dir, 'video_features_basic_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            vid_meta[r['video_id']] = r

    user_meta = {}
    with open(os.path.join(data_dir, 'user_features_pure.csv')) as fh:
        for r in csv.DictReader(fh):
            user_meta[r['user_id']] = r

    rows = []
    for f in ('log_standard_4_08_to_4_21_pure.csv', 'log_standard_4_22_to_5_08_pure.csv'):
        with open(os.path.join(data_dir, f)) as fh:
            for r in csv.DictReader(fh):
                vm = vid_meta.get(r['video_id'], {})
                um = user_meta.get(r['user_id'], {})
                rows.append((
                    int(r['date']), r['user_id'], r['video_id'],
                    vm.get('author_id', 'UNK'), r['tab'],
                    float(r['duration_ms']), 1 if r[LABEL] != '0' else 0,
                ) + tuple(vm.get(f, 'UNK') for f in ITEM_EXTRA_FIELDS)
                  + tuple(um.get(f, 'UNK') for f in USER_EXTRA_FIELDS))

    out = {}
    for name, (lo, hi) in SPLITS.items():
        out[name] = [x for x in rows if lo <= x[0] <= hi]
    return out

def _bucket_edges(durations, n=10):
    return np.quantile(np.asarray(durations), np.linspace(0, 1, n + 1)[1:-1])

def encode(splits, extra_fields=()):
    """把类别特征映射成连续 id。未见过的取值统一落到该域的 UNK 槽。
    extra_fields: EXTRA_FIELD_CHOICES 的任意子集（或 FEATURE_SETS 里的预设名之一），
    追加在基础 5 域之后。留空则行为与原始 5 域版本完全一致。
    返回 (X, y, users) per split，X 为 int32 (N, len(FIELDS)+len(extra_fields))，以及 field_dims。"""
    if isinstance(extra_fields, str):
        extra_fields = FEATURE_SETS.get(extra_fields)
        if extra_fields is None:
            raise ValueError(f"未知的 feature set，可选: {list(FEATURE_SETS)}")
    unknown = [f for f in extra_fields if f not in _EXTRA_FIELD_INDEX]
    if unknown:
        raise ValueError(f"未知的扩展特征域 {unknown}，可选: {EXTRA_FIELD_CHOICES}")

    tr = splits['train']
    edges = _bucket_edges([x[5] for x in tr])

    def raw(x):
        base = [x[1], x[2], x[3], x[4], str(int(np.searchsorted(edges, x[5])))]
        return base + [x[_EXTRA_FIELD_INDEX[f]] for f in extra_fields]

    vocabs = [dict() for _ in range(len(FIELDS) + len(extra_fields))]
    for x in tr:
        for i, v in enumerate(raw(x)):
            if v not in vocabs[i]:
                vocabs[i][v] = len(vocabs[i])
    unk = [len(v) for v in vocabs]                 # 每个域末尾留一个 UNK 槽
    field_dims = [len(v) + 1 for v in vocabs]
    offsets = np.cumsum([0] + field_dims[:-1]).astype(np.int32)

    enc = {}
    for name, rws in splits.items():
        X = np.empty((len(rws), len(FIELDS) + len(extra_fields)), dtype=np.int32)
        y = np.empty(len(rws), dtype=np.float32)
        users = []
        for n, x in enumerate(rws):
            for i, v in enumerate(raw(x)):
                X[n, i] = vocabs[i].get(v, unk[i]) + offsets[i]
            y[n] = x[6]
            users.append(x[1])
        enc[name] = (X, y, users)
    return enc, int(sum(field_dims))
