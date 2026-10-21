"""Independent saved-score replay for the complete controlled experiment.
Author: James Edward Ball. Does not import the original aggregation functions.
"""
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np

AXES = {'branches': ['initial', 'final'], 'arms': ['mean', 'single'],
        'levels': ['12', '25', '50', 'all'], 'budgets': [256, 512],
        'references': ['qwen_instructed', 'qwen_plain', 'mpnet'],
        'repeats': 10, 'pairs': 8755}

def require(value, message):
    if not value: raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''): h.update(chunk)
    return h.hexdigest()

def read(path): return json.loads(path.read_text())

def close(actual, expected, label):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(),
            'Invalid comparison: '+label)
    error = float(np.max(np.abs(a-b)))
    require(error < 1e-12, 'Numerical mismatch: '+label)
    return error

def nested_means(scores, classes, cells):
    """Average within cells, then within categories, using explicit groups."""
    require(classes.ndim == cells.ndim == 1 and classes.shape == cells.shape
            and scores.shape[-1] == len(classes), 'Invalid query support')
    labels = np.unique(classes)
    require(len(labels) > 0 and np.isfinite(scores).all()
            and np.all((scores > 0) & (scores <= 1)), 'Invalid reciprocal ranks')
    result = np.zeros(scores.shape[:-1], dtype=np.float64)
    weights = np.zeros(len(classes), dtype=np.float64)
    for label in labels:
        positions = np.flatnonzero(classes == label)
        groups = [positions[cells[positions] == cell]
                  for cell in np.unique(cells[positions])]
        category = np.zeros_like(result)
        for group in groups:
            category += np.mean(scores[..., group], axis=-1, dtype=np.float64)
            weights[group] = 1.0 / (len(labels)*len(groups)*len(group))
        result += category / len(groups)
    return result / len(labels), weights

def compare_tables(directory, points):
    require(points.shape == (6, 2, 2, 4, 2, 3, 10), 'Incomplete points array')
    expected = {'endpoints.csv': {}, 'paired-differences.csv': {},
                'vocabulary-interactions.csv': {}}
    primary, interactions = [], []
    for seed in range(6):
        for arm_i, arm in enumerate(AXES['arms']):
            for budget_i, budget in enumerate(AXES['budgets']):
                for ref_i, ref in enumerate(AXES['references']):
                    initial, trained = [], []
                    for level_i, level in enumerate(AXES['levels']):
                        pair = [float(sum(points[seed,b,arm_i,level_i,budget_i,ref_i])/10)
                                for b in range(2)]
                        initial.append(pair[0]); trained.append(pair[1])
                        key = (str(seed), arm, str(budget), ref, level)
                        for branch, value in zip(AXES['branches'], pair):
                            expected['endpoints.csv'][key+(branch,)] = {'mrr': value}
                        expected['paired-differences.csv'][key] = {
                            'trained_minus_initial_mrr': pair[1]-pair[0]}
                    gain0, gain1 = initial[3]-initial[0], trained[3]-trained[0]
                    expected['vocabulary-interactions.csv'][(str(seed),arm,str(budget),ref)] = {
                        'initial_all_minus_12': gain0, 'trained_all_minus_12': gain1,
                        'paired_difference_in_vocabulary_gain': gain1-gain0}
                    if (arm, budget, ref) == ('single',512,'qwen_instructed'):
                        primary.append(trained[3]-initial[3]); interactions.append(gain1-gain0)
    maximum = 0.0
    for name, values in expected.items():
        with (directory/name).open(newline='') as stream: rows = list(csv.DictReader(stream))
        require(len(rows) == len(values), 'Wrong row count: '+name)
        seen = set()
        for row in rows:
            fields = ['seed_index','arm','budget','reference']
            if name != 'vocabulary-interactions.csv': fields.append('level')
            if name == 'endpoints.csv': fields.append('branch')
            key = tuple(row[f] for f in fields)
            require(key in values and key not in seen, 'Invalid/duplicate row: '+name)
            seen.add(key)
            for field, value in values[key].items():
                maximum = max(maximum, close(float(row[field]),value,name+'/'+field))
    summary = read(directory/'summary.json')
    require(summary['seed_runs'] == 6 and summary['fitting_repeats_per_run'] == 10,
            'Wrong replication counts')
    require(summary['primary_condition'] == {'arm':'single','budget':512,
            'level':'all','reference':'qwen_instructed'}, 'Wrong primary condition')
    require(summary['population_inference_qualified'] is False and
            summary['population_confidence_intervals_reported'] is False,
            'Unexpected population inference claim')
    for key, series in [('primary_paired_difference',primary),
                        ('primary_vocabulary_gain_interaction',interactions)]:
        record = summary[key]
        for field, value in [('values_by_seed',series), ('mean',sum(series)/6),
                             ('minimum',min(series)), ('maximum',max(series))]:
            maximum = max(maximum,close(record[field],value,key+'/'+field))
        require(record['positive_seeds'] == sum(v>0 for v in series)
                and record['negative_seeds'] == sum(v<0 for v in series)
                and record['zero_seeds'] == sum(v==0 for v in series)
                and record['range_is_confidence_interval'] is False,
                'Wrong sign counts or interval interpretation')
    return {'endpoints_checked':576, 'paired_contrasts_checked':288,
            'vocabulary_interactions_checked':72, 'maximum_table_error':maximum}

def main(base, output):
    require(not output.exists(), 'Verification output exists')
    manifest = read(base/'collection-manifest.json')
    require(manifest['ready'] is True, 'Incomplete collection')
    root = base/'controlled-pretraining-20261021-v1'
    def checked(path):
        key = str(path.relative_to(base)); record = manifest['files'][key]
        require(path.is_file() and not path.is_symlink() and
                path.stat().st_size == record['bytes'] and sha(path) == record['sha256'],
                'Collection file changed: '+key)
        return path
    final = root/'final-analysis'
    for name in ['summary.json','points-by-seed.npy','endpoints.csv',
                 'paired-differences.csv','vocabulary-interactions.csv']: checked(final/name)
    summary = read(final/'summary.json')
    require(summary['all_six_runs_verified'] is True and len(summary['runs']) == 6,
            'Incomplete final analysis')
    arrays=[]; support=None; error=0.0
    for seed in range(6):
        directory = root/'summary'/('%02d'%seed)
        receipt = read(checked(directory/'verification.json'))
        require(receipt['passed'] is True and receipt['run_index'] == seed
                and receipt['axes'] == AXES, 'Invalid readout receipt')
        file = checked(directory/'scores.npz')
        require(sha(file) == receipt['scores_sha256'], 'Changed readout scores')
        require(summary['runs'][seed]['seed_index'] == seed and
                summary['runs'][seed]['readout_verification_sha256'] == sha(directory/'verification.json'),
                'Readout lineage differs')
        with np.load(file, allow_pickle=False) as data:
            scores = data['scores']; classes, cells = data['cls'], data['cell']
            require(scores.shape == (2,2,4,2,3,10,8755) and len(np.unique(classes)) == 188,
                    'Incomplete score support')
            if support is None: support=(classes.copy(),cells.copy())
            else: require(np.array_equal(classes,support[0]) and np.array_equal(cells,support[1]),
                          'Query support differs between runs')
            points, weights = nested_means(scores,classes,cells)
            error=max(error,close(weights,data['weights'],'query weights'),
                      close(points,data['points'],'per-repeat points'))
            arrays.append(points)
        del scores
    points=np.stack(arrays)
    error=max(error,close(points,np.load(final/'points-by-seed.npy',allow_pickle=False),
                          'complete point array'))
    report=compare_tables(final,points)
    report.update(author='James Edward Ball',passed=True,all_six_runs_replayed=True,
                  query_scores_checked=int(6*np.prod((2,2,4,2,3,10,8755))),
                  maximum_score_aggregation_error=error,
                  collection_manifest_sha256=sha(base/'collection-manifest.json'),
                  replay_script_sha256=sha(Path(__file__)),
                  model_fitting_repeated=False,population_inference_qualified=False)
    with output.open('x') as stream: json.dump(report,stream,indent=2); stream.write('\n')
    print(json.dumps(report,indent=2))

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('collection',type=Path)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    main(args.collection,args.output)
