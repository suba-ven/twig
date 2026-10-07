"""Extract states 0:200 (stride 1) from public MeshGraphNets Airfoil TFRecords.

Preserves record ordering, split membership, static geometry, and original dt.
Run before preprocess_airfoil.py. This streams records without TensorFlow.
"""
import argparse
import copy
import json
from pathlib import Path
import struct
import numpy as np
from tfrecord import example_pb2
from tfrecord.reader import tfrecord_iterator
from tfrecord.writer import TFRecordWriter


def prepare(source, output):
    source, output = Path(source), Path(output)
    meta = json.loads((source/'meta.json').read_text())
    if meta['trajectory_length'] != 601 or meta.get('temporal_subsampling', {}).get('stride', 1) != 1:
        raise ValueError('Expected original, unstrided 601-state MeshGraphNets Airfoil')
    for split in ('train', 'valid', 'test'):
        if not (source/f'{split}.tfrecord').is_file():
            raise FileNotFoundError(source/f'{split}.tfrecord')
    # Never overwrite an existing dataset, including one from an interrupted run.
    output.mkdir(parents=True, exist_ok=False)
    result = copy.deepcopy(meta)
    dynamic = [name for name, spec in meta['features'].items() if spec['type'] == 'dynamic']
    for name in dynamic:
        result['features'][name]['shape'][0] = 200
    result['trajectory_length'] = 200
    counts = {}
    for split in ('train', 'valid', 'test'):
        count = 0
        with (output/f'{split}.tfrecord').open('wb') as handle:
            for raw in tfrecord_iterator(str(source/f'{split}.tfrecord')):
                example = example_pb2.Example.FromString(bytes(raw))
                for name in dynamic:
                    spec = meta['features'][name]
                    field = example.features.feature[name].bytes_list.value
                    if len(field) != 1:
                        raise ValueError(f'Expected one bytes value: {split}/{name}')
                    values = np.frombuffer(field[0], dtype=np.dtype(spec['dtype'])).reshape(spec['shape'])
                    field[0] = np.ascontiguousarray(values[:200]).tobytes()
                payload = example.SerializeToString()
                length = struct.pack('<Q', len(payload))
                handle.write(length); handle.write(TFRecordWriter.masked_crc(length))
                handle.write(payload); handle.write(TFRecordWriter.masked_crc(payload))
                count += 1
        if count == 0:
            raise ValueError(f'Empty split: {split}')
        counts[split] = count
        print(f'{split}: retained 200 contiguous states in {count} records', flush=True)
    # Publish metadata only after all splits succeed.
    (output/'meta.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'preparation.json').write_text(json.dumps({
        'source': 'https://storage.googleapis.com/dm-meshgraphnets/airfoil/',
        'source_states': 601, 'source_indices': '0:200:1', 'records': counts}, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('data/airfoil/official'))
    parser.add_argument('--output', type=Path, default=Path('data/airfoil/raw'))
    args = parser.parse_args()
    prepare(args.source, args.output)
