"""Verify the public Airfoil conversion preserves a contiguous temporal prefix."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from tfrecord.reader import tfrecord_iterator
from tfrecord.writer import TFRecordWriter
from tfrecord import example_pb2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.preprocess.prepare_airfoil200 import prepare


class AirfoilPreparationTests(unittest.TestCase):
    def test_prefix_and_geometry_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'official'
            source.mkdir()
            output = Path(folder)/'prepared'
            values = np.arange(601*2, dtype=np.float32).reshape(601,2,1)
            static = np.array([[[1,2],[3,4]]],dtype=np.float32)
            meta = {'trajectory_length':601, 'dt':0.0002,
                'features':{'pressure':{'type':'dynamic','shape':[601,2,1],'dtype':'float32'},
                            'mesh_pos':{'type':'static','shape':[1,2,2],'dtype':'float32'}}}
            (source/'meta.json').write_text(json.dumps(meta))
            for split in ('train','valid','test'):
                writer=TFRecordWriter(str(source/f'{split}.tfrecord'))
                writer.write({'pressure':(values.tobytes(),'byte'), 'mesh_pos':(static.tobytes(),'byte')})
                writer.close()
            prepare(source,output)
            actual=json.loads((output/'meta.json').read_text())
            self.assertEqual(actual['trajectory_length'],200)
            self.assertEqual(actual['dt'],meta['dt'])
            for split in ('train','valid','test'):
                records=list(tfrecord_iterator(str(output/f'{split}.tfrecord')))
                self.assertEqual(len(records),1)
                example=example_pb2.Example.FromString(bytes(records[0]))
                self.assertEqual(example.features.feature['pressure'].bytes_list.value[0],values[:200].tobytes())
                self.assertEqual(example.features.feature['mesh_pos'].bytes_list.value[0],static.tobytes())
            with self.assertRaises(FileExistsError):
                prepare(source,output)


if __name__ == '__main__':
    unittest.main()
