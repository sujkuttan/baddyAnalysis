import unittest

import numpy as np
import torch

from src.shuttle import TrackNet, TrackNetShuttle


class TrackNetShuttleBatchingTest(unittest.TestCase):
    def test_tracknet_matches_reference_spatial_topology(self):
        model = TrackNet(in_channels=27, out_channels=8)
        x = torch.zeros((1, 27, 288, 512), dtype=torch.float32)

        with torch.no_grad():
            e1 = model.down_block_1(x)
            e2 = model.down_block_2(torch.nn.MaxPool2d((2, 2), stride=(2, 2))(e1))
            y = model(x)

        self.assertEqual(tuple(e1.shape), (1, 64, 288, 512))
        self.assertEqual(tuple(e2.shape), (1, 128, 144, 256))
        self.assertEqual(tuple(y.shape), (1, 8, 288, 512))

    def test_predict_frames_returns_only_new_batch_and_keeps_bounded_overlap(self):
        tracker = TrackNetShuttle.__new__(TrackNetShuttle)
        tracker.seq_len = 3
        tracker.img_size = (4, 4)
        tracker.crop = None
        tracker._buf = []

        calls = []

        def fake_predict_from_small(small, bg, fw, fh):
            calls.append(len(small))
            return np.column_stack([
                np.arange(len(small), dtype=np.float64),
                np.arange(len(small), dtype=np.float64),
            ])

        tracker._predict_from_small = fake_predict_from_small

        first = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(2)]
        second = [np.ones((4, 4, 3), dtype=np.uint8) for _ in range(3)]

        out1 = tracker.predict_frames(first)
        out2 = tracker.predict_frames(second)

        self.assertEqual(len(out1), 2)
        self.assertEqual(len(out2), 3)
        self.assertEqual(calls, [2, 5])
        self.assertLessEqual(len(tracker._buf), tracker.seq_len - 1)


if __name__ == "__main__":
    unittest.main()
