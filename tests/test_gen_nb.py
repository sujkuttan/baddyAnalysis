import pathlib
import unittest


class GeneratedNotebookSourceTest(unittest.TestCase):
    def test_colab_defaults_use_t4_safe_batch_params(self):
        source = pathlib.Path("gen_nb.py").read_text()

        self.assertIn("BATCH_SIZE = 64", source)
        self.assertIn("TrackNet's internal `chunk=8`", source)
        self.assertIn("lower TrackNet's internal chunk from 8 to 4", source)
        self.assertIn("900 = ~30s smoke test", source)
        self.assertIn("SAMPLE_FRAMES = 3600", source)


if __name__ == "__main__":
    unittest.main()
