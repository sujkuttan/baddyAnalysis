import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import run


class RunCliTest(unittest.TestCase):
    def test_pipeline_defaults_to_local_sample_15s_validation_run(self):
        argv = ["run.py", "pipeline"]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(run, "require_pipeline_dependencies"), \
             mock.patch.object(run.pipeline, "run_full_pipeline") as mocked:
            run.main()

        args = mocked.call_args.args
        kwargs = mocked.call_args.kwargs
        self.assertEqual(args[0], "/home/sujith/baddyCoach/videos/sample_5min.mp4")
        self.assertEqual(args[1], [[466, 77], [831, 80], [1181, 641], [148, 637]])
        self.assertEqual(kwargs["tracknet_weights"], "weights/TrackNet_best.pt")
        self.assertEqual(kwargs["inpaintnet_weights"], "weights/InpaintNet_best.pt")
        self.assertEqual(kwargs["max_frames"], 450)
        self.assertEqual(kwargs["batch_size"], 8)

    def test_pipeline_forwards_inpaintnet_weights(self):
        with tempfile.TemporaryDirectory() as td:
            corners_path = os.path.join(td, "corners.json")
            with open(corners_path, "w") as f:
                json.dump({"corners": [[0, 0], [10, 0], [10, 20], [0, 20]]}, f)

            argv = [
                "run.py",
                "pipeline",
                "--video", "clip.mp4",
                "--corners", corners_path,
                "--tracknet", "TrackNet_best.pt",
                "--inpaintnet", "InpaintNet_best.pt",
                "--max_frames", "5",
            ]
            with mock.patch.object(sys, "argv", argv), \
                 mock.patch.object(run, "require_pipeline_dependencies"), \
                 mock.patch.object(run.pipeline, "run_full_pipeline") as mocked:
                run.main()

        self.assertEqual(mocked.call_args.kwargs["inpaintnet_weights"], "InpaintNet_best.pt")

    def test_pipeline_missing_dependency_exits_with_install_hint(self):
        with mock.patch("run.importlib.util.find_spec", return_value=None):
            with self.assertRaises(SystemExit) as cm:
                run.require_pipeline_dependencies()

        self.assertIn("ultralytics", str(cm.exception))
        self.assertIn("python3 -m pip install -r requirements.txt", str(cm.exception))

    def test_shuttle_smoke_parser_invokes_command(self):
        argv = [
            "run.py",
            "shuttle-smoke",
            "--video", "clip.mp4",
            "--tracknet", "TrackNet_best.pt",
            "--inpaintnet", "InpaintNet_best.pt",
            "--max_frames", "5",
            "--batch_size", "2",
        ]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(run, "cmd_shuttle_smoke") as mocked:
            run.main()

        args = mocked.call_args.args[0]
        self.assertEqual(args.video, "clip.mp4")
        self.assertEqual(args.tracknet, "TrackNet_best.pt")
        self.assertEqual(args.inpaintnet, "InpaintNet_best.pt")
        self.assertEqual(args.max_frames, 5)
        self.assertEqual(args.batch_size, 2)

    def test_shuttle_smoke_defaults_are_cpu_safe(self):
        argv = [
            "run.py",
            "shuttle-smoke",
            "--video", "clip.mp4",
            "--tracknet", "TrackNet_best.pt",
        ]
        with mock.patch.object(sys, "argv", argv), \
             mock.patch.object(run, "cmd_shuttle_smoke") as mocked:
            run.main()

        args = mocked.call_args.args[0]
        self.assertEqual(args.max_frames, 16)
        self.assertEqual(args.batch_size, 8)


if __name__ == "__main__":
    unittest.main()
