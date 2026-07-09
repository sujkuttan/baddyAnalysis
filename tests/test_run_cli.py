import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import run


class RunCliTest(unittest.TestCase):
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
                 mock.patch.object(run.pipeline, "run_full_pipeline") as mocked:
                run.main()

        self.assertEqual(mocked.call_args.kwargs["inpaintnet_weights"], "InpaintNet_best.pt")

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
