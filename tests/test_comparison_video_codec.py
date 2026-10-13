import unittest
from unittest.mock import MagicMock, call, patch

from scripts.run_baseline_vs_marsp_compare import open_comparison_video_writer


class ComparisonVideoCodecTests(unittest.TestCase):
    @patch("scripts.run_baseline_vs_marsp_compare.cv2.VideoWriter_fourcc")
    @patch("scripts.run_baseline_vs_marsp_compare.cv2.VideoWriter")
    def test_prefers_browser_compatible_h264(self, video_writer, fourcc) -> None:
        writer = MagicMock()
        writer.isOpened.return_value = True
        video_writer.return_value = writer

        self.assertIs(open_comparison_video_writer("comparison.mp4", 25, (640, 360)), writer)
        fourcc.assert_called_once_with(*"avc1")
        video_writer.assert_called_once_with("comparison.mp4", fourcc.return_value, 25, (640, 360))

    @patch("scripts.run_baseline_vs_marsp_compare.cv2.VideoWriter_fourcc")
    @patch("scripts.run_baseline_vs_marsp_compare.cv2.VideoWriter")
    def test_falls_back_when_h264_encoder_is_unavailable(self, video_writer, fourcc) -> None:
        unavailable = MagicMock()
        unavailable.isOpened.return_value = False
        fallback = MagicMock()
        fallback.isOpened.return_value = True
        video_writer.side_effect = [unavailable, fallback]

        self.assertIs(open_comparison_video_writer("comparison.mp4", 25, (640, 360)), fallback)
        self.assertEqual(fourcc.call_args_list, [call(*"avc1"), call(*"mp4v")])
        unavailable.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
