"""Проверка усреднения и парных изменений при отклонённом повторе."""
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import noise_repeats as study


class RepeatStatistics(unittest.TestCase):
    def test_rejections_keep_paired_baselines(self):
        rows = []
        for photo in ('a.jpg', 'b.jpg', 'c.jpg'):
            for noise in study.NOISES:
                for index, seed in enumerate(study.SEEDS, 1):
                    for mask in study.VARIANTS:
                        rejected = photo == 'a.jpg' and noise == 'bicubic' and mask == 'one' and index == 1
                        value = index * 10 + (1 if mask != 'baseline' else 0)
                        rows.append(dict(image=photo, degradation=noise, seed=seed, mask=mask,
                                         status='safety_rejected' if rejected else 'ok',
                                         psnr='' if rejected else value,
                                         lpips='' if rejected else value,
                                         clipiqa='' if rejected else value))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            study.write_csv(output / 'metrics.csv', rows)
            with patch.object(study, 'OUTPUT', output):
                study.aggregate()
            with (output / 'summary.csv').open(encoding='utf-8') as handle:
                summary = list(csv.DictReader(handle))
        self.assertEqual(len(summary), 81)
        row = next(r for r in summary if (r['image'], r['degradation'], r['mask']) == ('a.jpg', 'bicubic', 'one'))
        self.assertEqual(int(row['n']), 4)
        self.assertEqual(int(row['rejected']), 1)
        self.assertEqual(float(row['psnr_mean']), 36)
        self.assertEqual(float(row['psnr_delta_mean']), 1)
        self.assertEqual(float(row['lpips_delta_mean']), -1)
        self.assertAlmostEqual(float(row['psnr_std']), np.std([21, 31, 41, 51], ddof=1))
        self.assertEqual(float(row['psnr_delta_std']), 0)


if __name__ == '__main__':
    unittest.main()
