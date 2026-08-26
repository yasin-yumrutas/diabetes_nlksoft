from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class WebReportTests(unittest.TestCase):
    def test_required_web_files_exist(self):
        for relative_path in ('index.html', 'web.css', 'web.js', 'vercel.json'):
            self.assertTrue((ROOT / relative_path).is_file(), relative_path)

    def test_report_uses_verified_metrics_and_responsible_framing(self):
        html = (ROOT / 'index.html').read_text(encoding='utf-8')
        self.assertIn('0.847', html)
        self.assertIn('0.479', html)
        self.assertIn('klinik tanı', html)
        self.assertIn('442', html)

    def test_gallery_assets_exist(self):
        for name in (
            '01_correlation_heatmap.png',
            '03_actual_vs_predicted.png',
            '06_feature_importances_elasticnet.png',
            '08_roc_auc_curves.png',
        ):
            self.assertTrue((ROOT / 'outputs' / 'charts' / name).is_file(), name)


if __name__ == '__main__':
    unittest.main()
