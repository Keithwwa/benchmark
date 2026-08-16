import unittest

from ais_bench.benchmark.utils.language_check import (
    chinese_context,
    detect_scripts,
    has_chinese,
    is_pure_ascii,
    language_check,
    significant_scripts,
)


class TestLanguageCheck(unittest.TestCase):
    def test_pure_ascii(self):
        self.assertTrue(is_pure_ascii("The correct answer is (A)"))
        self.assertTrue(is_pure_ascii(""))
        self.assertFalse(is_pure_ascii(None))
        self.assertFalse(is_pure_ascii("答案正确的是A"))

    def test_detect_chinese(self):
        self.assertIn("chinese", detect_scripts("这个答案是正确的"))
        self.assertIn("chinese", detect_scripts("The answer is 正确"))
        self.assertFalse(has_chinese("The correct answer is (A)"))

    def test_has_chinese_mixed(self):
        # 中英文混杂：ASCII 英文与中文字符同时出现
        self.assertTrue(has_chinese("The correct answer is 正确答案 A"))
        self.assertTrue(has_chinese("答案为（B）正确"))

    def test_detect_fullwidth(self):
        scripts = detect_scripts("答案：Ａ")
        self.assertTrue(scripts & {"chinese", "fullwidth_cjk"})

    def test_detect_other_scripts(self):
        self.assertIn("japanese_kana", detect_scripts("こんにちは"))
        self.assertIn("korean_hangul", detect_scripts("안녕하세요"))
        self.assertIn("greek", detect_scripts("The angle is α"))
        self.assertIn("cyrillic", detect_scripts("Привет"))

    def test_significant_scripts_ignores_math_and_latin_extended(self):
        # 纯英文技术输出中的乘号/连接号不应触发混杂告警
        chemistry = (
            "Structure: CH2=CH–CH(CH3)–CH2–CH3 -> internal alkene. "
            "Total 3 combinations × 2 = 6 possible products."
        )
        self.assertTrue(detect_scripts(chemistry))  # 存在非 ASCII（×/–）
        self.assertFalse(significant_scripts(chemistry))  # 但无实质外语文字

    def test_significant_scripts_still_detects_foreign_writing(self):
        self.assertIn("chinese", significant_scripts("答案是 (A) because SN2"))
        self.assertIn("fullwidth_cjk", significant_scripts("答案为（B）"))
        self.assertIn("cyrillic", significant_scripts("Привет"))
        self.assertIn("greek", significant_scripts("The angle is α"))
        self.assertNotIn("latin_extended", significant_scripts("café"))
        self.assertNotIn("other_non_latin", significant_scripts("→"))

    def test_chinese_context_marks_run_with_surrounding_text(self):
        text = "Start here " + "常规" + " and more English after"
        snippets = chinese_context(text, context=7)
        self.assertEqual(len(snippets), 1)
        self.assertIn("[[常规]]", snippets[0])
        self.assertEqual(snippets[0], "t here [[常规]] and mo")

    def test_chinese_context_trims_to_context_width(self):
        text = "a" * 30 + "中文混合" + "b" * 30
        (snippet,) = chinese_context(text, context=10)
        self.assertIn("[[中文混合]]", snippet)
        # 前 10 + 中文 4 + 标记 [[ ]] 4 + 后 10
        self.assertEqual(len(snippet), 10 + 4 + 4 + 10)
        self.assertNotIn("a" * 30, snippet)
        self.assertNotIn("b" * 30, snippet)

    def test_chinese_context_multiple_spans_and_cap(self):
        text = "x" * 20 + "甲" + "y" * 20 + "乙" + "z" * 20
        snippets = chinese_context(text, context=5, max_spans=5)
        self.assertEqual(len(snippets), 2)
        self.assertTrue(all("[[甲]]" in s or "[[乙]]" in s for s in snippets))

    def test_chinese_context_ignores_pure_ascii_and_noise(self):
        self.assertEqual(chinese_context("pure english × –"), [])
        self.assertEqual(chinese_context(""), [])
        self.assertEqual(chinese_context(None), [])

    def test_language_check_report_pure_english(self):
        report = language_check(
            ["The correct answer is (A)", "Option B is correct", ""],
            pred_ids=[0, 1, 2],
        )
        self.assertEqual(report["total"], 3)
        self.assertEqual(report["pure_ascii"], 3)
        self.assertEqual(report["has_chinese"], 0)
        self.assertEqual(report["has_non_ascii"], 0)
        self.assertEqual(report["details"], [])

    def test_language_check_report_mixed(self):
        report = language_check(
            ["The correct answer is (A)", "正确答案是B"],
            pred_ids=[0, 1],
        )
        self.assertEqual(report["total"], 2)
        self.assertEqual(report["pure_ascii"], 1)
        self.assertEqual(report["has_chinese"], 1)
        self.assertEqual(report["has_non_ascii"], 1)
        self.assertEqual(len(report["details"]), 1)
        detail = report["details"][0]
        self.assertEqual(detail["id"], 1)
        self.assertTrue(detail["has_chinese"])
        self.assertIn("chinese", detail["scripts"])

    def test_language_check_list_prediction(self):
        # self-consistency 场景下 prediction 可能是 list
        report = language_check([["The answer is A", "这个答案错误"]])
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["has_chinese"], 1)
        self.assertEqual(report["has_non_ascii"], 1)

    def test_language_check_non_str(self):
        report = language_check([None, 123, ["abc"]])
        self.assertEqual(report["total"], 1)
        self.assertEqual(report["has_non_ascii"], 0)


if __name__ == "__main__":
    unittest.main()
