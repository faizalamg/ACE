# Tests for query preprocessing
import unittest
from dataclasses import dataclass
from typing import List

try:
    from ace.query_preprocessor import QueryPreprocessor, PreprocessResult
except ImportError:
    @dataclass
    class PreprocessResult:
        cleaned_query: str
        is_valid_query: bool
        original_query: str
        transformations_applied: List[str]
    
    class QueryPreprocessor:
        def preprocess(self, query: str):
            raise NotImplementedError()

class TestQueryPreprocessor(unittest.TestCase):
    def setUp(self):
        self.preprocessor = QueryPreprocessor()
    
    def test_normalize_text_lowercase(self):
        result = self.preprocessor.preprocess("HELLO WORLD")
        self.assertEqual(result.cleaned_query, "hello world")
        self.assertIn("normalized_case", result.transformations_applied)
    
    def test_normalize_text_strip_whitespace(self):
        result = self.preprocessor.preprocess("  hello world  ")
        self.assertEqual(result.cleaned_query, "hello world")
        self.assertIn("stripped_whitespace", result.transformations_applied)
    
    def test_normalize_text_multiple_spaces(self):
        result = self.preprocessor.preprocess("hello    world")
        self.assertEqual(result.cleaned_query, "hello world")
        self.assertIn("collapsed_spaces", result.transformations_applied)
    
    def test_normalize_punctuation_basic(self):
        result = self.preprocessor.preprocess("what is this???")
        self.assertEqual(result.cleaned_query, "what is this?")
        self.assertIn("normalized_punctuation", result.transformations_applied)
    
    def test_normalize_punctuation_exclamation(self):
        result = self.preprocessor.preprocess("amazing!!!")
        self.assertEqual(result.cleaned_query, "amazing!")
        self.assertIn("normalized_punctuation", result.transformations_applied)
    
    def test_detect_non_query_table_markdown(self):
        table = "|Column|Col2|\n|---|---|\n|a|b|"
        result = self.preprocessor.preprocess(table)
        self.assertFalse(result.is_valid_query)
        self.assertIn("detected_table", result.transformations_applied)
    
    def test_detect_non_query_verdict_format(self):
        verdict = "VERDICT: Success\nReasoning: Tests passed"
        result = self.preprocessor.preprocess(verdict)
        self.assertFalse(result.is_valid_query)
        self.assertIn("detected_verdict", result.transformations_applied)
    
    def test_detect_non_query_command_output(self):
        command = "$ npm test\n> Running..."
        result = self.preprocessor.preprocess(command)
        self.assertFalse(result.is_valid_query)
        self.assertIn("detected_command_output", result.transformations_applied)
    
    def test_detect_non_query_json_dump(self):
        json_dump = '{"key": "value", "nested": {"data": 123}}'
        result = self.preprocessor.preprocess(json_dump)
        self.assertFalse(result.is_valid_query)
        self.assertIn("detected_json", result.transformations_applied)
    
    def test_detect_non_query_code_block(self):
        code_block = "```python\ndef foo():\n    return 'bar'\n```"
        result = self.preprocessor.preprocess(code_block)
        self.assertFalse(result.is_valid_query)
        self.assertIn("detected_code_block", result.transformations_applied)
    
    def test_extract_core_question_yes_prefix(self):
        result = self.preprocessor.preprocess("yes! how do I implement this?")
        self.assertEqual(result.cleaned_query, "how do i implement this?")
        self.assertIn("removed_conversational_wrapper", result.transformations_applied)
    
    def test_extract_core_question_so_prefix(self):
        result = self.preprocessor.preprocess("so, what about error handling?")
        self.assertEqual(result.cleaned_query, "what about error handling?")
        self.assertIn("removed_conversational_wrapper", result.transformations_applied)
    
    def test_extract_core_question_and_prefix(self):
        result = self.preprocessor.preprocess("and how does this work?")
        self.assertEqual(result.cleaned_query, "how does this work?")
        self.assertIn("removed_conversational_wrapper", result.transformations_applied)
    
    def test_extract_core_question_okay_prefix(self):
        result = self.preprocessor.preprocess("okay, where is the config file?")
        self.assertEqual(result.cleaned_query, "where is the config file?")
        self.assertIn("removed_conversational_wrapper", result.transformations_applied)
    
    def test_extract_core_question_multiple_wrappers(self):
        result = self.preprocessor.preprocess("yes! and so, what about performance?")
        self.assertEqual(result.cleaned_query, "what about performance?")
        self.assertIn("removed_conversational_wrapper", result.transformations_applied)
    
    def test_valid_query_question_mark(self):
        result = self.preprocessor.preprocess("what is the capital of France?")
        self.assertTrue(result.is_valid_query)
        self.assertEqual(result.cleaned_query, "what is the capital of france?")
    
    def test_valid_query_imperative(self):
        result = self.preprocessor.preprocess("explain the ACE framework")
        self.assertTrue(result.is_valid_query)
        self.assertEqual(result.cleaned_query, "explain the ace framework")
    
    def test_valid_query_how_to(self):
        result = self.preprocessor.preprocess("how to install dependencies")
        self.assertTrue(result.is_valid_query)
        self.assertEqual(result.cleaned_query, "how to install dependencies")
    
    def test_original_query_preserved(self):
        original = "  YES! What is THIS???  "
        result = self.preprocessor.preprocess(original)
        self.assertEqual(result.original_query, original)
    
    def test_original_query_unchanged_by_transformations(self):
        original = "HELLO WORLD!!!"
        result = self.preprocessor.preprocess(original)
        self.assertEqual(result.original_query, original)
        self.assertNotEqual(result.cleaned_query, original)
    
    def test_empty_query(self):
        result = self.preprocessor.preprocess("")
        self.assertFalse(result.is_valid_query)
        self.assertEqual(result.cleaned_query, "")
    
    def test_whitespace_only_query(self):
        result = self.preprocessor.preprocess("   \n\t   ")
        self.assertFalse(result.is_valid_query)
        self.assertEqual(result.cleaned_query, "")
    
    def test_single_word_query(self):
        result = self.preprocessor.preprocess("help")
        self.assertTrue(result.is_valid_query)
        self.assertEqual(result.cleaned_query, "help")
    
    def test_full_preprocessing_pipeline(self):
        original = "  YES!!! So,   what is  the ANSWER???  "
        result = self.preprocessor.preprocess(original)
        self.assertIn("stripped_whitespace", result.transformations_applied)
        self.assertIn("normalized_case", result.transformations_applied)
        self.assertIn("collapsed_spaces", result.transformations_applied)
        self.assertIn("normalized_punctuation", result.transformations_applied)
        self.assertIn("removed_conversational_wrapper", result.transformations_applied)
        self.assertEqual(result.cleaned_query, "what is the answer?")
        self.assertTrue(result.is_valid_query)
        self.assertEqual(result.original_query, original)

if __name__ == "__main__":
    unittest.main()
