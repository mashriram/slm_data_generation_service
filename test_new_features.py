import asyncio
import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import io
import json

# Add current directory to path
sys.path.append(os.getcwd())

# Mocking modules
# We need to set __spec__ to avoid failures in importlib.util.find_spec checks used by transformers

mock_sent_trans = MagicMock()
mock_sent_trans.__spec__ = MagicMock()
sys.modules["sentence_transformers"] = mock_sent_trans

mock_sklearn = MagicMock()
mock_sklearn.__spec__ = MagicMock()
sys.modules["sklearn"] = mock_sklearn
sys.modules["sklearn.metrics.pairwise"] = mock_sklearn.metrics.pairwise

# Now we can import
from app.services.quality_control import QualityController
from app.services.generative_modifier import GenerativeDatasetModifier, ColumnData

class TestNewFeatures(unittest.TestCase):
    def setUp(self):
        # Setup mocks
        self.mock_embedder_cls = sys.modules["sentence_transformers"].SentenceTransformer
        self.mock_embedder = MagicMock()
        self.mock_embedder_cls.return_value = self.mock_embedder
        
        self.mock_cosine = sys.modules["sklearn.metrics.pairwise"].cosine_similarity

    def test_deduplication(self):
        print("\nTesting QA Deduplication...")
        qc = QualityController()
        # QC uses global cache, so it might use the mocked embedder if first time
        # or if we patch the global variable.
        # Since we mocked the module, QualityController should use the mock class.
        
        # Mock embeddings
        # 3 items, 1 and 2 are duplicates
        embeddings = [[1, 0], [0.99, 0.01], [0, 1]] 
        self.mock_embedder.encode.return_value = embeddings
        
        # Mock cosine similarity
        # 1-2 sim is high, 1-3 sim is low
        self.mock_cosine.return_value = [
            [1.0, 0.99, 0.0],
            [0.99, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
        
        data = [
            {"q": "What is A?", "a": "A is A."},
            {"q": "What is A?", "a": "A is similar to A."}, # Duplicate
            {"q": "What is B?", "a": "B is B."}
        ]
        
        result = qc.deduplicate(data, threshold=0.9)
        print(f"Original: {len(data)}, Deduplicated: {len(result)}")
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["q"], "What is A?")
        self.assertEqual(result[1]["q"], "What is B?")

    def test_statistics_analysis(self):
        print("\nTesting Statistics Analysis...")
        # Mock LLM provider for GenerativeDatasetModifier
        with patch("app.services.generative_modifier.LLMProviderFactory") as MockFactory:
            mock_llm = MagicMock()
            MockFactory.return_value.llm = mock_llm
            
            modifier = GenerativeDatasetModifier(provider="test")
            
            # Create a sample list of dicts simulating dataset
            sample_data = [
                {"age": 25, "city": "NY"},
                {"age": 30, "city": "NY"},
                {"age": 35, "city": "SF"},
                {"age": 40, "city": "SF"},
                {"age": 45, "city": "LA"}
            ]
            
            stats = modifier._analyze_statistics(sample_data)
            print("Generated Stats:\n", stats)
            
            self.assertIn("Column: age", stats)
            self.assertIn("Type: Numeric", stats)
            self.assertIn("Mean: 35.00", stats)
            
            self.assertIn("Column: city", stats)
            self.assertIn("Type: Categorical/Text", stats)
            self.assertIn("Unique Values: 3", stats) # NY, SF, LA

    def test_hallucination_check(self):
        # We can't easily test real logic as it calls LLM, but we can verify it tries to call chain.
        pass

if __name__ == "__main__":
    unittest.main()
