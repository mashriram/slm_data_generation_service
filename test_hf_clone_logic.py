import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from app.services.hf_service import HuggingFaceService

class TestHFCloneLogic(unittest.TestCase):
    @patch("app.services.hf_service.load_dataset")
    @patch("app.services.hf_service.HfApi")
    @patch("app.services.hf_service.Dataset")
    @patch("app.services.hf_service.login")
    @patch("app.services.hf_service.concatenate_datasets")
    def test_clone_on_403(self, mock_concat, mock_login, mock_dataset_cls, mock_hf_api, mock_load_dataset):
        print("\n--- Testing Clone Logic on 403 Error ---")
        
        # Setup Mocks
        mock_ds = MagicMock()
        # Make push_to_hub raise a 403 error
        mock_ds.push_to_hub.side_effect = Exception("403 Client Error: Forbidden for url")
        mock_load_dataset.return_value = mock_ds
        
        # Mock HfApi
        mock_api_instance = MagicMock()
        mock_hf_api.return_value = mock_api_instance
        mock_api_instance.whoami.return_value = {"name": "test_user"}
        
        # Make dataset_info raise exception to trigger duplicate_repo
        mock_api_instance.dataset_info.side_effect = Exception("Repo not found")
        
        # Call modify_dataset
        # We expect it to:
        # 1. Try push (fail)
        # 2. Call duplicate_repo
        # 3. Recursively call modify_dataset (which we need to handle or it loops if we don't change mock behavior)
        
        # To avoid infinite recursion in test, we can make the recursive call succeed.
        # But since we are mocking the class method locally, the recursive call goes to the *real* method again.
        # We can patch the class method itself, or just control the mock side effect.
        
        # Actually, the recursive call will call HuggingFaceService.modify_dataset.
        # If we patch `app.services.hf_service.load_dataset` globally, the recursive call also uses it.
        # On the second call, we want `push_to_hub` to SUCCEED.
        
        # Logic to change side effect on second call:
        # Since we use append_rows, we push the COMBINED dataset, which comes from concatenate_datasets
        
        mock_combined = MagicMock()
        mock_concat.return_value = mock_combined
        
        mock_combined.push_to_hub.side_effect = [
            Exception("403 Client Error: Forbidden"), # First call fails
            None # Second call succeeds
        ]
        
        result = HuggingFaceService.modify_dataset(
            repo_id="original/repo",
            token="fake_token",
            new_data=[{"a": 1}],
            operation="append_rows"
        )
        
        print(f"Result: {result}")
        
        # Assertions
        self.assertTrue("Cloned to test_user/repo" in result)
        mock_api_instance.duplicate_repo.assert_called_with(
            repo_id="original/repo",
            new_repo_id="test_user/repo",
            repo_type="dataset",
            private=False
        )
        print("SUCCESS: duplicate_repo was called correctly.")

if __name__ == "__main__":
    unittest.main()
