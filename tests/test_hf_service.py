import pytest
from unittest.mock import patch, MagicMock
from app.services.hf_service import HuggingFaceService

def test_push_dataset():
    data = [{"q": "test", "a": "pass"}]
    with patch("app.services.hf_service.login") as mock_login,          patch("app.services.hf_service.Dataset") as mock_dataset:

        mock_ds_instance = MagicMock()
        mock_dataset.from_list.return_value = mock_ds_instance

        result = HuggingFaceService.push_dataset(data, "user/repo", "token")

        assert "Successfully pushed" in result
        mock_login.assert_called_with(token="token", add_to_git_credential=False)
        # Expect split='train' default
        mock_ds_instance.push_to_hub.assert_called_with(repo_id="user/repo", private=False, split="train")

def test_modify_dataset_append():
    new_data = [{"q": "new"}]
    with patch("app.services.hf_service.login"),          patch("app.services.hf_service.load_dataset") as mock_load,          patch("app.services.hf_service.concatenate_datasets") as mock_concat:

        mock_existing = MagicMock()
        mock_load.return_value = mock_existing
        mock_combined = MagicMock()
        mock_concat.return_value = mock_combined

        # Use keyword args to be safe
        result = HuggingFaceService.modify_dataset(
            repo_id="user/repo",
            token="token",
            new_data=new_data,
            operation="append_rows"
        )

        assert "Successfully modified" in result
        mock_concat.assert_called()
        # Verify call to push_to_hub on combined dataset
        # It calls combined.push_to_hub(repo_id=..., config_name=..., split=...)
        # We need to check args roughly
        mock_combined.push_to_hub.assert_called()
        args, kwargs = mock_combined.push_to_hub.call_args
        assert kwargs["repo_id"] == "user/repo"
        assert kwargs["split"] == "train"
