from unittest.mock import MagicMock, patch

from gradio_app import generate_data


@patch("requests.post")
def test_generate_data(mock_post):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "success": True,
        "data": [{"question": "Q1", "answer": "A1"}],
    }
    mock_post.return_value = mock_response

    # Mock file paths
    files = ["/tmp/file1.pdf", "/tmp/file2.docx"]
    demo_file = "/tmp/demo.csv"

    # Mock open() to avoid file not found
    with patch("builtins.open", new_callable=MagicMock):
        df, json_out, status = generate_data(
            prompt="Test prompt",
            files=files,
            demo_file=demo_file,
            provider="groq",
            model="llama3",
            temperature=0.7,
            count=5,
            agentic=False,
            mcp_servers=None,
            use_rag=False,
            conserve_tokens=False,
            rate_limit=None,
            hf_repo=None,
            hf_token=None,
            hf_private=False,
            hf_append=False,
        )

    assert "Q1" in df.to_string()
    assert "Success" in status
    assert mock_post.called

    # Verify payload
    args, kwargs = mock_post.call_args
    assert kwargs["data"]["provider"] == "groq"
    assert len(kwargs["files"]) == 3  # 2 files + 1 demo file


if __name__ == "__main__":
    test_generate_data()
