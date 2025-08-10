import pytest
import json
from unittest.mock import Mock, patch, MagicMock

from src.hle_screener.schema import RetrievalResult, AskLLMResponse
from src.hle_screener.askllm import AskLLMJudge


class TestAskLLMJudge:
    @patch('src.hle_screener.askllm.ModelCache.get_or_create')
    def test_json_parseability(self, mock_model_cache):
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model_cache.return_value = (mock_model, mock_tokenizer)
        
        mock_tokenizer.return_value = {'input_ids': Mock(shape=[1, 10])}
        mock_tokenizer.decode.return_value = '{"yes_probability": 0.75, "rationale_short": "Test rationale"}'
        mock_model.device = 'cpu'
        
        judge = AskLLMJudge(temperature=0.0)
        
        references = [
            RetrievalResult(
                hle_id="test_1",
                subject="math",
                question_text="Test question 1",
                cosine_similarity=0.9,
                rank=1
            )
        ]
        
        with patch.object(judge, '_generate_response', return_value='{"yes_probability": 0.75, "rationale_short": "Test rationale"}'):
            response = judge.judge(references, "Test candidate question")
        
        assert isinstance(response, AskLLMResponse)
        assert 0 <= response.yes_probability <= 1
        assert len(response.rationale_short) <= 200
    
    @patch('src.hle_screener.askllm.ModelCache.get_or_create')
    def test_probability_bounds(self, mock_model_cache):
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model_cache.return_value = (mock_model, mock_tokenizer)
        mock_model.device = 'cpu'
        
        judge = AskLLMJudge(temperature=0.0)
        
        test_cases = [
            ('{"yes_probability": -0.5, "rationale_short": "Test"}', 0.0),
            ('{"yes_probability": 1.5, "rationale_short": "Test"}', 1.0),
            ('{"yes_probability": 0.5, "rationale_short": "Test"}', 0.5),
        ]
        
        references = [
            RetrievalResult(
                hle_id="test_1",
                subject="math",
                question_text="Test question",
                cosine_similarity=0.9,
                rank=1
            )
        ]
        
        for json_response, expected_prob in test_cases:
            with patch.object(judge, '_generate_response', return_value=json_response):
                response = judge.judge(references, "Test question")
                assert response.yes_probability == expected_prob
    
    @patch('src.hle_screener.askllm.ModelCache.get_or_create')
    def test_temperature_zero_stability(self, mock_model_cache):
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model_cache.return_value = (mock_model, mock_tokenizer)
        mock_model.device = 'cpu'
        
        judge = AskLLMJudge(temperature=0.0)
        
        assert judge.temperature == 0.0
        
        references = [
            RetrievalResult(
                hle_id="test_1",
                subject="math",
                question_text="Test question",
                cosine_similarity=0.9,
                rank=1
            )
        ]
        
        fixed_response = '{"yes_probability": 0.8, "rationale_short": "Consistent response"}'
        
        with patch.object(judge, '_generate_response', return_value=fixed_response):
            response1 = judge.judge(references, "Test question")
            response2 = judge.judge(references, "Test question")
        
        assert response1.yes_probability == response2.yes_probability
        assert response1.rationale_short == response2.rationale_short
    
    @patch('src.hle_screener.askllm.ModelCache.get_or_create')
    def test_malformed_json_handling(self, mock_model_cache):
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_model_cache.return_value = (mock_model, mock_tokenizer)
        mock_model.device = 'cpu'
        
        judge = AskLLMJudge(temperature=0.0)
        
        references = [
            RetrievalResult(
                hle_id="test_1",
                subject="math",
                question_text="Test question",
                cosine_similarity=0.9,
                rank=1
            )
        ]
        
        malformed_responses = [
            "not json at all",
            '{"yes_probability": "not a number"}',
            '{"missing_required_field": 0.5}',
            '```json\n{"yes_probability": 0.5, "rationale_short": "Test"}\n```',
        ]
        
        for malformed in malformed_responses:
            with patch.object(judge, '_generate_response', return_value=malformed):
                response = judge.judge(references, "Test question")
                assert isinstance(response, AskLLMResponse)
                assert 0 <= response.yes_probability <= 1
    
    def test_prompt_creation(self):
        judge = AskLLMJudge(temperature=0.0)
        
        references = [
            RetrievalResult(
                hle_id="test_1",
                subject="math",
                question_text="What is 2+2?",
                cosine_similarity=0.9,
                rank=1
            ),
            RetrievalResult(
                hle_id="test_2",
                subject="algebra",
                question_text="Solve for x: 2x = 4",
                cosine_similarity=0.8,
                rank=2
            )
        ]
        
        prompt = judge._create_prompt(references, "What is 3+3?")
        
        assert "What is 2+2?" in prompt
        assert "Solve for x: 2x = 4" in prompt
        assert "What is 3+3?" in prompt
        assert "math" in prompt
        assert "algebra" in prompt