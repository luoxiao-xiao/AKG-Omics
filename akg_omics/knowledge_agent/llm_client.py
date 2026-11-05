"""LLM clients for knowledge-source selection."""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
_NEXT_ALLOWED_TS: Dict[str, float] = {}

def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    text = re.sub('^```(?:json)?\\s*', '', text)
    text = re.sub('\\s*```$', '', text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    matches = re.findall('\\{[\\s\\S]*\\}', text)
    for block in reversed(matches):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    return None

def _debug_log(msg: str):
    if str(os.getenv('KNOWLEDGE_AGENT_DEBUG', '0')).strip().lower() in {'1', 'true', 'yes'}:
        print(f'[knowledge_agent][debug] {msg}', file=sys.stderr, flush=True)

def _respect_min_interval(provider: str='default'):
    global _NEXT_ALLOWED_TS
    min_interval = max(0.0, float(os.getenv('KNOWLEDGE_AGENT_LLM_MIN_INTERVAL_SEC', '0')))
    if min_interval <= 0:
        return
    now = time.time()
    next_ts = _NEXT_ALLOWED_TS.get(provider, 0.0)
    if now < next_ts:
        sleep_sec = next_ts - now
        _debug_log(f'llm_rate_limit_sleep provider={provider}: {sleep_sec:.2f}s')
        time.sleep(sleep_sec)
    _NEXT_ALLOWED_TS[provider] = max(time.time(), next_ts) + min_interval

@dataclass
class LLMConfig:
    provider: str
    model: str
    api_key: str
    endpoint: str
    temperature: float = 0.0
    timeout_sec: int = 60
    max_tokens: int = 4096

class BaseLLMClient:

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.last_call_trace: Dict[str, Any] = {}

    def generate_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def _make_retry_loop(self, call_fn, max_retries: int, backoff: float):
        """Generic retry loop. call_fn(attempt) -> result or raises."""
        retry_status = {408, 409, 425, 429, 500, 502, 503, 504}
        for attempt in range(max_retries + 1):
            _respect_min_interval(self.cfg.provider)
            try:
                return call_fn(attempt)
            except urllib.error.HTTPError as e:
                body_text = e.read().decode('utf-8', errors='ignore')
                snippet = body_text[:280].replace('\n', ' ')
                status = int(getattr(e, 'code', 0) or 0)
                _debug_log(f'{self.cfg.provider}_call_failed: HTTP {status} attempt={attempt + 1} body={snippet}')
                self.last_call_trace.update({'ok': False, 'error_type': 'HTTPError', 'status': status, 'error_body_snippet': snippet})
                if status in retry_status and attempt < max_retries:
                    retry_after = e.headers.get('Retry-After', '') if getattr(e, 'headers', None) else ''
                    sleep_sec = max(0.1, float(retry_after)) if str(retry_after).strip().isdigit() else min(60.0, backoff * 2 ** attempt)
                    time.sleep(sleep_sec)
                    continue
                return None
            except (urllib.error.URLError, TimeoutError, KeyError, TypeError, ValueError, json.JSONDecodeError) as e:
                _debug_log(f'{self.cfg.provider}_call_failed: {type(e).__name__}: {e} attempt={attempt + 1}')
                self.last_call_trace.update({'ok': False, 'error_type': type(e).__name__, 'error': str(e)})
                if attempt < max_retries:
                    time.sleep(min(60.0, backoff * 2 ** attempt))
                    continue
                return None
        return None

class OpenAICompatibleClient(BaseLLMClient):

    def generate_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        body: Dict[str, Any] = {'model': self.cfg.model, 'temperature': self.cfg.temperature, 'max_tokens': self.cfg.max_tokens, 'messages': [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': json.dumps(user_payload, ensure_ascii=False)}]}
        thinking = str(os.getenv('KNOWLEDGE_AGENT_LLM_THINKING', '')).strip().lower()
        if thinking in {'enabled', 'disabled'}:
            body['thinking'] = {'type': thinking}
        reasoning_effort = str(os.getenv('KNOWLEDGE_AGENT_LLM_REASONING_EFFORT', '')).strip().lower()
        if reasoning_effort:
            body['reasoning_effort'] = reasoning_effort
        disable_rf = str(os.getenv('KNOWLEDGE_AGENT_LLM_DISABLE_RESPONSE_FORMAT', '0')).strip().lower() in {'1', 'true', 'yes'}
        if not disable_rf:
            body['response_format'] = {'type': 'json_object'}
        max_retries = max(0, int(os.getenv('KNOWLEDGE_AGENT_LLM_MAX_RETRIES', '2')))
        backoff = max(0.1, float(os.getenv('KNOWLEDGE_AGENT_LLM_RETRY_BACKOFF_SEC', '1.0')))
        attempt_t0 = [time.time()]

        def call_fn(attempt: int):
            attempt_t0[0] = time.time()
            self.last_call_trace = {'provider': self.cfg.provider, 'model': self.cfg.model, 'endpoint': self.cfg.endpoint, 'system_prompt': system_prompt, 'user_payload': user_payload, 'request_body': dict(body), 'attempt': attempt + 1}
            req = urllib.request.Request(self.cfg.endpoint, data=json.dumps(body).encode('utf-8'), headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {self.cfg.api_key}'}, method='POST')
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_sec) as resp:
                raw = resp.read().decode('utf-8', errors='ignore')
                payload = json.loads(raw)
                choices = payload.get('choices')
                if not isinstance(choices, list) or not choices:
                    raise ValueError(f'No choices in response: {raw[:280]}')
                msg = choices[0].get('message', {}) if isinstance(choices[0], dict) else {}
                content = msg.get('content', '')
                reasoning_content = msg.get('reasoning_content', '')
                parsed = _extract_json_block(content)
                self.last_call_trace.update({'ok': parsed is not None, 'latency_sec': round(time.time() - attempt_t0[0], 3), 'raw_response': raw, 'reasoning_content': reasoning_content, 'message_content': content, 'message_content_snippet': content[:1000], 'parse_error': '' if parsed is not None else 'no_valid_json_object', 'parsed_response': parsed})
                return parsed
        return self._make_retry_loop(call_fn, max_retries, backoff)

class GeminiClient(BaseLLMClient):

    def _build_url(self) -> str:
        endpoint = self.cfg.endpoint.strip()
        if endpoint:
            return endpoint
        model_name = urllib.parse.quote(self.cfg.model, safe='')
        key = urllib.parse.quote(self.cfg.api_key, safe='')
        return f'https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key}'

    def generate_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        prompt = 'Return strict JSON object only.\n' + json.dumps(user_payload, ensure_ascii=False)
        body = {'system_instruction': {'parts': [{'text': system_prompt}]}, 'contents': [{'role': 'user', 'parts': [{'text': prompt}]}], 'generationConfig': {'temperature': self.cfg.temperature, 'responseMimeType': 'application/json', 'maxOutputTokens': self.cfg.max_tokens}}
        max_retries = max(0, int(os.getenv('KNOWLEDGE_AGENT_LLM_MAX_RETRIES', '2')))
        backoff = max(0.1, float(os.getenv('KNOWLEDGE_AGENT_LLM_RETRY_BACKOFF_SEC', '1.0')))
        attempt_t0 = [time.time()]

        def call_fn(attempt: int):
            attempt_t0[0] = time.time()
            self.last_call_trace = {'provider': self.cfg.provider, 'model': self.cfg.model, 'endpoint': self._build_url().split('?key=', 1)[0], 'system_prompt': system_prompt, 'user_payload': user_payload, 'request_body': body, 'attempt': attempt + 1}
            req = urllib.request.Request(self._build_url(), data=json.dumps(body).encode('utf-8'), headers={'Content-Type': 'application/json'}, method='POST')
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_sec) as resp:
                raw = resp.read().decode('utf-8', errors='ignore')
                payload = json.loads(raw)
                candidates = payload.get('candidates')
                if not isinstance(candidates, list) or not candidates:
                    raise ValueError(f'No candidates in Gemini response: {raw[:280]}')
                parts = candidates[0]['content']['parts']
                text = '\n'.join([p.get('text', '') for p in parts if isinstance(p, dict)])
                parsed = _extract_json_block(text)
                self.last_call_trace.update({'ok': parsed is not None, 'latency_sec': round(time.time() - attempt_t0[0], 3), 'raw_response': raw, 'message_content': text, 'message_content_snippet': text[:1000], 'parse_error': '' if parsed is not None else 'no_valid_json_object', 'parsed_response': parsed})
                return parsed
        return self._make_retry_loop(call_fn, max_retries, backoff)

class AnthropicClient(BaseLLMClient):

    def generate_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        body: Dict[str, Any] = {'model': self.cfg.model, 'max_tokens': self.cfg.max_tokens, 'temperature': self.cfg.temperature, 'system': system_prompt + ' Return one valid JSON object only.', 'messages': [{'role': 'user', 'content': json.dumps(user_payload, ensure_ascii=False)}]}
        thinking_budget = int(os.getenv('KNOWLEDGE_AGENT_CLAUDE_THINKING_BUDGET', '0'))
        if thinking_budget > 0:
            body['thinking'] = {'type': 'enabled', 'budget_tokens': thinking_budget}
            body.pop('temperature', None)
        max_retries = max(0, int(os.getenv('KNOWLEDGE_AGENT_LLM_MAX_RETRIES', '2')))
        backoff = max(0.1, float(os.getenv('KNOWLEDGE_AGENT_LLM_RETRY_BACKOFF_SEC', '1.0')))
        attempt_t0 = [time.time()]

        def call_fn(attempt: int):
            attempt_t0[0] = time.time()
            self.last_call_trace = {'provider': self.cfg.provider, 'model': self.cfg.model, 'endpoint': self.cfg.endpoint, 'system_prompt': system_prompt, 'user_payload': user_payload, 'request_body': dict(body), 'attempt': attempt + 1}
            req = urllib.request.Request(self.cfg.endpoint, data=json.dumps(body).encode('utf-8'), headers={'Content-Type': 'application/json', 'x-api-key': self.cfg.api_key, 'anthropic-version': os.getenv('KNOWLEDGE_AGENT_ANTHROPIC_VERSION', '2023-06-01')}, method='POST')
            with urllib.request.urlopen(req, timeout=self.cfg.timeout_sec) as resp:
                raw = resp.read().decode('utf-8', errors='ignore')
                payload = json.loads(raw)
                content = payload.get('content')
                if not isinstance(content, list) or not content:
                    raise ValueError(f'No content in Anthropic response: {raw[:280]}')
                text_parts = []
                thinking_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get('type') == 'text':
                            text_parts.append(str(part.get('text', '')))
                        elif part.get('type') == 'thinking':
                            thinking_parts.append(str(part.get('thinking', '')))
                text = '\n'.join(text_parts)
                parsed = _extract_json_block(text)
                self.last_call_trace.update({'ok': parsed is not None, 'latency_sec': round(time.time() - attempt_t0[0], 3), 'raw_response': raw, 'reasoning_content': '\n'.join(thinking_parts), 'message_content': text, 'message_content_snippet': text[:1000], 'parse_error': '' if parsed is not None else 'no_valid_json_object', 'parsed_response': parsed})
                return parsed
        return self._make_retry_loop(call_fn, max_retries, backoff)



def _make_openai_compatible(provider: str, model: str, api_key: str, endpoint: str, temperature: float, timeout_sec: int, max_tokens: int) -> OpenAICompatibleClient:
    cfg = LLMConfig(provider=provider, model=model, api_key=api_key, endpoint=endpoint, temperature=temperature, timeout_sec=timeout_sec, max_tokens=max_tokens)
    return OpenAICompatibleClient(cfg)

def create_llm_client_from_env() -> Optional[BaseLLMClient]:
    enable = str(os.getenv('KNOWLEDGE_AGENT_ENABLE_AGENT', os.getenv('KNOWLEDGE_AGENT_USE_LLM', '0'))).strip().lower()
    if enable not in {'1', 'true', 'yes'}:
        return None
    provider = str(os.getenv('KNOWLEDGE_AGENT_LLM_PROVIDER', os.getenv('KNOWLEDGE_AGENT_PROVIDER', 'gemini'))).strip().lower()
    temperature = float(os.getenv('KNOWLEDGE_AGENT_LLM_TEMPERATURE', '0'))
    timeout_sec = int(os.getenv('KNOWLEDGE_AGENT_LLM_TIMEOUT_SEC', '60'))
    max_tokens = int(os.getenv('KNOWLEDGE_AGENT_LLM_MAX_TOKENS', '4096'))
    if provider == 'gemini':
        api_key = os.getenv('KNOWLEDGE_AGENT_GEMINI_API_KEY', os.getenv('KNOWLEDGE_AGENT_LLM_API_KEY', ''))
        model = os.getenv('KNOWLEDGE_AGENT_LLM_MODEL', 'gemini-2.0-flash')
        endpoint = os.getenv('KNOWLEDGE_AGENT_LLM_ENDPOINT', '')
        if not api_key:
            return None
        cfg = LLMConfig(provider=provider, model=model, api_key=api_key, endpoint=endpoint, temperature=temperature, timeout_sec=timeout_sec, max_tokens=max_tokens)
        return GeminiClient(cfg)
    if provider in {'openai', 'openai_compatible', 'custom'}:
        api_key = os.getenv('KNOWLEDGE_AGENT_LLM_API_KEY', os.getenv('OPENAI_API_KEY', ''))
        model = os.getenv('KNOWLEDGE_AGENT_LLM_MODEL', 'gpt-4o-mini')
        endpoint = os.getenv('KNOWLEDGE_AGENT_LLM_ENDPOINT', 'https://api.openai.com/v1/chat/completions')
        if not api_key:
            return None
        return _make_openai_compatible(provider, model, api_key, endpoint, temperature, timeout_sec, max_tokens)
    if provider in {'anthropic', 'claude'}:
        api_key = os.getenv('KNOWLEDGE_AGENT_ANTHROPIC_API_KEY', os.getenv('KNOWLEDGE_AGENT_LLM_API_KEY', ''))
        model = os.getenv('KNOWLEDGE_AGENT_LLM_MODEL', 'claude-3-5-sonnet-20241022')
        endpoint = os.getenv('KNOWLEDGE_AGENT_LLM_ENDPOINT', 'https://api.anthropic.com/v1/messages')
        if not api_key:
            return None
        cfg = LLMConfig(provider=provider, model=model, api_key=api_key, endpoint=endpoint, temperature=temperature, timeout_sec=timeout_sec, max_tokens=max_tokens)
        return AnthropicClient(cfg)
    if provider == 'deepseek':
        api_key = os.getenv('KNOWLEDGE_AGENT_DEEPSEEK_API_KEY', os.getenv('KNOWLEDGE_AGENT_LLM_API_KEY', ''))
        model = os.getenv('KNOWLEDGE_AGENT_LLM_MODEL', 'deepseek-chat')
        endpoint = os.getenv('KNOWLEDGE_AGENT_LLM_ENDPOINT', 'https://api.deepseek.com/v1/chat/completions')
        if not api_key:
            return None
        return _make_openai_compatible('deepseek', model, api_key, endpoint, temperature, timeout_sec, max_tokens)
    if provider == 'qwen':
        api_key = os.getenv('KNOWLEDGE_AGENT_QWEN_API_KEY', os.getenv('KNOWLEDGE_AGENT_LLM_API_KEY', ''))
        model = os.getenv('KNOWLEDGE_AGENT_LLM_MODEL', 'qwen-max')
        endpoint = os.getenv('KNOWLEDGE_AGENT_LLM_ENDPOINT', 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions')
        if not api_key:
            return None
        return _make_openai_compatible('qwen', model, api_key, endpoint, temperature, timeout_sec, max_tokens)
    if provider == 'glm':
        api_key = os.getenv('KNOWLEDGE_AGENT_GLM_API_KEY', os.getenv('KNOWLEDGE_AGENT_LLM_API_KEY', ''))
        model = os.getenv('KNOWLEDGE_AGENT_LLM_MODEL', 'glm-4')
        endpoint = os.getenv('KNOWLEDGE_AGENT_LLM_ENDPOINT', 'https://open.bigmodel.cn/api/paas/v4/chat/completions')
        if not api_key:
            return None
        return _make_openai_compatible('glm', model, api_key, endpoint, temperature, timeout_sec, max_tokens)
    return None

