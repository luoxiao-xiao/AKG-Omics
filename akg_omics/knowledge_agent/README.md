# Knowledge Agent

The AKG-Omics Knowledge Agent selects external biological knowledge sources for each task, validates the selected sources against local availability, and returns a reproducible source-selection report.

## Modes

- `rule-only`: deterministic source selection, no external API calls.
- `agent-rule-data`: LLM-guided selection with rule constraints and data-profile coverage signals.
- `agent-only`: LLM-guided selection without mandatory rule anchoring.

If the LLM call fails or no API key is configured, the pipeline falls back to deterministic source selection unless strict failure is explicitly requested.

## DeepSeek V4 Pro configuration

The final AKG-Omics experiments used DeepSeek V4 Pro through a chat-completions API:

```bash
export KNOWLEDGE_AGENT_ENABLE_AGENT=1
export KNOWLEDGE_AGENT_LLM_PROVIDER=deepseek
export KNOWLEDGE_AGENT_LLM_MODEL=deepseek-v4-pro
export KNOWLEDGE_AGENT_LLM_ENDPOINT=https://api.deepseek.com/v1/chat/completions
export DEEPSEEK_API_KEY=<your_deepseek_key>
```

`DEEPSEEK_API_KEY` is read by the public run scripts and forwarded to `KNOWLEDGE_AGENT_LLM_API_KEY` when needed.

## Other providers

The client also supports OpenAI-compatible endpoints, Gemini, Anthropic, Qwen, GLM, and a mock mode for offline debugging:

```bash
export KNOWLEDGE_AGENT_ENABLE_AGENT=1
export KNOWLEDGE_AGENT_LLM_PROVIDER=openai_compatible
export KNOWLEDGE_AGENT_LLM_MODEL=<model_name>
export KNOWLEDGE_AGENT_LLM_ENDPOINT=<chat_completions_endpoint>
export KNOWLEDGE_AGENT_LLM_API_KEY=<your_key>
```

For offline debugging:

```bash
export KNOWLEDGE_AGENT_ENABLE_AGENT=1
export KNOWLEDGE_AGENT_MOCK_RESPONSE_JSON='{"selected_source_ids":["uniprot","reactome"],"notes":["mock"]}'
```

## Probe

```bash
python -m akg_omics.knowledge_agent.probe_agent --task-id he_protein_to_gene --source he,protein --target gene
```