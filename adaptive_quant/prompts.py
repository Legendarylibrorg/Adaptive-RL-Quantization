from __future__ import annotations

import random

from adaptive_quant.types import PromptSample


def default_prompt_library() -> list[PromptSample]:
    return [
        PromptSample("simple_qa", "What is the capital of France?", "qa"),
        PromptSample("short_code", "Write a Python function that reverses a list.", "code"),
        PromptSample("math_reasoning", "Solve step by step: if a train travels 60 miles in 45 minutes, what is the average speed?", "reasoning"),
        PromptSample("chain_of_thought", "Compare the causes of the French Revolution and the Russian Revolution in a concise but analytical paragraph.", "history"),
        PromptSample("sql_generation", "Generate a SQL query to compute monthly revenue grouped by product category and region.", "code"),
        PromptSample("legal_summary", "Summarize the main obligations in a contract termination clause with emphasis on notice periods and indemnification.", "legal"),
        PromptSample("biomed_explanation", "Explain how mRNA vaccines trigger immune responses and note two limitations of the platform.", "biomed"),
        PromptSample("creative_long", "Write an atmospheric scene where a stranded pilot discovers an abandoned weather station on Europa, and hint that the station is still communicating.", "creative"),
        PromptSample("tool_use", "Plan a debugging session for intermittent latency spikes in a distributed inference service. Include hypotheses, instrumentation, and rollback criteria.", "systems"),
        PromptSample("translation", "Translate the following paragraph from English to Japanese while preserving technical terminology about distributed systems and cache invalidation.", "translation"),
        PromptSample("low_complexity", "List three benefits of daily exercise.", "wellness"),
        PromptSample("very_complex", "Design a fault-tolerant serving stack for a multilingual assistant that must satisfy strict privacy, mixed CPU/GPU deployment, and unpredictable latency constraints. Explain tradeoffs and rollout phases.", "systems"),
    ]


class PromptLibrary:
    def __init__(self, prompts: list[PromptSample] | None = None) -> None:
        self.prompts = prompts or default_prompt_library()
        self._by_id = {prompt.prompt_id: prompt for prompt in self.prompts}

    def sample(self, rng: random.Random) -> PromptSample:
        return self.prompts[rng.randrange(len(self.prompts))]

    def split_ids(self, *, rng: random.Random, train_fraction: float) -> tuple[set[str], set[str]]:
        train_fraction = max(0.0, min(1.0, float(train_fraction)))
        ids = [prompt.prompt_id for prompt in self.prompts]
        rng.shuffle(ids)
        cutoff = int(round(len(ids) * train_fraction))
        train_ids = set(ids[:cutoff])
        eval_ids = set(ids[cutoff:])
        if not eval_ids and ids:
            # Ensure eval is non-empty when possible.
            eval_ids.add(ids[-1])
            train_ids.discard(ids[-1])
        if not train_ids and ids:
            train_ids.add(ids[0])
            eval_ids.discard(ids[0])
        return train_ids, eval_ids

    def by_id(self, prompt_id: str) -> PromptSample:
        try:
            return self._by_id[prompt_id]
        except KeyError as exc:
            raise KeyError(f"Unknown prompt id: {prompt_id}") from exc

    def probes(
        self,
        prompt: PromptSample,
        count: int,
        rng: random.Random,
        *,
        allowed_ids: set[str] | None = None,
    ) -> list[PromptSample]:
        if count <= 0:
            return []
        candidates = [
            candidate
            for candidate in self.prompts
            if candidate.prompt_id != prompt.prompt_id and (allowed_ids is None or candidate.prompt_id in allowed_ids)
        ]
        if not candidates:
            return []
        result: list[PromptSample] = []
        for _ in range(count):
            result.append(candidates[rng.randrange(len(candidates))])
        return result

    def probes_deterministic(
        self,
        prompt: PromptSample,
        count: int,
        *,
        allowed_ids: set[str] | None = None,
    ) -> list[PromptSample]:
        """Stable probe order for reproducible stability_penalty (sorted by prompt_id, round-robin)."""
        if count <= 0:
            return []
        candidates = [
            candidate
            for candidate in self.prompts
            if candidate.prompt_id != prompt.prompt_id and (allowed_ids is None or candidate.prompt_id in allowed_ids)
        ]
        if not candidates:
            return []
        ordered = sorted(candidates, key=lambda p: p.prompt_id)
        return [ordered[i % len(ordered)] for i in range(count)]
