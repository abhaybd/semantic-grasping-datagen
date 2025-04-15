import warnings
from collections import defaultdict

from langchain_core.load import dumps, loads
from langchain_openai import ChatOpenAI


MODEL_CONSTANTS = {
    "gpt-4o": dict(
        # https://platform.openai.com/settings/organization/limits
        # https://platform.openai.com/docs/pricing
        # https://platform.openai.com/docs/models/gpt-4o
        COST_PER_INPUT_1K_TOKENS=2.50 / 1_000,
        COST_PER_OUTPUT_1K_TOKENS=10.0 / 1_000,
        COST_PER_INPUT_TOKEN=2.50 / 1_000_000,
        COST_PER_OUTPUT_TOKEN=10.0 / 1_000_000,
        # # Tier 5 account
        # ACCOUNT_TOKEN_LIMITS_TPM=150_000_000,
        # ACCOUNT_REQUEST_LIMITS_RPM=50_000,
        # ACCOUNT_BATCH_QUEUE_LIMITS_TPD=50_000_000_000,
        # MODEL_TOKEN_LIMITS_TPM=30_000_000,
        # MODEL_REQUEST_LIMITS_RPM=10_000,
        # MODEL_BATCH_QUEUE_LIMITS_TPD=5_000_000_000,
    ),
    "gpt-4o-mini": dict(
        # https://platform.openai.com/settings/organization/limits
        # https://platform.openai.com/docs/pricing
        # https://platform.openai.com/docs/models/gpt-4o
        COST_PER_INPUT_1K_TOKENS=0.15 / 1_000,
        COST_PER_OUTPUT_1K_TOKENS=0.60 / 1_000,
        COST_PER_INPUT_TOKEN=0.15 / 1_000_000,
        COST_PER_OUTPUT_TOKEN=0.60 / 1_000_000,
        # # Tier 5 account
        # ACCOUNT_TOKEN_LIMITS_TPM=150_000_000,
        # ACCOUNT_REQUEST_LIMITS_RPM=50_000,
        # ACCOUNT_BATCH_QUEUE_LIMITS_TPD=50_000_000_000,
        # MODEL_TOKEN_LIMITS_TPM=30_000_000,
        # MODEL_REQUEST_LIMITS_RPM=10_000,
        # MODEL_BATCH_QUEUE_LIMITS_TPD=5_000_000_000,
    ),
    "gpt-4.1": dict(
        # https://platform.openai.com/settings/organization/limits
        # https://platform.openai.com/docs/pricing
        # https://platform.openai.com/docs/models/gpt-4o
        COST_PER_INPUT_1K_TOKENS=2.0 / 1_000,
        COST_PER_OUTPUT_1K_TOKENS=8.0 / 1_000,
        COST_PER_INPUT_TOKEN=2.0 / 1_000_000,
        COST_PER_OUTPUT_TOKEN=8.0 / 1_000_000,
        # # Tier 5 account
        # ACCOUNT_TOKEN_LIMITS_TPM=150_000_000,
        # ACCOUNT_REQUEST_LIMITS_RPM=50_000,
        # ACCOUNT_BATCH_QUEUE_LIMITS_TPD=50_000_000_000,
        # MODEL_TOKEN_LIMITS_TPM=30_000_000,
        # MODEL_REQUEST_LIMITS_RPM=10_000,
        # MODEL_BATCH_QUEUE_LIMITS_TPD=5_000_000_000,
    ),
}
MODEL_CONSTANTS["gpt-4o-2024-05-13"] = MODEL_CONSTANTS["gpt-4o"]


class LangchainWrapper:
    def __init__(self, llm):
        self.llm = llm
        self.costs_list = []
        self.serialized = None

    def __call__(self, input, config=None, *, stop=None, log=None, **kwargs):
        if self.llm is None and self.serialized is not None:
            self.reload()

        res = self.llm.invoke(input, config, stop=stop, **kwargs)

        interaction_cost = (
            res.response_metadata["token_usage"]["completion_tokens"]
            / 1000
            * MODEL_CONSTANTS[self.llm.model_name]["COST_PER_OUTPUT_1K_TOKENS"]
            + res.response_metadata["token_usage"]["prompt_tokens"]
            / 1000
            * MODEL_CONSTANTS[self.llm.model_name]["COST_PER_INPUT_1K_TOKENS"]
        )
        self.costs_list.append((log, interaction_cost))
        # print(log, interaction_cost)
        return res.content

    def add_costs(self, log, costs):
        self.costs_list.extend(costs)
        # print(log, costs)

    def total_cost(self):
        return sum(x[1] for x in self.costs_list)

    def print_costs(self):
        print(f"\nAll costs \n{self.costs_list}")

        total_cost = self.total_cost()

        per_concept = defaultdict(float)
        for entry, cost in self.costs_list:
            per_concept[entry] += cost

        print("\nAggregates per concept")
        for contribution in sorted(
            per_concept.keys(), key=lambda x: per_concept[x], reverse=True
        ):
            print(
                f"${per_concept[contribution]:.4f} ({per_concept[contribution] / total_cost * 100:.3f}%) {contribution}"
            )

        print(f"\nTotal cost ${total_cost:.4f}")

    def offload(self):
        res = self.llm
        self.llm = None
        self.serialized = dumps(res)
        return res

    def reload(self, orig_llm=None):
        if orig_llm is not None:
            self.llm = orig_llm
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.llm = loads(self.serialized)
        self.serialized = None
