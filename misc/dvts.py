"""
Diverse Verifier Tree Search (DVTS) Implementation

This module implements a Monte Carlo Tree Search variant that uses a local LLM (Llama)
for generating candidate solutions and a remote verifier for evaluating them.
The algorithm explores diverse solution paths and selects the highest quality response
through iterative search and verification.

Key components:
- TreeSearchNode: Represents states in the search space
- DiverseVerifierTreeSearch: Implements the core DVTS algorithm with selection,
  expansion, simulation, and backpropagation phases
- Batch processing for efficient verification

Use this for complex reasoning tasks where solution quality matters more than speed.
"""

import asyncio
from dataclasses import dataclass

import numpy as np
from openai import OpenAI
from vllm import LLM, SamplingParams


@dataclass
class TreeSearchNode:
    state: str
    parent: "TreeSearchNode" = None
    children: list["TreeSearchNode"] = None
    visits: int = 0
    value: float = 0.0

    def __post_init__(self):
        self.children = [] if self.children is None else self.children


class DiverseVerifierTreeSearch:
    def __init__(self, model_name: str, openai_api_key: str, max_depth: int = 5, batch_size: int = 4):
        self.main_model = LLM(model_name=model_name)
        self.verifier = OpenAI(api_key=openai_api_key)
        self.max_depth = max_depth
        self.batch_size = batch_size

        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=100, stop=None)

    async def verify_solution(self, solution: str) -> float:
        try:
            response = await self.verifier.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a solution verifier. Rate the following solution's quality from 0 to 1.",
                    },
                    {
                        "role": "user",
                        "content": f"Solution to verify: {solution}\nProvide only the numerical score.",
                    },
                ],
                max_tokens=10,
                temperature=0,
            )
            return float(response.choices[0].message.content.strip())
        except Exception as e:
            print(f"Verification error: {e}")
            return 0.0

    async def batch_verify_solutions(self, solutions: list[str]) -> list[float]:
        tasks = [self.verify_solution(solution) for solution in solutions]
        return await asyncio.gather(*tasks)

    def _select(self, node: TreeSearchNode) -> TreeSearchNode:
        while node.children:
            # UCB1 formula with exploration constant.
            ucb_values = [
                (child.value / (child.visits + 1e-6)) + np.sqrt(2 * np.log(node.visits + 1) / (child.visits + 1e-6))
                for child in node.children
            ]
            node = node.children[np.argmax(ucb_values)]
        return node

    async def _expand(self, node: TreeSearchNode) -> list[TreeSearchNode]:
        outputs = self.main_model.generate(prompts=[node.state], sampling_params=self.sampling_params)

        new_states = [output.outputs[0].text for output in outputs]
        new_nodes = [TreeSearchNode(state=state, parent=node) for state in new_states]
        node.children.extend(new_nodes)
        return new_nodes

    async def _simulate_batch(self, nodes: list[TreeSearchNode]) -> list[float]:
        solutions = [node.state for node in nodes]
        return await self.batch_verify_solutions(solutions)

    def _backpropagate(self, node: TreeSearchNode, value: float):
        while node:
            node.visits += 1
            node.value += value
            node = node.parent

    async def search(self, prompt: str, n_iterations: int = 10) -> str:
        root = TreeSearchNode(state=prompt)

        for _ in range(n_iterations):
            # Selection.
            leaf = self._select(root)

            # Expansion.
            if leaf.visits > 0 and len(leaf.children) < self.max_depth:
                new_nodes = await self._expand(leaf)

                # Batch simulation.
                values = await self._simulate_batch(new_nodes)

                # Backpropagation.
                for node, value in zip(new_nodes, values):
                    self._backpropagate(node, value)

        # Return best solution
        best_child = max(root.children, key=lambda c: c.value / (c.visits + 1e-6))
        return best_child.state


async def solve_problem(prompt: str, openai_api_key: str) -> str:
    dvts = DiverseVerifierTreeSearch(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        openai_api_key=openai_api_key,
        max_depth=5,
        batch_size=4,
    )

    solution = await dvts.search(prompt, n_iterations=10)
    return solution


async def main():
    problem = "Explain how quantum computing works."
    solution = await solve_problem(problem, openai_api_key="your-api-key")
    print(f"Problem: {problem}")
    print(f"Solution: {solution}")


if __name__ == "__main__":
    asyncio.run(main())
