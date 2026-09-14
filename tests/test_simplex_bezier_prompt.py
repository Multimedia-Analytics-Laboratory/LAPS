import unittest

import torch

from src.simplex_bezier_prompt import SimplexBezierPrompt


class SimplexBezierPromptIsolationTest(unittest.TestCase):
    def make_prompt(self, residual_scale=1.0):
        torch.manual_seed(7)
        prompt = SimplexBezierPrompt(
            3, 3, 2, 5, residual_scale=residual_scale,
        )
        with torch.no_grad():
            prompt.controls.normal_()
        return prompt

    def test_isolation_preserves_forward_values(self):
        prompt = self.make_prompt(residual_scale=0.7)
        preferences = torch.tensor([
            [0.2, 0.3, 0.5],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        regular = prompt(preferences)
        isolated = prompt(
            preferences, isolate_vertex_gradients=True,
        )
        torch.testing.assert_close(regular, isolated)

    def test_interior_preference_detaches_all_vertex_controls(self):
        prompt = self.make_prompt()
        prompt(
            torch.tensor([[0.2, 0.3, 0.5]]),
            isolate_vertex_gradients=True,
        ).sum().backward()
        vertices = prompt.multi_indices.eq(prompt.degree).any(-1)
        self.assertEqual(float(prompt.controls.grad[vertices].abs().max()), 0.0)
        self.assertGreater(float(prompt.controls.grad[~vertices].abs().sum()), 0.0)

    def test_exact_apex_updates_only_matching_vertex(self):
        prompt = self.make_prompt()
        prompt(
            torch.tensor([[0.0, 1.0, 0.0]]),
            isolate_vertex_gradients=True,
        ).sum().backward()
        vertices = prompt.multi_indices.eq(prompt.degree).any(-1)
        vertex_tasks = prompt.multi_indices.argmax(-1)
        matching = vertices & vertex_tasks.eq(1)
        other = vertices & ~matching
        self.assertGreater(float(prompt.controls.grad[matching].abs().sum()), 0.0)
        self.assertEqual(float(prompt.controls.grad[other].abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
