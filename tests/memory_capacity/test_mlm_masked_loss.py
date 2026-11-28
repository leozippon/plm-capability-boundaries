"""CPU regression for MLM corruption and its masked, row-weighted loss."""

from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.capability.memory_capacity.training import (
    compute_mlm_batch_loss,
    sample_mlm_stratified_batch,
)


def test_mlm_corruption_and_masked_loss() -> None:
    sequences = torch.arange(16, dtype=torch.uint8).reshape(4, 4)
    masked_input, targets, mask, n_mask = sample_mlm_stratified_batch(
        sequences,
        generator=torch.Generator().manual_seed(123),
        pos_arange=torch.arange(4),
    )

    assert targets.dtype == torch.long
    assert torch.equal(targets, sequences.long())
    assert mask.dtype == torch.bool
    assert torch.equal(mask.sum(dim=1), n_mask)
    assert torch.equal(n_mask.sort().values, torch.arange(1, 5))
    assert torch.equal(masked_input[mask], torch.full_like(masked_input[mask], 20))
    assert torch.equal(masked_input[~mask], targets[~mask])

    logits = torch.randn(
        4, 4, 20, generator=torch.Generator().manual_seed(456), requires_grad=True
    )
    # Compute each row independently so that rows, not masked tokens, have equal weight.
    row_losses = []
    for row in range(sequences.shape[0]):
        selected_logits = logits[row, mask[row]]
        selected_targets = targets[row, mask[row]]
        token_losses = selected_logits.logsumexp(dim=-1) - selected_logits.gather(
            1, selected_targets.unsqueeze(1)
        ).squeeze(1)
        row_losses.append(token_losses.mean())
    reference = torch.stack(row_losses).mean()
    token_weighted = (torch.stack(row_losses) * n_mask).sum() / n_mask.sum()
    assert not torch.isclose(reference, token_weighted)

    loss = compute_mlm_batch_loss(logits, targets, mask, n_mask)
    assert torch.allclose(loss, reference, rtol=1e-6, atol=1e-7)
    loss.backward()
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[~mask]) == 0
    assert torch.all(logits.grad[mask].abs().sum(dim=-1) > 0)
