# Week 6: your task

You write the physics of the physics-informed network. The file is `simulator/physics_loss.py`. The specification is
`tests/test_physics_loss.py`: 19 tests, each with hand-calculated numbers. Everything else (the data loader, the network, the
training loop and the comparison) is already built and waiting for your four functions.

## Order of work
1. Read the docstring at the top of `physics_loss.py`. It explains the two facts being enforced and what each function must do.
2. Run the tests and watch them fail: `python -m pytest tests/test_physics_loss.py -v`
3. Write the functions in this order, running the tests after each one:
   `to_absolute`, then `bounds_penalty`, then `no_regrowth_penalty`, then `combined_loss`.
4. When all 19 pass, run `python simulator\train_pinn.py` (about 5 to 15 minutes; it resumes if interrupted).
5. Write in `docs/decisions.md`, in your own words: what each penalty means physically, and what the results show.

## A refresher on the PyTorch you need (an unrelated example)
```python
import torch
x = torch.tensor([[1.0, -2.0, 3.0],
                  [0.5,  0.5, -1.0]])        # shape (2, 3): two rows, three columns
x[:, 1]                                      # the second column, shape (2,)
torch.relu(-x[:, 1])                         # how far column 1 is below zero (0 where it is not)
(torch.relu(-x[:, 1]) / 0.1) ** 2            # scaled and squared
torch.stack([x[:, 0], x[:, 2]], dim=1)       # glue columns back into shape (2, 2)
v.sum(dim=1)                                 # add up across the columns of each row
v.mean()                                     # average over everything: a single 0-d number
```

## When a test fails
- Read the test's name and docstring: it says what it expects and often shows the arithmetic.
- Print intermediate tensors (`print(a)`), and check the shapes with `.shape`.
- Common slips: summing where the test wants a mean, forgetting to divide by `VIOLATION_SCALE`, mixing standardised outputs and metres
  (multiply by `y_std` to get metres), or using column 1 where you need column 2.
- Still stuck? Paste your function and the failing test's message and ask for a hint. Ask for hints, not for the answer.

## Why these two facts
The data-only network is never told that a skull cannot be negative, that the 1150 C line cannot lie beyond the solid wall, or that
wear is irreversible. These are true of every real hearth, so they are physics the network can be given at almost no risk.
