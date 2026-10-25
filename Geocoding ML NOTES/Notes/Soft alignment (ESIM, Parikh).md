# Soft alignment: how ESIM / Parikh et al. compare two addresses word by word

*Worked out 2026-10-23 (Step 4 reading). Applies to ESIM §3.2 (Chen et al. 2017) and the "Attend" step of Parikh et al. 2016.*

## The idea in one line
Line up the pieces of the two addresses (`12`↔`12`, `MAIN`↔`MAIN`, `STREET`↔`ST`), then check each piece against its counterpart, and notice what is left over (`APT 2`). Like comparing two addresses by hand.

Why not compare word 1 with word 1? Words don't line up by position (extra prefixes, dropped tokens, units in different places). Alignment finds the matching piece wherever it is.

Why it should beat Step 3: Step 3 squashes each address into one vector *before* comparing, so a one-digit house-number change is averaged away. Alignment compares `12` directly with its counterpart, so the difference stays on that one word.

## The steps

| Step | What | Shape |
|---|---|---|
| Inputs | a_i = vector of word i of a (an embedding, not an id); b_j = vector of word j of b | a: len_a × d, b: len_b × d |
| 1. Scores e | e_ij = dot product of a_i and b_j, for every pair: how related word i of a is to word j of b | len_a × len_b |
| 2. Weights | softmax on e, which turns scores into percentages (0–1, summing to 1). **Down a column** (over a's words) for α; **along a row** (over b's words) for β | len_a × len_b |
| 3. Weighted words | each cell (i, j): a_i's vector × its percentage. Each cell is a vector | len_a × len_b × d |
| 4. Sum (the outer Σ) | add each column's weighted words → one vector per word of b. That is α_j: b_j translated into a ("what in a does b_j line up with?") | α: len_b × d |
| β (other direction) | same e, softmax along rows, sum b's weighted vectors → a_i translated into b | β: len_a × d |
| 5. Compare | put each word next to its translation: `[word; translation; word − translation; word ⊙ translation]` (ESIM). Small difference = matched, large = no counterpart | len_a × 4d, len_b × 4d |

Two sums in the formula, don't mix them up:
- **Denominator Σ:** the softmax normaliser. It adds scores (scalars) so the weights sum to 1.
- **Outer Σ:** adds the weighted word *vectors* → one vector (d,). Nothing is compressed to a scalar.

The scalar weight only says **how much** of a word to take ("match strength with the b word I'm currently building a translation for"); the vector carries the content. Each word of a has a different weight for each word of b. Which b word the weight belongs to = which translation you are building.

## Toy example
a = `MAIN STREET APT`, b = `MAIN ST`, d = 2

```
a: MAIN=[3,0]  STREET=[0,3]  APT=[-1,-1]        b: MAIN=[3,0]  ST=[0,2.8]

e          b:MAIN  b:ST        softmax rows (β)    b:MAIN  b:ST
a:MAIN        9     0                                1.00   0.00
a:STREET      0    8.4                               0.00   1.00
a:APT        -3   -2.8                               0.45   0.55

β_MAIN   = 1.00·[3,0] + 0.00·[0,2.8] = [3.0, 0.0]    ≈ b's MAIN
β_STREET = 0.00·[3,0] + 1.00·[0,2.8] = [0.0, 2.8]    ≈ b's ST
β_APT    = 0.45·[3,0] + 0.55·[0,2.8] = [1.35, 1.54]  blur: no counterpart

word − translation:  MAIN [0,0]  STREET [0,0.2]  APT [-2.35,-2.54]  ← APT is unmatched
```

Each cell of a translation is one embedding dimension (one learned feature), blended across words using the weights.

## Why both directions (two "translation matrices")
- a → b (β) checks: does every word of a have a counterpart in b?
- b → a (α) checks: does every word of b have a counterpart in a?

With one direction only, an odd word in b (e.g. b = `12 MAIN ST`, a = `MAIN ST`: b's `12`) gets a low weight from every a word, so it barely shows up in any translation and is **never checked**. Its own turn (b → a) is what finds it unmatched.

One direction would also make score(a, b) ≠ score(b, a), the same problem the Step 3 swap test guarded against. The second direction is almost free: it reuses the same e.

## After alignment (ESIM)
BiLSTM over each address's compare vectors → mean + max pooling per address → `[a_avg; a_max; b_avg; b_max]` (fixed size, 4h) → MLP → one match score. Parikh et al. do the same with a sum ("Aggregate") instead of the BiLSTM + pooling.

## Side notes
- Softmax ≠ scaling. Transformers divide scores by √d before softmax to stop huge dot products from making softmax one-hot (vanishing gradients). ESIM doesn't scale.
- Parikh et al. have no BiLSTM: each word goes through a feed-forward F alone, so no context from neighbours. ESIM adds a BiLSTM before alignment.
- In batched code, e is (batch, max_len_a, max_len_b): padded words must be masked out of the softmax (same idea as Step 3 masked pooling).
