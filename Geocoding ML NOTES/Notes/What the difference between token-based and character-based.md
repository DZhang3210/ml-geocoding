**Character-based** are much more easily able to handle typos and are precisely meant to handle that. In comparison, **token-based** methods first split up all the characters and then try to compare them. This means they are unaffected by ordering and can more easily handle semantically similar examples
- `Blue Street = Street Blue`
In comparison, token-based methods have a much harder time of dealing with typos
- `Blee Streot = Blue Street ??` (character-based is confused)
In comparison **character-based** are precisely meant to be able to handle typos, so there's a seperation of responsibilities