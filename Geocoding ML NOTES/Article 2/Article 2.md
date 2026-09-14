# Article 1 — Lee, Claridades & Lee (2020)

**Title:** "Improving a Street-Based Geocoding Algorithm Using Machine Learning Techniques"
**Venue:** Applied Sciences, 10(16):5628 (August 13, 2020)
**Link:** https://www.mdpi.com/2076-3417/10/16/5628

### The Summary
So we have multiple different ways of measuring similarities between words; however, none of them can fully solve the problem. This paper aims to solve it, by instead collating many different methods of measuring string similarity and then using an ML model to dynamically learn how to weight each.

Something to note, is that the model never actually touches the words in the normal sense. By the time it's receiving the data, it's in a similarity array. Just like `[0.87, 1.0, 1.0, 0.82, 0.0, 0.0]`, which lets it get a more granular view of things, but can potentially somewhat limit it
