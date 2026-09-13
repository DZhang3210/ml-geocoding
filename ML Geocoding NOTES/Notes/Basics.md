- Core of this is asking how, similar are these two strings?
- **So why doesn't geocoding use 1 score?**
	- Because addresses have multiple aspects of various importance.
	- **Example:** "123 Main St" and "456 Main St"
		- They match but the street name is still completely off
	- **Solution:** 
		- Parse the address into pieces
			- number, street-name, street-type, zip, etc...
		- Then compile a similarity score for **each piece**
		-  This allows us to weigh weight certain areas more based on their importance in similarity

### Similarity Scoring
- **Levenshtein Distance**
	- Counts the minimum number of single-character insertions, deletions, or substitutions, to turn one string into the other.
		- **Example:** "Main St" -> "Main Street"
	- **Pros:** simple and intuitive
	- **Cons:** not reflective of how address typos typically behave
- **Jaro-Wrinkler**
	- tuned for short-strings, like names and addresses
	- Scores/Rewards based on position
	- give extra weight to **common prefixes**
	- **Example:** "123 Main St" and "123 Main Street"
		- Common prefix for first 11 characters

### But what actually is geocoding?
1) Isolate and identify candidates, so let's first use some basic heuristics to narrow down the search to a short list of plausible candidates
	- **What kind of heuristics?**: No ML yet, right now we're just matching w/ simply stuff like matching zip, matching city, etc
2) We've generated 10-50 candidates, **Now it's time to start matching**
	- We also assume a minimum threshold, in case nothing is a good fit
	- **This is the part that gets augment in article 1b**

### How do we build a dataset?
- **Maybe there's an online dataset for geocoding?**
	- No, there is not. There are many proprietary datasets by companies like USPS, Google, or Smarty Streets, but they don't give it away because it's **part of their actual commercial product**
- **So what do we do?**
	- We may not be able to get the incorrect datasets, but we can get a correct dataset, then add error/noise to simulate our own mistakes
	- This is also what paper 1b does
	- **But wouldn't that cause noise, and there's no way that algorithmic noises, matches the true population of errors**
		- Your right, and that's a big limitation, will get addressed later

### What is OpenAddress?
- It's a freely, open-licensed, global address dataset, that aggregates address data that various governments have already published as open data, and **standardizes** it into one common format
- Creates a usable dataset without needing to reverse-engineer different local schemas