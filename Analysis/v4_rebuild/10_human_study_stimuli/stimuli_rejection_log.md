# Stimulus rejection log (final draw, October 21, 2026)

Forty-three strings are excluded via MANUAL_EXCLUDE in notebooks/_build_10.py. Each exclusion replaces that item with the next candidate in the cell's seeded queue (seed 20260219). Of the final 336 items, 306 were accepted by a majority of three AI judges on the first draw (October 14). The other 30 are replacements from the October 15 review rounds; Claude, the AI assistant, reviewed them, and the investigator confirmed all 30 on October 22, 2026. The investigator confirmed the 19 first-round rejections.

Excluded strings, in the order and groups entered:

- testing
- new light
- antagonist
- dismay
- yorker
- weight
- west
- parody
- reality
- horror
- charge
- star
- theme
- production
- cold blood
- band
- indian
- entity
- actor
- new film
- new book
- orientation
- screen
- saving
- antic
- operation
- wood
- gas
- footage
- director
- rock
- music
- seal
- shell
- iron
- con
- skin
- performance
- replacement
- superior
- ray
- humor
- old west

Grounds: extraction artifacts (truncations, idiom fragments), meta or analytic words (genre labels, 'theme', 'footage', 'director'), ambiguous nouns, and 'indian', excluded in the first round as a demographic label that is also ambiguous between Native American and South Asian. Per-item judge reasons for the first draw are in stimuli_review_sheet_judged.csv and stimuli_rejects_for_confirmation.md.

October 19, 2026 (withdrawn October 21). "white man" (Western, low prevalence and high distinctiveness) was excluded on the same ground as "black man": plot summaries name a character's race mainly to mark it, so the count reflects how summaries are written rather than how often such characters appear, and a race label is not rated as a genre element. The next Western candidate in that cell, "gang member", replaced it. Because "gang member" is also a top-tercile Crime element, it left the foil candidate lists of the other target genres. The foil draw uses one random sequence shared across target genres, so that change shifted three later foil draws: in Fantasy, "toilet" and "mafia" were replaced by "veteran" and "heiress", and in Science Fiction "cocaine" was replaced by "beating". Item ids within the affected cells were renumbered. All four new items satisfy the cell or foil rule and were reviewed by Claude.

Display text. From October 19, eighteen items are shown with capitals, hyphens, possessives, natural plurals or US spelling applied (DISPLAY_OVERRIDES in the builder; listed in stimuli_report.txt). The pipeline string is kept in the element column and drives every statistic.

October 21, 2026. By investigator decision, five earlier exclusions were withdrawn: "black man" and "white man" (race labels kept as character types) and "orgasm", "intercourse" and "sex" (sexual content kept). Withdrawing an exclusion returns the string to its place in its cell's seeded queue. Only "white man" (Western, low prevalence and high distinctiveness) falls within the twelve items its cell takes, so it returns and "gang member" leaves; the three foils that had shifted on October 19 return as well ("toilet" and "mafia" in Fantasy, "cocaine" in Science Fiction). The other four strings lie well past the end of their queues' draws ("black man" is 21st in a Crime queue that takes 6; "orgasm" 100th, "intercourse" 68th, and "sex" 132nd and 77th in their queues), so they are eligible but not rated. All five are in the generation analysis set.
