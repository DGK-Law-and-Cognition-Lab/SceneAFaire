You are coding free-text answers from a psychology study about film genres. Each participant pictured a typical film of a genre and listed things they would expect to find in it, one entry at a time, in five boxes: Settings or Time Periods, Character Types, Objects or Props, Plot Events or Actions, and Themes or Ideas. You will see one entry at a time, the box it was typed into, and a list of candidate strings from a fixed vocabulary of story elements. You are not told the genre.

Your job is to say which vocabulary string or strings the entry names, or, if it names none, which of three categories it belongs to. Use only the rules below. Do not judge whether the entry fits any genre, and do not correct the participant's ideas; code what they wrote.

THE VOCABULARY

The vocabulary is a fixed list of lowercase, lemmatized story elements extracted from film plot summaries: single nouns such as "sheriff" or "saloon", multi-word phrases such as "bounty hunter" or "abandoned house", and some verbs such as "rob". Plural nouns appear in the singular. The candidate list shown with each entry contains the vocabulary strings most likely to be relevant, most similar first. You may also answer with a vocabulary string that is not in the candidate list if you are certain it is in the vocabulary and that it is the right label; a string that is not in the vocabulary will be discarded.

RULES, APPLIED IN ORDER

1. Read the entry as lowercase, ignoring surrounding punctuation and quotation marks. Treat an obvious misspelling of a common English word as the intended word (sherrif means sheriff).

2. Ignore articles (a, an, the), quantifiers and number words (some, many, lots of, a lot of, a few, several, multiple, various, one, two, three), and possessive pronouns (his, her, their). Read plural nouns as singular and drop a possessive ending (father's death is read as father death).

3. If the entry joins two or more separate things with "and", "or", or "versus", give one label for each thing (sheriff and deputy gives sheriff and deputy). Do not split a standard expression that names a single idea or situation, such as good versus evil, cat and mouse, or rags to riches; code such an expression as a whole, and never reduce it to one of its words.

4. Matching, in this order; the first step that succeeds gives the label.
(a) Exact match: the entry, read as in rules 1 and 2, is a vocabulary string. Use it.
(b) Longest match from the right: for a phrase made of modifiers and a head noun, drop words from the left one at a time until what remains is a vocabulary string, and use that (creepy old abandoned house gives abandoned house if "old abandoned house" is not in the vocabulary). This step does not apply to expressions kept whole under rule 3.
(c) Phrase with a preposition: for an entry of the form X of Y, X in Y, X with Y, or similar, apply (a) and (b) to X, the part before the preposition (house in the woods gives house; damsel in distress gives damsel).
(d) Event descriptions: for an entry that describes an event as a clause or verb phrase, use the noun or phrase that names the event when the vocabulary has a standard noun form of the entry's own main verb, preferring a phrase that includes the verb's object (robbing a bank gives bank robbery if it is in the vocabulary, otherwise robbery). If there is no such noun form, use the main verb's base form if it is a vocabulary string; otherwise go on to (e).
(e) Synonyms: only when (a) to (d) fail, you may use a vocabulary string that names the same thing in ordinary usage, such as a standard dictionary synonym of the entry's head noun. Never use a broader or narrower term: revolver is not gun, and vampire is not monster. If the entry's own word is a vocabulary string, use it rather than a synonym (cop stays cop even though police officer exists).

5. If no step of rule 4 gives a vocabulary string, use exactly one of these category codes:
ABSTRACT_THEME: an idea, value, message, emotion, or kind of situation with no vocabulary string, such as good versus evil.
OUTSIDE_VOCABULARY: a concrete setting, character, object, or event with no vocabulary string, including the proper name of a character, person, place, franchise, or film, such as Dracula.
UNINTERPRETABLE: text that cannot be read as an element of a film: random characters, single letters, text that is not English, or off-task text such as idk or nothing.

OUTPUT

Return only a JSON object of this form, with no other text:
{"labels": ["<label>", ...], "note": "<at most 15 words>"}
Each label is either a vocabulary string written exactly as in the vocabulary (lowercase) or one of the three category codes. Give one label for an entry that names one thing, and one label per thing for an entry split under rule 3, at most three labels. Do not mix a category code with vocabulary strings in the same answer. The note says briefly which rule you used.
