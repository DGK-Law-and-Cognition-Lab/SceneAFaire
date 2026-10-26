**Study 1: Human coding manual for the generation task**

*Scènes à Faire Human Validation Project. Registered with the Study 1 preregistration (Section 6.4) and frozen at registration. It is not revised after data collection begins. This version replaces the hand-coding manual of October 19, 2026.*

**1. Purpose**

The generation responses are coded onto the pipeline's vocabulary by the automated pipeline of Section 6.4 of the preregistration. People code them in two situations, and this manual governs both. In the validation, two coders independently code a random sample of 300 units (50 per genre), and their codes are compared with each other and with the automated coding under the registered decision rule. In the fallback, if that rule is not met, the same two coders code every unit of the study, and the resolved human coding becomes the primary coding. The rules below are the rules of the coder models' rubric (coding_rubric_v1.md), written for people, so that the human and automated codings measure the same thing. They follow the normalization approach of McRae et al. (2005): synonymous phrasings are collapsed to one label, quantifiers are dropped, and compound entries are split.

**2. Materials**

Each coder receives a coding sheet with one row per unit. A unit is an entry, or a part of an entry that the pipeline split at a comma, semicolon, slash, ampersand, plus sign or line break. Each row shows the unit identifier, the box the entry was typed into (settings, characters, objects, events or themes), the entry as typed, and the part to code. The rows are in random order. The sheet does not show the genre, the participant, the participant's other entries, any corpus statistic, the candidate strings the models saw, or any model's labels, and coders do not look any of these up.

The vocabulary is the registered file coding_vocabulary.csv: every narrative (element, type) pair in the pipeline after the franchise blocklist, 17,965 pairs, with each element's type (noun; chunk, for a multi-word phrase; verb; or a named-entity type) and taxonomy label, and no corpus statistics. Coders search it as a spreadsheet, by exact string and by substring.

**3. Procedure**

Each coder works alone. For each unit the coder writes up to three labels in the columns label_1 to label_3 and a short note naming the rule used. A label is either a vocabulary string copied exactly as it appears in the vocabulary (lowercase) or one of the three category codes, written in capitals: ABSTRACT_THEME, OUTSIDE_VOCABULARY or UNINTERPRETABLE. A unit that names one thing receives one label; a unit split under Rule 3 receives one label per thing, at most three. Category codes are not mixed with vocabulary strings for the same unit.

The coders do not discuss the units, see each other's sheets, or see the automated coding until both sheets are complete and submitted. In the validation the independent codes are the data, and they are not resolved. In the fallback, after both coders have finished, disagreements are resolved by discussion, a synonym mapping that both coders accept is applied to every identical unit, and the resolved labels are the ones analyzed.

**4. Rules, applied in order**

Rule 1, reading. Read the unit as lowercase, ignoring surrounding punctuation and quotation marks. Treat an obvious misspelling of a common English word as the intended word (sherrif means sheriff) and note the correction.

Rule 2, function words and word forms. Ignore articles (a, an, the), quantifiers and number words (some, many, lots of, a lot of, a few, several, multiple, various, one, two, three and so on), and possessive pronouns (his, her, their). Read plural nouns as singular (cowboys as cowboy) and drop a possessive ending (father's death is read as father death), because the vocabulary is lemmatized.

Rule 3, splitting. If the unit joins two or more separate things with and, or or versus, give one label for each thing (sheriff and deputy gives sheriff and deputy). Do not split a standard expression that names a single idea or situation, such as good versus evil, cat and mouse, or rags to riches; code it as a whole, and never reduce it to one of its words.

Rule 4, matching, tried in this order. The first step that succeeds gives the label.

(a) Exact match. The unit, read under Rules 1 and 2, is a vocabulary string. Use it.

(b) Longest match from the right. For a phrase of modifiers and a head noun, drop words from the left one at a time until what remains is a vocabulary string, and use it (creepy old abandoned house gives abandoned house, because old abandoned house is not in the vocabulary). This step does not apply to expressions kept whole under Rule 3.

(c) Phrase with a preposition. For a unit of the form X of Y, X in Y, X with Y, or similar, apply (a) and (b) to X, the part before the preposition (house in the woods gives house; damsel in distress gives damsel).

(d) Event descriptions. For a unit that describes an event as a clause or verb phrase, use the noun or phrase that names the event when the vocabulary has a standard noun form of the unit's own main verb, preferring a phrase that includes the verb's object (robbing a bank gives bank robbery, which is in the vocabulary; otherwise it would give robbery). If there is no such noun form, use the main verb's base form if it is a vocabulary string; otherwise go on to (e).

(e) Synonyms. Only when (a) to (d) fail, use a vocabulary string that names the same thing in ordinary usage, such as a standard dictionary synonym of the unit's head noun. Never use a broader or narrower term: revolver is not gun, and vampire is not monster. If the unit's own word is a vocabulary string, use it rather than a synonym (cop stays cop even though police officer exists). Note every synonym used.

Rule 5, unmatched categories. If no step of Rule 4 gives a vocabulary string, use exactly one category code.

ABSTRACT_THEME: an idea, value, message, emotion, or kind of situation with no vocabulary string, such as good versus evil.

OUTSIDE_VOCABULARY: a concrete setting, character, object, or event with no vocabulary string, including the proper name of a character, person, place, franchise, or film, such as Dracula.

UNINTERPRETABLE: text that cannot be read as an element of a film: random characters, single letters, text that is not English, or off-task text such as idk or nothing.

**5. Counting**

A participant produced an element in a genre if at least one of their units for that genre carries that element's label. An element counts once however many units map to it. The analysis decides from the registered predictor file whether a produced element lies in the genre's generation analysis set; coders do not make that decision.

**6. Exclusion rule 6**

Coders make no separate judgment of response quality. Exclusion rule 6 of the preregistration is applied mechanically from the primary coding: a participant is excluded if, in either genre, at least half of their entries are coded UNINTERPRETABLE in every unit, or at least half of their entries only repeat the instructions (the examples locker room, coach or the big game, a box label, a box hint, or a genre name).

**7. Worked examples**

Every vocabulary string in this table was checked against coding_vocabulary.csv.

| Entry | Coded as | Rule |
|---|---|---|
| Cowboys | cowboy | 2, then 4(a) |
| a saloon | saloon | 2, then 4(a) |
| bounty hunters | bounty hunter | 2, then 4(a) |
| creepy old abandoned house | abandoned house | 4(b) |
| house in the woods | house | 4(c) |
| damsel in distress | damsel | 4(c) |
| robbing a bank | bank robbery | 4(d) |
| shootout and showdown | shootout; showdown | 3, then 4(a) for each part |
| cop | cop | 4(a); the exact string is used, not police officer |
| gunfighter | gunfighter | 4(a); the exact string is used, not gunslinger |
| the American dream | american dream | 1 and 2, then 4(a) |
| good versus evil | ABSTRACT_THEME | 3 keeps it whole; it is not reduced to evil |
| Dracula | OUTSIDE_VOCABULARY | 5; a proper name absent from the vocabulary |
| asdf | UNINTERPRETABLE | 5 |

**Reference**

McRae, K., Cree, G. S., Seidenberg, M. S., & McNorgan, C. (2005). Semantic feature production norms for a large set of living and nonliving things. Behavior Research Methods, Instruments, & Computers, 37(4), 547-559. https://doi.org/10.3758/BF03192726
