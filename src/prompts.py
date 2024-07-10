INFORMATION_INTEGRITY_PROMPT = """Verify the integrity of two provided texts, Document 1 and Document 2. Document 1 has been converted into a markdown format from Document 2, although it is not perfect (some information may be missing). Document 2's contents are perfect, although it is in a human-unreadable format. Find what information is missing in Document 1 from Document 2. Only care about information integrity and nothing else. The output should only be a new version of Document 1 containing the missing information from Document 2.

Example 1:

---

Input:

Document 1:
# John Doe\n\n**(123) 456-7890**\n**johndoe@outlook.com**\n# Education\nBachelor of Science Computer Science - May 2029\n# Certifications\n- Certificate 1\n- Certificate 2

Document 2:
John Doe (123) 456-7890 johndoe@outlook.com Education Bachelor of Science Computer Science - May 2029 Certifications Certificate 1 Certificate 2 Certificate 3

Output:
# John Doe\n\n**(123) 456-7890**\n**johndoe@outlook.com**\n# Education\nBachelor of Science Computer Science - May 2029\n# Certifications\n- Certificate 1\n- Certificate 2\n- Certificate 3

---

Example 2:

---

Input: 

Document 1:
# Emily Williams\n\n**(123) 456-7890**\n**emilywilli@outlook.com**\n# Education\nChemical Engineering - May 2026\n# Work Experience\n- Google - Intern (2022)\n- Microsoft - Intern (2020)\n# Projects\n- **Project 1**\n- Item 1\n- Item 2\n- Item 3

Document 2:
Emily Williams                      (123) 456-7890 emilywilli@outlook.com    Education     University of Arizona Chemical Engineering - May 2026  Work Experience Google - Intern (2022) Microsoft - Intern (2020) Projects Project 1 Item 1 Item 2 Item 3

Output:

# Emily Williams\n\n**(123) 456-7890**\n**emilywilli@outlook.com**\n# Education\nUniversity of Arizona\nChemical Engineering - May 2026\n# Work Experience\n- Google - Intern (2022)\n- Microsoft - Intern (2020)\n# Projects\n- **Project 1**\n- Item 1\n- Item 2\n- Item 3"""

PROPOSITION_GENERATOR_PROMPT = """Decompose the resume/cv into clear and simple propositions, ensuring they are interpretable out of
context. The name of the creator of the resume/cv should always be present in each proposition.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input
whenever possible.
2. For any named entity that is accompanied by additional descriptive information, separate this
information into its own distinct proposition.
3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences
and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the
entities they refer to.
4. Present the results as a list of strings, formatted in JSON.

Example:

Input: Title: ¯Eostre. Section: Theories and interpretations, Connection to Easter Hares. Content:
The earliest evidence for the Easter Hare (Osterhase) was recorded in south-west Germany in
1678 by the professor of medicine Georg Franck von Franckenau, but it remained unknown in
other parts of Germany until the 18th century. Scholar Richard Sermon writes that "hares were
frequently seen in gardens in spring, and thus may have served as a convenient explanation for the
origin of the colored eggs hidden there for children. Alternatively, there is a European tradition
that hares laid eggs, since a hare's scratch or form and a lapwing's nest look very similar, and
both occur on grassland and are first seen in the spring. In the nineteenth century the influence
of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular throughout Europe.
German immigrants then exported the custom to Britain and America where it evolved into the
Easter Bunny."
Output: [ "The earliest evidence for the Easter Hare was recorded in south-west Germany in
1678 by Georg Franck von Franckenau.", "Georg Franck von Franckenau was a professor of
medicine.", "The evidence for the Easter Hare remained unknown in other parts of Germany until
the 18th century.", "Richard Sermon was a scholar.", "Richard Sermon writes a hypothesis about
the possible explanation for the connection between hares and the tradition during Easter", "Hares
were frequently seen in gardens in spring.", "Hares may have served as a convenient explanation
for the origin of the colored eggs hidden in gardens for children.", "There is a European tradition
that hares laid eggs.", "A hare's scratch or form and a lapwing's nest look very similar.", "Both
hares and lapwing's nests occur on grassland and are first seen in the spring.", "In the nineteenth
century the influence of Easter cards, toys, and books was to make the Easter Hare/Rabbit popular
throughout Europe.", "German immigrants exported the custom of the Easter Hare/Rabbit to
Britain and America.", "The custom of the Easter Hare/Rabbit evolved into the Easter Bunny in
Britain and America."]"""

STRAIGHT_FORWARD_PROMPT = 'Given a sentence, answer the question in the most straightforward manner. Do not add additional information, just only say what the user has asked. If there is no answer, then only say "None"'
