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
context, and while keeping the creator's name in each proposition.
1. Split compound sentence into simple sentences. Maintain the original phrasing from the input
whenever possible. The name of the creator of the resume/cv should always be present.
2. For any named entity that is accompanied by additional descriptive information, separate this
information into its own distinct proposition while keeping it within the context of the creator of the resume/cv.
3. Decontextualize the proposition by adding necessary modifier to nouns or entire sentences
and replacing pronouns (e.g., "it", "he", "she", "they", "this", "that") with the full name of the
entities they refer to. 
4. Present the results as a list of strings, formatted in JSON.
5. The first element of the list of strings should be the name of the author of the resume/cv
6. If a location is given about a named entity, assume that the creator went there.
    - Under work experiences, it is common to list the location. Say that the creator worked at the place in that location. If the work is remote, say they worked remotely.

Example:

Input: # Andy Lu\n\n**Contact Information:**\n- Phone: 520-333-4444\n- Email: andylu@arizona.edu\n- LinkedIn: [linkedin.com/in/andyelu](https://linkedin.com/in/andyelu)\n- GitHub: [github.com/andyelu](https://github.com/andyelu)\n\n## Education\n\n**University of Arizona** – GPA: 4.0  \n_Tucson, AZ_  \n**Bachelor of Science in Computer Science**  \n_Aug 2022 - May 2026_\n\n## Experience\n\n**Software Engineer Intern**  \n_Sion Power, Tucson, AZ_  \n_May 2023 – August 2023_\n- Developed a data processing app with Python and Pandas to parse 10,000+ ASCII datasets for specific measurements and perform electrochemistry calculations; reduced a manual data retrieval process from 1 hour to seconds.\n- Implemented vast unit testing to verify my program’s calculations, guaranteeing the app’s reliability for end-users.\n- Presented my app to a department comprising of twelve scientists, adding features and updates based on their feedback to meet user needs effectively.\n- Adopted object-oriented design principles to enhance code extensibility, facilitating future development efforts.\n- Designed and contributed to a migration strategy to transition a Microsoft VBA-based app to Python, laying the groundwork for enhanced code scalability and future maintainability.\n\n**Undergraduate Research Assistant**  \n_University of Arizona - Supervised by Dr. Eung-Joo Lee, Tucson, AZ_  \n_December 2023 - Present_\n- Developed a full-stack app to address overcrowding at the university recreational centers, utilizing a YOLOv5 object detection model with PyTorch to analyze live security camera footage to record human congestion levels.\n- Engineered a REST API using Python and the Django REST framework to effectively log the recorded data to a MySQL database.\n- Developed a regression model with scikit-learn to predict peak and off-peak times, trained on collected congestion data.\n\n**Computer Science Teaching Assistant**  \n_University of Arizona, Tucson, AZ_  \n_January 2024 - Present_\n- Collaborated with a team of TAs and the professor to support 170+ students in a Software Development class.\n- Utilized strong communication skills to break down complex coding topics into simpler terms, significantly enhancing student engagement and comprehension.\n- Conducted office hours to resolve student questions, ensuring effective support regarding the assignments and content.\n\n## Projects\n\n**RateMyRestroom.net**  \n_Technologies: Java, Spring Boot, PostgreSQL, AWS S3, AWS EC2_\n- Created a full-stack web app allowing students to locate and leave ratings on University of Arizona restrooms, bringing attention to restroom quality standards and wheelchair/all-gender accessibility.\n- Developed a Java REST API using Spring Boot to manage review data. The API handles GET and POST requests, allowing interaction with a PostgreSQL database.\n- Created comprehensive documentation for the REST API - including clear descriptions of endpoints, request/response formats, and use cases - contributing to the overall maintainability and scalability of the application.\n\n**Raspberry PI GPT Assistant**  \n_Technologies: OpenAI GPT-3.5 Turbo API, Google Cloud Speech API_\n- Engineered a Raspberry Pi-based personal assistant using OpenAI’s GPT-3.5 Turbo API for intuitive interaction.\n- Implemented wake-word activation and vocal recognition to allow for seamless user engagement by leveraging the Google Cloud Speech API.\n- Curated vocalized responses by leveraging the ElevenLabs Text-To-Speech API.\n\n**AI-Integrated Closet Security System**  \n_Technologies: Python, Django, Arduino, ESP32, Raspberry Pi, AWS S3_\n- Engineered a light-activated security system using Raspberry Pi, Arduino, and an ESP32 cam, leveraging Google Vision AI API for accurate human detection on the images taken by the camera.\n- Streamlined system communication and data management through a Django REST API, leveraging AWS S3 for image storage and SQLite for timestamped event logging with S3 object URLs.\n- Developed a Python-based email notification system for immediate alerts upon detection events.\n\n## Technical Skills\n\n**Languages:** Python, Java, C#, JavaScript  \n**Frameworks/Tools:** Django, Springboot, React, Vanilla (HTML, CSS), Unity, Git, SQL, Postgres, Postman, Pandas, Linux, JUnit, Docker, Amazon Web Services

Output: ["Andy Lu's phone number is 520-333-4444.", "Andy Lu's email is andylu@arizona.edu.", "Andy Lu's LinkedIn profile is linkedin.com/in/andyelu.", "Andy Lu's GitHub profile is github.com/andyelu.", 'Andy Lu attended the University of Arizona.', 'Andy Lu achieved a GPA of 4.0 at the University of Arizona.', 'Andy Lu studied in Tucson, AZ.', 'Andy Lu pursued a Bachelor of Science in Computer Science.', 'Andy Lu attended the University of Arizona from August 2022 to May 2026.', 'Andy Lu worked as a Software Engineer Intern.', 'Andy Lu worked at Sion Power in Tucson, AZ.', 'Andy Lu worked at Sion Power from May 2023 to August 2023.', 'Andy Lu developed a data processing app with Python and Pandas.', "Andy Lu's app parsed 10,000+ ASCII datasets for specific measurements.", "Andy Lu's app performed electrochemistry calculations.", "Andy Lu's app reduced a manual data retrieval process from 1 hour to seconds.", 'Andy Lu implemented vast unit testing to verify the program's calculations.', 'Andy Lu guaranteed the app's reliability for end-users.', 'Andy Lu presented the app to a department comprising twelve scientists.', 'Andy Lu added features and updates based on feedback from twelve scientists.', 'Andy Lu adopted object-oriented design principles to enhance code extensibility.', 'Andy Lu facilitated future development efforts through object-oriented design.', 'Andy Lu designed and contributed to a migration strategy.', 'Andy Lu transitioned a Microsoft VBA-based app to Python.', 'Andy Lu laid the groundwork for enhanced code scalability and future maintainability.', 'Andy Lu worked as an Undergraduate Research Assistant.', 'Andy Lu was supervised by Dr. Eung-Joo Lee.', 'Andy Lu worked at the University of Arizona in Tucson, AZ.', 'Andy Lu started working as an Undergraduate Research Assistant in December 2023.', 'Andy Lu developed a full-stack app to address overcrowding at university recreational centers.', 'Andy Lu utilized a YOLOv5 object detection model with PyTorch.', 'Andy Lu analyzed live security camera footage to record human congestion levels.', 'Andy Lu engineered a REST API using Python and the Django REST framework.', 'Andy Lu logged recorded data to a MySQL database.', 'Andy Lu developed a regression model with scikit-learn.', "Andy Lu's regression model predicted peak and off-peak times.", 'Andy Lu trained the regression model on collected congestion data.', 'Andy Lu worked as a Computer Science Teaching Assistant.', 'Andy Lu worked at the University of Arizona in Tucson, AZ.', 'Andy Lu started working as a Computer Science Teaching Assistant in January 2024.', 'Andy Lu collaborated with a team of TAs and the professor.', 'Andy Lu supported 170+ students in a Software Development class.', 'Andy Lu utilized strong communication skills to break down complex coding topics.', 'Andy Lu significantly enhanced student engagement and comprehension.', 'Andy Lu conducted office hours to resolve student questions.', 'Andy Lu ensured effective support regarding assignments and content.', 'Andy Lu created RateMyRestroom.net.', 'Andy Lu used Java, Spring Boot, PostgreSQL, AWS S3, and AWS EC2 for RateMyRestroom.net.', 'Andy Lu created a full-stack web app for students to locate and rate University of Arizona restrooms.', "Andy Lu's app brought attention to restroom quality standards and wheelchair/all-gender accessibility.", 'Andy Lu developed a Java REST API using Spring Boot to manage review data.', "Andy Lu's API handles GET and POST requests.", "Andy Lu's API allows interaction with a PostgreSQL database.", 'Andy Lu created comprehensive documentation for the REST API.', "Andy Lu's documentation included clear descriptions of endpoints, request/response formats, and use cases.", 'Andy Lu contributed to the overall maintainability and scalability of the application.', 'Andy Lu created a Raspberry PI GPT Assistant.', 'Andy Lu used OpenAI GPT-3.5 Turbo API and Google Cloud Speech API for the Raspberry PI GPT Assistant.', 'Andy Lu engineered a Raspberry Pi-based personal assistant using OpenAI's GPT-3.5 Turbo API.', 'Andy Lu implemented wake-word activation and vocal recognition.', 'Andy Lu leveraged the Google Cloud Speech API for seamless user engagement.', 'Andy Lu curated vocalized responses using the ElevenLabs Text-To-Speech API.', 'Andy Lu created an AI-Integrated Closet Security System.', 'Andy Lu used Python, Django, Arduino, ESP32, Raspberry Pi, and AWS S3 for the AI-Integrated Closet Security System.', 'Andy Lu engineered a light-activated security system using Raspberry Pi, Arduino, and an ESP32 cam.', 'Andy Lu leveraged Google Vision AI API for accurate human detection on images taken by the camera.', 'Andy Lu streamlined system communication and data management through a Django REST API.', 'Andy Lu leveraged AWS S3 for image storage.', 'Andy Lu used SQLite for timestamped event logging with S3 object URLs.', 'Andy Lu developed a Python-based email notification system for immediate alerts upon detection events.', "Andy Lu's technical skills include Python, Java, C#, and JavaScript.", "Andy Lu's technical skills include Django, Springboot, React, Vanilla (HTML, CSS), Unity, Git, SQL, Postgres, Postman, Pandas, Linux, JUnit, Docker, and Amazon Web Services."]
"""

STRAIGHT_FORWARD_PROMPT = 'Given a sentence, answer the question in the most straightforward manner. Do not add additional information, just only say what the user has asked. If there is no answer, then only say "None"'

CHUNK_DETERMINER_PROMPT = """
Given a proposition from a resume/cv, determine whether or not it belongs to any of the existing chunks.

A proposition should belong to a chunk if the meaning, direction, or intention is similar to that of the chunk.
The goal is to group similar propositions and chunks.

If you think a proposition should be joined with a chunk, return the chunk id.
If you do not think an item should be joined with an existing chunk, return "None".


Example:
Input:
    - Proposition: "Kashyap has worked on a project called BattleshipPy"
    - Current Chunks:
        - Chunk ID: TJBseUUn
        - Chunk Name: Work Experience
        - Chunk Summary: This chunk contains information about Kashyap's work experience

        - Chunk ID: sWzhE9wW
        - Chunk Name: Projects
        - Chunk Summary: This chunk contains information about Kashyap's personal projects
Output: sWzhE9wW

Input:
    - Proposition: "Kashyap has interned at Oracle"
    - Current Chunks:
        - Chunk ID: TJBseUUn
        - Chunk Name: Work Experience
        - Chunk Summary: This chunk contains information about Kashyap's work experience

        - Chunk ID: sWzhE9wW
        - Chunk Name: Projects
        - Chunk Summary: This chunk contains information about Kashyap's personal projects
Output: TJBseUUn

Only respond with the new chunk summary, nothing else.
"""

CHUNK_SUMMARY_PROMPT = """You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
You should generate a very brief 1-sentence summary which will inform viewers what a chunk group is about.

A good summary will say what the chunk is about, and give any clarifying instructions on what to add to the chunk.

You will be given a list of propositions which represent a chunk. This chunk needs a summary.

Your summaries should anticipate generalization within the context of a resume/cv. If you get a proposition about a college degree, generalize it to educational background.
Or about employment, generalize it to "work experience". If it is a project, generalize it to "project"

Example:
Input: Propositions: {name} has worked at Microsoft as a Software Engineer Intern from June 2024 to August 2024. {name} has worked at Microsoft in Virginia.
Output: This chunk contains information about {name}'s work experience.

Input: Proposition: {name} has worked on a project called Joyful Jobs to improve the hiring experience. {name}'s Joyful Jobs has around 20,000 active users. {name}'s Joyful Jobs is used by over 100 companies.
Output: This chunk contains information about {name}'s projects.

Only respond with the new chunk summary, nothing else."""

CHUNK_TITLE_PROMPT = """You are the steward of a group of chunks which represent groups of sentences that talk about a similar topic
You should generate a very brief few word chunk title which will inform viewers what a chunk group is about.

A good chunk title is brief but encompasses what the chunk is about

You will be given a summary of a chunk which needs a title

Your titles should anticipate generalization within the context of a resume/cv. If you get a proposition about a college degree, generalize it to educational background.
Or about employment, generalize it to "work experience".

Example:
Input: Summary: This chunk is about John Smith's certifications
Output: Certifications

Only respond with the new chunk title, nothing else."""
