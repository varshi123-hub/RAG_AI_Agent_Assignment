# RAG AI Agent - Student Assignments

## Setup Instructions (Everyone)

1. Fork or clone: `https://github.com/NisargKadam/RAG_AI_Agent_Assignment`
2. Install dependencies: `pip install -r requirements.txt`
3. Create `.env` file with your OpenAI API key (see `.env.example`)
4. Place your chosen PDF(s) in the `data/` folder
5. Run ingestion: `python ingestion.py`
6. Run the agent: `python main.py`

## Submission Requirements (Everyone)

- Push your modified code to your own GitHub repository
- Include at least 2 screenshots showing your agent answering questions
- Include a short `REPORT.md` explaining:
  - What you changed and why
  - What difference it made in the answers
  - 3 example queries you tested with their outputs

---

## Assignment 1: Sethumeenakshi

**Topic: Medical/Health Domain RAG Agent**

**Tasks:**
1. Find 2-3 public health/medical PDFs (WHO reports, medical guidelines, etc.) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as a **medical information assistant** that:
   - Uses simple, patient-friendly language
   - Always adds a disclaimer: "This is for informational purposes only, consult a doctor"
   - Organizes answers with bullet points
3. In `ingestion.py`, change `CHUNK_SIZE` to `500` and `CHUNK_OVERLAP` to `100` — observe how smaller chunks affect medical Q&A precision
4. In `rag_agent.py`, change `TOP_K` to `6` to retrieve more context for medical queries
5. Test with 3 health-related questions and document the results

---

## Assignment 2: Abbiramy V Ra

**Topic: Legal Document Assistant**

**Tasks:**
1. Find 2-3 public legal PDFs (terms of service, privacy policies, open-source licenses, etc.) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as a **legal document analyzer** that:
   - Always quotes exact text from the source when referencing a clause
   - Numbers each point in the answer
   - Ends with "Key Takeaway:" summary
3. In `ingestion.py`, change `CHUNK_SIZE` to `1500` and `CHUNK_OVERLAP` to `300` — legal documents need larger chunks to preserve clause context
4. In `rag_agent.py`, switch retrieval to **METHOD 2** (similarity search with relevance scores) and set the score threshold to `0.4`
5. Test with 3 legal questions and document the results

---

## Assignment 3: Thiagaraj Karthikeyan

**Topic: Technical Documentation Bot**

**Tasks:**
1. Find 2-3 technical documentation PDFs (Python docs, API guides, framework tutorials, etc.) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as a **coding assistant** that:
   - Provides code examples in its answers when relevant
   - Uses markdown formatting for code blocks
   - Rates answer confidence as High/Medium/Low
3. In `ingestion.py`, switch to **STRATEGY 3** (TokenTextSplitter) with `chunk_size=200` tokens and `chunk_overlap=50` tokens
4. In `rag_agent.py`, change `TEMPERATURE` to `0.2` for slightly more creative code suggestions
5. Test with 3 programming questions and document the results

---

## Assignment 4: Tushar Bambal

**Topic: Educational Tutor Agent**

**Tasks:**
1. Find 2-3 educational PDFs (textbook chapters, course notes, study guides) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as a **friendly tutor** that:
   - Explains concepts as if teaching a beginner
   - Uses analogies and real-world examples
   - Ends each answer with a "Quick Quiz:" question to test understanding
3. In `ingestion.py`, change `CHUNK_SIZE` to `800` and `CHUNK_OVERLAP` to `150`
4. In `rag_agent.py`, change `TOP_K` to `5` and `TEMPERATURE` to `0.3`
5. Test with 3 educational questions and document the results

---

## Assignment 5: Sindhura Veerabomma

**Topic: Research Paper Summarizer**

**Tasks:**
1. Find 2-3 academic research paper PDFs (from arXiv, Google Scholar, etc.) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as a **research assistant** that:
   - Structures answers as: Findings, Methodology, Limitations
   - Uses academic tone
   - Highlights if the answer comes from abstract, introduction, or results section
3. In `ingestion.py`, change `CHUNK_SIZE` to `1200` and `CHUNK_OVERLAP` to `250` — research papers need larger context windows
4. In `rag_agent.py`, switch to **METHOD 3** (MMR search) to get diverse chunks from different paper sections
5. Test with 3 research-related questions and document the results

---

## Assignment 6: Kavi Suruthi

**Topic: Product Manual / FAQ Bot**

**Tasks:**
1. Find 2-3 product manual PDFs (electronics manuals, software user guides, etc.) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as a **customer support agent** that:
   - Gives step-by-step instructions numbered clearly
   - Uses simple non-technical language
   - Adds "Need more help?" with suggested follow-up questions at the end
3. In `ingestion.py`, change `CHUNK_SIZE` to `600` and `CHUNK_OVERLAP` to `100` — shorter chunks work well for FAQ-style content
4. In `rag_agent.py`, change `TOP_K` to `3` for more focused answers
5. Test with 3 how-to questions and document the results

---

## Assignment 7: Rasika Sudhir Rasal

**Topic: Financial Document Analyzer**

**Tasks:**
1. Find 2-3 public financial PDFs (annual reports, SEC filings, economic reports) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as a **financial analyst** that:
   - Highlights key numbers and percentages in answers
   - Compares data points when multiple are found
   - Adds "Risk Factors:" section when relevant
3. In `ingestion.py`, change `CHUNK_SIZE` to `1000` and `CHUNK_OVERLAP` to `200`, and try the **STRATEGY 2** (CharacterTextSplitter) to see how it handles tabular financial data
4. In `rag_agent.py`, switch to **METHOD 2** with relevance scores and print the scores
5. Test with 3 financial questions and document the results

---

## Assignment 8: Sharandeep Singh

**Topic: News/Current Affairs Summarizer**

**Tasks:**
1. Find 2-3 news report PDFs (UN reports, government publications, NGO reports) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as a **news analyst** that:
   - Answers in a structured format: Who, What, When, Where, Why
   - Identifies if information might be outdated
   - Provides balanced perspectives if the topic is debatable
3. In `ingestion.py`, change `CHUNK_SIZE` to `700` and `CHUNK_OVERLAP` to `150`
4. In `rag_agent.py`, change the embedding model to `"all-MiniLM-L12-v2"` (update both `ingestion.py` and `rag_agent.py`) and compare results
5. Test with 3 news/policy questions and document the results

---

## Assignment 9: Varshitha Vuyyuru

**Topic: Recipe / Cooking Assistant**

**Tasks:**
1. Find 2-3 recipe or cookbook PDFs (public domain cookbooks, food guides) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as a **cooking assistant** that:
   - Lists ingredients first, then step-by-step instructions
   - Suggests substitutions for common allergens
   - Adds estimated cooking time and difficulty level
3. In `ingestion.py`, change `CHUNK_SIZE` to `500` and `CHUNK_OVERLAP` to `50` — recipes are short and self-contained
4. In `rag_agent.py`, change `TEMPERATURE` to `0.4` to allow some creative suggestions
5. Test with 3 cooking questions and document the results

---

## Assignment 10: Renuka Agarwal

**Topic: HR Policy / Employee Handbook Bot**

**Tasks:**
1. Find 2-3 public HR policy PDFs (employee handbooks, workplace guidelines, company policies) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as an **HR assistant** that:
   - Answers formally and professionally
   - Always references the specific policy section
   - Adds "Action Items:" listing what the employee should do next
3. In `ingestion.py`, change `CHUNK_SIZE` to `800` and `CHUNK_OVERLAP` to `200`
4. In `rag_agent.py`, change `TOP_K` to `5` and use **METHOD 3** (MMR) to get diverse policy sections
5. Test with 3 HR-related questions and document the results

---

## Assignment 11: Girija Selvakumar

**Topic: Environmental / Climate Science Bot**

**Tasks:**
1. Find 2-3 environmental PDFs (IPCC reports, EPA documents, climate research) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as an **environmental science educator** that:
   - Explains scientific data in accessible language
   - Highlights cause-and-effect relationships
   - Ends with "What You Can Do:" actionable suggestions
3. In `ingestion.py`, change `CHUNK_SIZE` to `1100` and `CHUNK_OVERLAP` to `200`
4. In `rag_agent.py`, switch to `"all-mpnet-base-v2"` embedding model for better quality (update both files)
5. Test with 3 environment questions and document the results

---

## Assignment 12: Sushant Kamble

**Topic: Sports Analytics Assistant**

**Tasks:**
1. Find 2-3 sports-related PDFs (match reports, player statistics, sports science articles) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as a **sports analyst** that:
   - Uses statistics and numbers prominently
   - Compares players/teams when relevant data exists
   - Formats answers with tables when presenting comparative data
3. In `ingestion.py`, change `CHUNK_SIZE` to `600` and `CHUNK_OVERLAP` to `100`
4. In `rag_agent.py`, change `TOP_K` to `6` and `TEMPERATURE` to `0.1`
5. Test with 3 sports questions and document the results

---

## Assignment 13: Hareharan KM

**Topic: History / Social Studies Bot**

**Tasks:**
1. Find 2-3 history PDFs (historical documents, textbook chapters, museum guides) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as a **history professor** that:
   - Presents events in chronological order
   - Provides historical context and significance
   - Ends with "Historical Perspective:" connecting past events to present
3. In `ingestion.py`, change `CHUNK_SIZE` to `900` and `CHUNK_OVERLAP` to `180`
4. In `rag_agent.py`, use **METHOD 3** (MMR) to retrieve diverse historical perspectives
5. Test with 3 history questions and document the results

---

## Assignment 14: Uday Bhanu Bethi

**Topic: Travel Guide Assistant**

**Tasks:**
1. Find 2-3 travel guide PDFs (city guides, travel blogs saved as PDF, tourism documents) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as a **travel guide** that:
   - Answers enthusiastically with descriptive language
   - Organizes info as: Must-See, Best Time to Visit, Tips
   - Adds "Budget Estimate:" when possible
3. In `ingestion.py`, change `CHUNK_SIZE` to `700` and `CHUNK_OVERLAP` to `100`
4. In `rag_agent.py`, change `TEMPERATURE` to `0.5` for more engaging descriptions and `TOP_K` to `5`
5. Test with 3 travel questions and document the results

---

## Assignment 15: Jaya Raju Ganta

**Topic: Agriculture / Farming Knowledge Bot**

**Tasks:**
1. Find 2-3 agriculture PDFs (farming guides, crop reports, agriculture research) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as an **agricultural advisor** that:
   - Gives practical, actionable farming advice
   - Mentions seasonal considerations
   - Adds "Warnings:" for common mistakes to avoid
3. In `ingestion.py`, switch to **STRATEGY 2** (CharacterTextSplitter) with `CHUNK_SIZE=800`
4. In `rag_agent.py`, use **METHOD 2** (similarity with scores) and set threshold to `0.35`
5. Test with 3 agriculture questions and document the results

---

## Assignment 16: Sanjana Nandkishor Narkar

**Topic: Psychology / Mental Wellness Bot**

**Tasks:**
1. Find 2-3 psychology PDFs (mental health guides, psychology textbooks, wellness resources) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as a **wellness guide** that:
   - Uses empathetic and supportive language
   - Provides evidence-based information
   - Always adds: "If you're struggling, please reach out to a professional"
   - Lists "Practical Tips:" at the end
3. In `ingestion.py`, change `CHUNK_SIZE` to `900` and `CHUNK_OVERLAP` to `200`
4. In `rag_agent.py`, change `TEMPERATURE` to `0.2` and `TOP_K` to `4`
5. Test with 3 wellness-related questions and document the results

---

## Assignment 17: Sripad Mhaddalkar

**Topic: Cybersecurity Knowledge Base**

**Tasks:**
1. Find 2-3 cybersecurity PDFs (OWASP guides, NIST frameworks, security best practices) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as a **security advisor** that:
   - Categorizes threats by severity: Critical, High, Medium, Low
   - Provides both the problem and remediation steps
   - Adds "Quick Checklist:" with actionable security items
3. In `ingestion.py`, change `CHUNK_SIZE` to `1000` and `CHUNK_OVERLAP` to `250`
4. In `rag_agent.py`, change `TOP_K` to `6` and switch to **METHOD 3** (MMR) for diverse security perspectives
5. Test with 3 security questions and document the results

---

## Assignment 18: Jogula Satya Aditya

**Topic: Automotive / Vehicle Knowledge Bot**

**Tasks:**
1. Find 2-3 automotive PDFs (car manuals, vehicle maintenance guides, automotive engineering docs) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as a **mechanic advisor** that:
   - Diagnoses problems step by step
   - Rates urgency: "Fix Immediately", "Schedule Soon", "Can Wait"
   - Adds estimated difficulty for DIY: Easy, Moderate, Professional Only
3. In `ingestion.py`, change `CHUNK_SIZE` to `600` and `CHUNK_OVERLAP` to `120`
4. In `rag_agent.py`, change `LLM_MODEL` to `"gpt-4o-mini"` and compare with default
5. Test with 3 vehicle-related questions and document the results

---

## Assignment 19: Nutan Kiran Mahale

**Topic: Education Policy / Curriculum Analyzer**

**Tasks:**
1. Find 2-3 education policy PDFs (NEP 2020, curriculum frameworks, education research) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as an **education policy expert** that:
   - Explains policy implications for students and teachers
   - Compares old vs new policies when data is available
   - Adds "Implementation Status:" if mentioned in source
3. In `ingestion.py`, change `CHUNK_SIZE` to `1200` and `CHUNK_OVERLAP` to `300` — policy documents need larger context
4. In `rag_agent.py`, use **METHOD 2** (relevance scores) with threshold `0.3` and print scores
5. Test with 3 education policy questions and document the results

---

## Assignment 20: Jayesh Hariba Thorat

**Topic: Real Estate / Property Guide Bot**

**Tasks:**
1. Find 2-3 real estate PDFs (property guides, RERA documents, housing market reports) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as a **real estate advisor** that:
   - Highlights key financial figures (prices, EMI, area)
   - Lists pros and cons when comparing options
   - Adds "Legal Checklist:" for property transactions
3. In `ingestion.py`, switch to **STRATEGY 3** (TokenTextSplitter) with `chunk_size=250` tokens
4. In `rag_agent.py`, change `TOP_K` to `5` and `TEMPERATURE` to `0.1`
5. Test with 3 real estate questions and document the results

---

## Assignment 21: Rahul Dusane

**Topic: Fitness / Workout Planner Bot**

**Tasks:**
1. Find 2-3 fitness PDFs (workout guides, nutrition guides, exercise science papers) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as a **fitness coach** that:
   - Structures workouts with sets, reps, and rest periods
   - Adds safety warnings for exercises
   - Includes "Beginner Modification:" for each exercise
   - Ends with "Nutrition Tip:" related to the workout
3. In `ingestion.py`, change `CHUNK_SIZE` to `500` and `CHUNK_OVERLAP` to `80`
4. In `rag_agent.py`, change `TEMPERATURE` to `0.3` and switch to **METHOD 3** (MMR)
5. Test with 3 fitness questions and document the results

---

## Assignment 22: Khalid Khan

**Topic: Islamic Studies / Religious Text Bot**

**Tasks:**
1. Find 2-3 Islamic studies PDFs (Islamic history, Hadith compilations, scholarly articles) and place them in `data/`
2. In `rag_agent.py`, modify the `SYSTEM_PROMPT` to act as a **scholarly assistant** that:
   - Always provides references to specific sources/scholars
   - Presents multiple scholarly opinions when they exist
   - Uses respectful, academic language
   - Adds "Further Reading:" suggestions based on the topic
3. In `ingestion.py`, change `CHUNK_SIZE` to `1000` and `CHUNK_OVERLAP` to `250`
4. In `rag_agent.py`, change `TOP_K` to `5` and use **METHOD 2** (relevance scores) with threshold `0.3`
5. Test with 3 questions about Islamic history/studies and document the results

---

## Grading Rubric

| Criteria | Points |
|----------|--------|
| Code runs successfully (ingestion + agent) | 20 |
| System prompt modified appropriately for domain | 20 |
| Chunking/retrieval settings changed with reasoning | 20 |
| REPORT.md with 3 test queries + outputs + analysis | 20 |
| Screenshots of working agent | 10 |
| Code pushed to GitHub with clean commit history | 10 |
| **Total** | **100** |

---

## Tips for All Students

- **Read the comments** in `ingestion.py` and `rag_agent.py` — they explain what each setting does
- **Delete `chroma_db/` folder** before re-running `ingestion.py` if you change chunk settings or embedding model
- **Try different combinations** — change one setting at a time and observe the difference
- **Ask good questions** — the quality of your test queries matters. Ask specific questions that your PDFs can answer
- **Document your findings** — "I changed X because Y, and it resulted in Z" is the kind of analysis we want
