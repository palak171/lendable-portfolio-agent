# Lendable Portfolio Agent

A natural-language agent over the `lendable_portfolio.db` loan portfolio. It
converts a question into one or more SQL queries, runs them against the
SQLite database, and returns a table or chart -- all retrieval and
computation happens in SQL, not in Python.

## How it works

- `agent.py` holds the schema/business-rules context and the agent loop.
  The agent is given one tool, `execute_sql`, and can call it multiple times
  (e.g. explore, then refine) before returning a final JSON answer that
  names which columns to plot.
- `app.py` is a Streamlit chat UI: type a question, see the agent's
  explanation, a table or chart, and (in an expander) the exact SQL it ran --
  useful for the "what I built" walkthrough and for a risk officer to trust
  the answer.
- Guardrails: only `SELECT`/`WITH` statements are allowed; anything else
  (`INSERT`, `DROP`, `PRAGMA`, ...) is rejected before it reaches the DB.

## Setup

1. Get a free Groq API key at https://console.groq.com (sign up, then
   "API Keys" -> "Create API Key"). No card required.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Either paste the key into the Streamlit sidebar at runtime, or create a
   `.env` file (copy `.env.example`) with:
   ```
   GROQ_API_KEY=gsk_...
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## The 5 case-study questions

The sidebar has one-click buttons for all five:
1. Is PAR going up?
2. What is driving PAR up?
3. Does PAR include the lateness from rescheduled loans?
4. How much of PAR is driven by rescheduled loans?
5. Do rescheduled loans perform differently than non-rescheduled loans?

## Notes for the walkthrough

- **What I tried**: a single-shot NL->SQL prompt first; switched to a
  tool-calling loop so the agent can run an exploratory query, look at the
  shape of the result, and then issue a precise follow-up query rather than
  guessing the right SQL in one shot.
- **What broke**: cumulative `amount_due`/`amount_paid` columns and the fact
  that paid-off loans keep rows in `history` after completion were the two
  easiest ways to get PAR wrong -- both are called out explicitly in the
  system prompt so the model doesn't rediscover them per question.
- **What I'd improve**: cache/reuse query results across turns for follow-up
  questions in the same conversation; add a schema-aware SQL validator (e.g.
  `EXPLAIN QUERY PLAN`) to catch malformed SQL before showing an error;
  support drill-down (click a chart point to re-query at that granularity).
