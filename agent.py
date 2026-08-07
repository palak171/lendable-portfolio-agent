"""
NL -> SQL agent over the Lendable loan portfolio SQLite database.

The agent is given a natural-language question and a single tool,
`execute_sql`, which runs a read-only SELECT against the database. It can
call the tool multiple times (e.g. run one query to check a hypothesis,
then another to drill in) before producing a final answer. The final
answer is always backed by the last SQL query it ran -- the agent never
computes the reported numbers in Python, it only asks the database.
"""

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field

import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DB_PATH = os.path.join(os.path.dirname(__file__), "lendable_portfolio.db")
MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_STEPS = 6
PREVIEW_ROWS = 25

SCHEMA_CONTEXT = """
You are a SQL analyst agent for Lendable, a lender that funds loan originators
across Africa. You answer questions from a (non-technical) risk officer about
a loan portfolio, stored in a SQLite database with two tables. All amounts
are in KES.

TABLE loans (one row per loan)
  loan_id TEXT               unique loan id, e.g. L00001
  customer_id TEXT           borrower id
  originator_id TEXT         O001-O004
  originator_name TEXT       e.g. Faulu Kenya
  product_type TEXT          Working Capital / Asset Finance / Agricultural / Consumer
  begin_date TEXT            loan start date YYYY-MM-DD
  original_end_date TEXT     scheduled end date at origination
  rescheduled_end_date TEXT  new end date if rescheduled, else NULL
  reschedule_flag INTEGER    1 = rescheduled, 0 = not
  principal REAL             principal amount KES
  total_principal_interest REAL   total repayable (principal + flat interest)
  currency_type TEXT         always KES
  monthly_rate REAL          monthly flat interest rate, e.g. 0.03 = 3%

TABLE history (one row per loan per month on book -- cumulative monthly snapshot)
  loan_id TEXT             FK to loans.loan_id
  record_date TEXT         snapshot date YYYY-MM-DD, first of month
  month_on_book INTEGER    months since begin_date, 1 = first month
  amount_due REAL          CUMULATIVE amount due as of this month
  amount_paid REAL         CUMULATIVE amount paid as of this month
  days_late INTEGER        days late at this snapshot, 0 = on time
  status TEXT              Current / PAR / PAR30 / PAR60 / Write-off / Paid-off

Status definitions: Current = 0 days late. PAR = 1-29 days late.
PAR30 = 30-59 days late. PAR60 = 60-89 days late. Write-off = 90+ days late.
Paid-off = loan completed and fully repaid.

Critical facts about this data -- get these wrong and your answer is wrong:
1. amount_due and amount_paid are CUMULATIVE. To get the amount due/paid in a
   single period you must subtract the previous month's value for that loan
   (e.g. via a self-join or LAG window function on month_on_book).
2. To get each loan's LATEST snapshot, filter to the max(month_on_book) per
   loan_id (e.g. via a correlated subquery, window function, or GROUP BY).
3. Loans with status = 'Paid-off' keep appearing in history for months after
   completion. Analyses of the ACTIVE portfolio (e.g. PAR) should generally
   EXCLUDE loans whose latest status is 'Paid-off' or 'Write-off', unless the
   question is specifically about write-offs.
4. days_late in the history table reflects lateness against whatever schedule
   is currently in force -- for rescheduled loans this is measured against
   the rescheduled_end_date/repayment plan, not the original one. If asked
   whether PAR captures rescheduled-loan lateness, or whether lateness is
   measured from the original vs. rescheduled date, you should run queries
   that compare rescheduled_flag against days_late/status patterns to give an
   evidence-based answer, not just assert the schema comment.
5. PAR (portfolio at risk) can be measured multiple ways: by loan count, by
   customer count, or by amount outstanding (amount_due - amount_paid at the
   latest snapshot). Prefer amount-outstanding-based PAR when not specified,
   but you can compute more than one basis if it helps answer the question.
6. reschedule_flag = 1 identifies rescheduled loans; use it to split the
   portfolio into rescheduled vs. non-rescheduled cohorts for comparisons.

You have ONE tool, execute_sql, which runs a single read-only SELECT
statement against this database and returns the row count, columns, and up
to the first 25 rows as a preview. You may call it multiple times: e.g. run
an exploratory query first, look at the shape of the result, then run a
follow-up query that answers the question precisely. Prefer to do
aggregation, joins, filtering, and math IN SQL, not by describing it in
prose -- the whole point is that retrieval and computation go through the
database.

When you have enough information, STOP calling tools and reply with ONLY a
single JSON object (no markdown fences, no extra text) with these fields:

{
  "explanation": "2-5 sentences in plain English answering the risk officer's
                   question, citing the actual numbers you found.",
  "display": "table" or "chart",
  "chart_type": "line" | "bar" | "grouped_bar" | "pie" | null,
  "x": "<column name from your FINAL query's result to use as x-axis>" or null,
  "y": "<column name>" or ["<column1>", "<column2>", ...] or null,
  "series": "<optional column name to color/group by>" or null,
  "title": "short chart/table title"
}

Rules for the final answer:
- The table/chart shown to the user will be exactly the result of the LAST
  execute_sql call you made. Make sure your last query returns a clean,
  final result set (sensible column names, sorted, not just an exploratory
  scratch query).
- Use "chart" when showing a trend over time (e.g. PAR by month) or a
  comparison across a few categories (e.g. rescheduled vs not). Use "table"
  for detailed breakdowns, single numbers, or many rows/columns.
- "x"/"y"/"series" must be exact column names appearing in your final
  query's result set.
- Never invent numbers that didn't come from a query result.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": (
                "Execute a single read-only SQL SELECT statement against the "
                "loans/history SQLite database and return the results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A SQL SELECT statement (SQLite dialect).",
                    }
                },
                "required": ["query"],
            },
        },
    }
]

_READ_ONLY_RE = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_FORBIDDEN_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|ATTACH|DETACH|PRAGMA|CREATE|REPLACE|VACUUM|TRIGGER)\b",
    re.IGNORECASE,
)


class UnsafeQueryError(Exception):
    pass


def run_sql(query: str) -> pd.DataFrame:
    if not _READ_ONLY_RE.match(query) or _FORBIDDEN_RE.search(query):
        raise UnsafeQueryError("Only read-only SELECT/WITH statements are allowed.")
    conn = sqlite3.connect(DB_PATH)
    try:
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


@dataclass
class AgentResult:
    explanation: str
    display: str
    chart_type: str | None
    x: str | None
    y: object
    series: str | None
    title: str
    dataframe: pd.DataFrame
    sql_trail: list = field(default_factory=list)
    error: str | None = None


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def ask(question: str, api_key: str | None = None) -> AgentResult:
    client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    messages = [
        {"role": "system", "content": SCHEMA_CONTEXT},
        {"role": "user", "content": question},
    ]

    sql_trail = []
    last_df = pd.DataFrame()

    for _ in range(MAX_STEPS):
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                    query = args["query"]
                    df = run_sql(query)
                    last_df = df
                    sql_trail.append({"query": query, "row_count": len(df)})
                    tool_content = json.dumps(
                        {
                            "row_count": len(df),
                            "columns": list(df.columns),
                            "preview": json.loads(
                                df.head(PREVIEW_ROWS).to_json(orient="records")
                            ),
                        }
                    )
                except Exception as e:  # noqa: BLE001
                    tool_content = json.dumps({"error": str(e)})
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": tool_content}
                )
            continue

        # Final answer
        raw = _strip_json_fences(msg.content or "{}")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return AgentResult(
                explanation=msg.content or "The agent did not return a parseable answer.",
                display="table",
                chart_type=None,
                x=None,
                y=None,
                series=None,
                title=question,
                dataframe=last_df,
                sql_trail=sql_trail,
                error="Could not parse agent's final JSON response.",
            )

        return AgentResult(
            explanation=parsed.get("explanation", ""),
            display=parsed.get("display", "table"),
            chart_type=parsed.get("chart_type"),
            x=parsed.get("x"),
            y=parsed.get("y"),
            series=parsed.get("series"),
            title=parsed.get("title", question),
            dataframe=last_df,
            sql_trail=sql_trail,
        )

    return AgentResult(
        explanation="The agent used too many steps without reaching a final answer.",
        display="table",
        chart_type=None,
        x=None,
        y=None,
        series=None,
        title=question,
        dataframe=last_df,
        sql_trail=sql_trail,
        error="max_steps_exceeded",
    )
