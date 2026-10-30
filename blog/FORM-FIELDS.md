# Blog Submission Form — field-by-field answers

Copy each block into the matching question on the
[Blog Submission Form](https://docs.google.com/forms/d/e/1FAIpQLSfSE7TDJDEjv0qjEsfer90l4XlsZ89RcfBVkupEO9iaHyE2mQ/viewform).

Deadline: **Sept 14, 11:59pm PST**.
Questions: Content SPOC — `amanda.nelson@hcl-software.com`

---

## 1. Role/Function *  — YOU FILL

Your own job title / function.

---

## 2. GEO *  — YOU FILL

Dropdown. (This is the field showing the red "required question" error in your
screenshot.)

---

## 3. HCLSoftware Solution *  — ⚠️ NEEDS A DECISION

**Do not just pick one.** This article is vendor-neutral by design — the brief
forbids competitor mentions and promotional content, so it names no product at
all. Nothing in it maps to a specific HCLSoftware solution.

Before submitting, either:

- pick the solution your team actually works on **and** confirm with the Content
  SPOC that a vendor-neutral engineering piece is acceptable under it, or
- ask the SPOC which value fits a general software-engineering submission.

Selecting a product the article has nothing to do with would misrepresent it.

---

## 4. Blog Title *

**Recommended:**

```
Parse, Map, Verify: A Blueprint for Automating SQL Stored Procedure Migration to Python
```

Alternates:

```
From PL/SQL to FastAPI: Why Your Migration Pipeline Needs a Syntax Gate
```

```
Why Move Business Logic Out of Stored Procedures - And How to Automate Getting It Out
```

---

## 5. Blog Abstract (250 word max) *

*238 words — within limit.*

```
Legacy stored procedures are the quietest form of technical debt an enterprise
carries: they run reliably for years, which is exactly why nobody touches them.
The cost surfaces elsewhere - logic locked inside PL/SQL or T-SQL cannot be unit
tested without a live database, cannot scale independently of the database tier,
and cannot be reasoned about by the application team that owns the feature.

Most writing on this subject argues that the debt is bad. This article instead
sets out a working pipeline for extracting that logic mechanically, and for
verifying the result.

It walks through five stages: dialect fingerprinting from weighted syntax
markers; semantic extraction of cursor loops, transaction boundaries and
exception intent; schema-driven mapping of SQL types onto Pydantic models;
generation of a FastAPI/SQLAlchemy service function; and an Abstract Syntax Tree
verification gate.

That final stage is the argument of the piece. Treating Python's own ast.parse
as a hard gate - rejecting and regenerating unparseable output automatically,
rather than reporting it - is what moves generated code from a promising first
draft to a reviewable artifact.

A concrete Oracle procedure involving a cursor loop, a SELECT ... FOR UPDATE row
lock, a stock-sufficiency check and a single trailing COMMIT is shown beside its
generated Python equivalent, with the translation decisions annotated inline.
The article closes with enterprise trade-offs, a five-point readiness checklist
for engineering leads, and three FAQs on performance, transaction handling and
test coverage.
```

---

## 6. Audience Persona *

```
Senior backend engineers, application architects and engineering leads
responsible for enterprise systems where significant business logic still lives
in Oracle PL/SQL, PostgreSQL PL/pgSQL or Microsoft T-SQL stored procedures.

Readers are assumed to be fluent in both SQL and Python and to have hands-on
migration experience - the piece assumes familiarity with cursors, transaction
isolation, FastAPI and ORM patterns, and does not explain them.

It is aimed at practitioners scoping or executing a modernization programme who
need a concrete technical blueprint and a realistic account of the trade-offs,
rather than a business case for modernizing.
```

---

## 7. Upload Blog Content (Word Doc or Google Docs only) *

Upload:

```
blog/SP2PY-Blog-Voices-of-Innovation.docx
```

Full path:
`C:\Users\subha\Downloads\sp2py_app\sp2py_app\blog\SP2PY-Blog-Voices-of-Innovation.docx`

1,289 prose words (limit is 800-1,500). Markdown source is
`blog/voices-of-innovation-draft.md` if you prefer to paste into Google Docs.

**Note:** the .docx is a snapshot. Edit the .md and it will not update - ask for
a regenerate.

---

## 8. Author Bio (max 250 words) *  — YOU FILL THE BRACKETS

Written as a skeleton because inventing your credentials would be worse than
leaving blanks.

```
Subhajit Ganguli is a [role] at [organisation], where he works on [domain -
e.g. data platform modernization and backend systems engineering]. He has [N]
years of experience across [areas], including hands-on work migrating
database-bound business logic to service architectures.

His current focus is [e.g. verification techniques for generated code and
automated refactoring pipelines]. [Optional: education, certifications, notable
projects, speaking, GitHub/LinkedIn.]
```

---

## 9. Author Bio Picture  — YOU UPLOAD

Optional (no red asterisk). Image, max 10 MB, up to 5 files.

---

## 10. Which of the following best describes the primary goal of your blog post? *

Pick the option closest to **thought leadership / educating a technical
audience**.

The article is explicitly non-promotional and names no products, so do not
select a demand-generation or product-awareness option if one is offered.

---

## 11. Certification and Acknowledgement *  — YOUR CALL

Two checkboxes:

1. *"the content herein represents my own original thoughts and abides by all
   HCLSoftware rules and regulations"*
2. *"HCLSoftware retains all editorial and publication control"*

Worth knowing before you tick box 1: this draft was written with substantial AI
assistance, based on a project built the same way. Whether that needs disclosing
is an HCLSoftware policy question - ask the SPOC rather than assuming either way.

---

## Still outstanding

- **Architecture diagram** - the draft has a `[DIAGRAM PLACEHOLDER]` with a full
  textual description of the five-stage flow. Someone needs to draw it, or ask
  and I will generate an SVG/PNG.
- **HCLSoftware Solution** field (see section 3).
