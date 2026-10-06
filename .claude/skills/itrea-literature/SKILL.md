---
name: itrea-literature
description: >
  Add a citation or a prior-art entry to I-TREA II: verify the bibliographic record, find a legally readable full text,
  read it, record what it does and how it relates, and write the bib entry — or file the work as paywalled. Use
  whenever a paper, preprint, manual, talk or code base is about to be cited in the paper, the vault or AGENTS.md, and
  when a claim needs a source.
---

# Adding a citation (I-TREA II)

Owner's rules (Berk Orbay): cite only work whose full text is **legally readable**; never claim "first" or "novel";
plain, precise wording about what a work does.

## 1. Verify the record
- Resolve the DOI / arXiv id / URL; check title, authors, year, venue, volume and pages against Crossref or DataCite
  (or the publisher's page). Leave out any field you could not confirm — never guess one.
- Prefer the version of record; note when the numbers you quote belong to a preprint revision.

## 2. Find a legal full text
Acceptable: publisher open access (state the licence, e.g. CC BY 4.0), arXiv, institutional or national repositories,
author or lab pages, technical reports, official documentation, source code at a pinned commit, or an Internet Archive
capture of one of these. Search with Unpaywall, OpenAlex, Semantic Scholar, CORE and the authors' pages.
**Never send Berk's e-mail address (or anyone's) to an outside service** without his explicit ask — Unpaywall's API
asks for an `email=` parameter: use a placeholder (incident of 2026-10-05/06 in `vault/60-incidents/`).
**Never** shadow libraries, ResearchGate or Academia.edu uploads, or copies of unclear provenance (mark a doubtful
mirror as such, as sweep C did).

## 3. Read it, then record it
- Add or update a row in the relevant sweep under `vault/40-literature/prior-art/` (A constraint handling with
  infeasible solutions, B metaheuristics cooperating with MIP solvers, C GPU primal heuristics and GPU EC, …; a new
  topic gets a new lettered sweep): key, authors, year, venue/DOI, link to the legal copy, access, **what it does**,
  **relation to I-TREA II** (lineage / technique we use / competitor / background), verification level (*full text
  read*, *abstract / landing read*, *metadata only*).
- Claims in our text must come from what was read; anything taken from a secondary description says so.
- A work that does something close to ours is recorded as such; our differences are stated as differences, not as
  novelty. A negative search is recorded with its queries and dates.

## 4. Paywalled works
If no legal full text exists: the work goes **only** into `vault/40-literature/Paywalled research.md` with the record and what is known from its abstract, and may be cited in the paper only in the appendix
subsection `app:paywalled`, under its disclaimer. In the main text cite an open proxy instead where one exists (e.g.
`berthold2006primal` for Rothberg's combination operator, `boschetti2022matheuristics` for matheuristics).

## 5. Bib entry
- The paper's only bibliography is `paper/refs.bib` — legally readable works only. Part I's reference audit lives in
  `vault/40-literature/I-TREA (2015) references.bib`, deliberately outside `paper/` so its closed works cannot be cited
  by accident; to cite one of its open works, copy the entry into `refs.bib`. Key = first author's surname + year +
  one word (`luteberget2023feasibility`), the same key as in the sweep table and [[Prior art]].
- Fields as verified; `doi` when one exists; `url` = the legal full text; `note` = the kind of copy
  ("Open version: \url{…}", "author manuscript", "preprint", "CC BY 4.0"). Closed works carry
  `note = {Full text: not openly available}` and are cited only under §4.
- Software and documentation: the version or commit hash and the access date (`/current/` URLs drift).
- Part I is cited through Berk's open M.S. thesis (`orbay2011itrea`, METU 2011, hdl:11511/20509). The 2015 journal
  manuscript (unpublished) is not cited with a key until Berk agrees; describe it in prose.
