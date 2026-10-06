# Upstream reports — drafts only

This folder holds **drafts** of issues or pull requests for projects this study depends on (SCIP / PySCIPOpt, HiGHS,
NVIDIA cuOpt, CuPy, Redis, …). **Nothing is filed without Berk's explicit yes for that one post.** Permission is
per post, not standing: an earlier "yes" does not cover the next report, and a "you may file issues" in general does
not cover any particular one. The same applies to anything submitted elsewhere (MIPLIB solutions, arXiv, Zenodo).

## Procedure
1. **Reproduce minimally** on a toy or a public instance, with versions, commands and output. Never include private or
   customer data. Confirm the signature in our own logs before citing someone else's issue.
2. **Search for prior art**: `gh issue list --repo <owner/repo> --state all --search "<keywords>"`, plus the project's
   discussions and changelog. A clean reproduction is not evidence of novelty; optopt filed two duplicates once.
3. **Write the draft here** as `DRAFT <project> <short title>.md`: title, body, minimal reproduction, environment, what
   we expected, what happened, links to duplicates found or "none found (searched: …)".
4. **State the authorship plainly in the body:** drafted by AgentRA (Agent Research Assistant; mainly Claude Opus 5.5,
   via Claude Code) on behalf of Berk Orbay, who reviewed it.
5. **Show Berk the draft and wait for an explicit yes** for this post. Record the yes (date, his words) in the draft.
6. After filing, rename the file to `<project> <number> <short title>.md`, add the link, and add a line to the
   journal. Corrections after filing (a duplicate found later, a wrong claim) are posted as a comment and recorded here.

## Drafts
| file | project | status |
|---|---|---|
| — | — | none yet |

Candidates seen so far (not drafted): none. Items from the predecessor (HiGHS `time_limit` on `s100`, ERGO-Code/HiGHS
#3314) are optopt's and are not re-filed from here.
