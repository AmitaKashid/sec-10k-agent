\# Evaluation evidence



\## Qualitative evaluation — October 27, 2026



Report: 20261027T181136Z.md



This run evaluated six golden qualitative questions and no adversarial

questions. The questions covered Apple, Microsoft, Alphabet, and Nvidia.



The application used Gemini embeddings and Groq-hosted chat models.

The judge model was openai/gpt-oss-120b.



| Measurement | Reported result |

|---|---|

| Questions evaluated | 6 |

| Citation accuracy | 100% |

| Unsupported claims | 0 of 30 judged claims |

| False-refusal rate | 0% |

| Reported median graph latency | 10,308 ms |

| Keyword retrieval diagnostic at 5 | 100% |

| Keyword retrieval diagnostic at 8 | 100% |



\## Interpretation



The judge found supporting evidence for all 30 claims it assessed

in this sample. Citation accuracy passed the harness's checks.



This result does not establish that all future answers are correct,

complete, relevant, or free of hallucinations.



\## Measurements not established by this run



\- Numerical accuracy: no numerical cases were included.

\- Adversarial refusal accuracy: no adversarial cases were included.

\- Standard retrieval Recall@k: the reported retrieval diagnostics use

&#x20; keyword matching, not independently labeled relevant-document sets.

\- Hosted response latency: this was an in-process graph evaluation,

&#x20; not a browser-to-Vercel-to-Render benchmark.



The report displays passing numerical and refusal metrics even where

the corresponding test categories have no examples. Those values

must not be interpreted as measured success.



\## Limitations



\- The sample contains only six questions.

\- An LLM judge can make mistakes.

\- The judge uses the same model as the configured validation stage,

&#x20; so its assessment is not independent of that model family.

\- Claim support does not measure whether an answer covers every

&#x20; part of the question.

\- The reported latency excludes the subsequent judge call.

\- The harness's even-sample median calculation selects the upper

&#x20; middle value rather than averaging the two middle values.

\- These results describe this saved run, not a fresh evaluation

&#x20; of every later code revision.



\## Evidence-driven fixes



During earlier evaluation, the judge received only the first 1,200

characters of each source chunk. Supporting evidence later in a chunk

could therefore be omitted from its input.



The judge was updated to receive the complete source chunk, and its

cache version was changed to prevent reuse of earlier verdicts.



A separate sentence-extraction fix preserved "U.S." within sentences,

avoiding an incorrect split before phrases such as "District Court".



Unit tests were added for these cases.



\## Automated checks versus live evaluation



GitHub Actions checks provider compatibility with mocked calls,

evaluation logic, database schema behavior, frontend types, and lint.



Those checks do not replace live answer-quality evaluation against

filings and XBRL facts.

