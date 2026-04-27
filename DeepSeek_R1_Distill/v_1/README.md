## Prompt
SYSTEM_PROMPT = """\
You are a BGL (Blue Gene/L) log anomaly classifier.

Classify each log line as:
0 = NORMAL
1 = ABNORMAL

Rules:
0 NORMAL:
- INFO or WARNING severity messages
- Corrected hardware errors (cache parity corrected, DDR errors corrected, CE syms)
- Alignment exceptions
- Core file generation
- Routine retry or retransmission messages
- Register dumps that are part of a normal diagnostic sequence

1 ABNORMAL:
- rts panic
- kernel terminated
- Lustre mount FAILED
- Link has been severed
- Connection reset by peer / Connection timed out
- data TLB error interrupt
- data storage interrupt
- Fatal errors that stop execution

NORMAL examples (label 0):
Log: 1117838978 2005.06.03 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
Answer: 0

Log: 1118271740 2005.06.08 R03-M1-N9-C:J09-U11 RAS KERNEL INFO 1 ddr errors(s) detected and corrected on rank 0, symbol 25, bit 1
Answer: 0

Log: 1132236972 2005.11.17 R72-M1-N6-C:J04-U01 RAS KERNEL INFO 26741629 torus sender z- retransmission error(s) detected and corrected over 268 seconds
Answer: 0

ABNORMAL examples (label 1):
Log: 1121115817 2005.07.11 R01-M0-N1-C:J09-U11 RAS KERNEL FATAL rts panic! - stopping execution
Answer: 1

Log: 1124071359 2005.08.14 R21-M0-N8-I:J18-U11 RAS APP FATAL ciod: Error reading message prefix after LOAD_MESSAGE on CioStream socket to 172.16.96.116:42213: Link has been severed
Answer: 1

Log: 1126202752 2005.09.08 R01-M1-N4-I:J18-U11 RAS KERNEL FATAL Lustre mount FAILED : bglio23 : point /p/gb1
Answer: 1

Return exactly one character: 0 or 1.
"""


## Results
============================================================
RESULTS
============================================================
Accuracy  : 0.6385
Precision : 0.1651
Recall    : 1.0000
F1-Score  : 0.2834

Confusion Matrix:
                   Pred Normal  Pred Abnormal
  Actual Normal        1,134            723
  Actual Abnormal          0            143

Performance:
  Avg time/log : 8029.8 ms
  Total time   : 16059.7 s  (267.7 min)
  GPU memory   : 0 MB

Misclassified: 723 / 2,000 (36.1%)

