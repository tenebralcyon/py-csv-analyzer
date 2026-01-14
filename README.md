\# Python CSV Analyzer



A lightweight Python CLI tool that profiles a CSV file and prints a readable report.



It’s designed for quick “data sanity checks” before you do analysis, dashboards, or ML.



\## What it reports

\- Rows / columns

\- Missing values per column

\- Inferred column type (numeric / text / mixed)

\- Numeric stats: min / max / mean / standard deviation

\- Top K most common values for text columns

\- Numeric outliers using:

&nbsp; - Z-score threshold

&nbsp; - IQR (interquartile range) bounds

\- Text cleanup suggestions:

&nbsp; - Detects “near duplicates” caused by casing/extra spaces (e.g., `Vancouver`, ` VANCOUVER `, `vancouver`)

&nbsp; - Suggests recommended normalized values



\## Group-by summaries (built-in)

You can generate group summaries like “average score by city”:



\- count / sum / mean / min / max per group



\## Usage



Basic analysis (prints report):

py analyze\_csv.py path\\to\\file.csv



Show top 10 text values per column:

py analyze\_csv.py path\\to\\file.csv --top 10



Analyze only the first 1000 rows:

py analyze\_csv.py path\\to\\file.csv --sample-rows 1000



Save report to a file:

py analyze\_csv.py path\\to\\file.csv --out report.txt



Group-by summary (example: average score by city):

py analyze\_csv.py path\\to\\file.csv --group-by city --value score



Outlier sensitivity controls:

py analyze\_csv.py path\\to\\file.csv --z 2.5 --iqr-k 1.0



\## Notes

\- Built on Windows 11

\- Uses only Python standard library (no extra packages required)



