\# Python CSV Analyzer



A lightweight Python CLI tool that analyzes a CSV file and prints a readable summary report.



It reports:

\- rows and columns

\- missing values per column

\- inferred column type (numeric / text / mixed)

\- numeric stats (min / max / mean / standard deviation)

\- top K most common values for text columns



\## Usage



Analyze a CSV (print report to terminal):

py analyze\_csv.py path\\to\\file.csv



Show top 10 text values per column:

py analyze\_csv.py path\\to\\file.csv --top 10



Analyze only the first 1000 rows (useful for big files):

py analyze\_csv.py path\\to\\file.csv --sample-rows 1000



Save the report to a text file:

py analyze\_csv.py path\\to\\file.csv --out report.txt



\## Notes

\- Built on Windows 11

\- Uses only Python standard library (no extra packages required)

