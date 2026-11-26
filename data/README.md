# Data

## Source

Medical Appointment No Shows, Kaggle: https://www.kaggle.com/datasets/joniarroba/noshowappointments

File used: `KaggleV2-May-2016.csv`

## Dataset record

| Item | Value |
|---|---|
| Licence | [write the licence name from the Kaggle Licence tab] |
| Date downloaded | [write the day you downloaded it] |
| Rows and columns | 110,527 rows, 14 columns |
| File size | 10,739,535 bytes |
| MD5 | cc55525de6e2b615aeba50095e8aaa95 |

Training refuses to run on a file whose MD5 differs from `EXPECTED_DATA_MD5` in `src/config.py`.

## Layout

- `raw/` holds the original download (git-ignored). Put the CSV here.
- `processed/` holds the cleaned model table and `cleaning_report.json`, written by `python src/train.py` and by notebook 01.

## Freeze rule

Download once and never edit the raw file. Every tracked run records the file hash.