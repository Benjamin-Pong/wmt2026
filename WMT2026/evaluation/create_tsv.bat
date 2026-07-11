python create_tsv.py ^
--results_json "C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\results\xcometmetriclayer.wmt2025esa.full.result.redo.jsonl" ^
--prediction_tsv "C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\input\res\xcometlayer.tsv"


python create_tsv.py ^
--results_json "C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\results\xcomet.wmt2025esa.full.result.redo.jsonl" ^
--prediction_tsv "C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\input\res\xcomet.tsv"

python create_gold_tsv.py ^
--gold_json "C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\results\xcomet.wmt2025esa.full.result.redo.jsonl" ^
--gold_tsv "C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\input\ref\task2_refs.tsv"
