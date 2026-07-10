python create_tsv.py ^
--results_json "C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\results\xcometmetriclayer.wmt2025esa.full.result.redo.jsonl" ^
--prediction_tsv "C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\res\xcometmetriclayer.tsv"


python create_tsv.py ^
--results_json "C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\results\xcomet.wmt2025esa.full.result.redo.jsonl" ^
--prediction_tsv "C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\results\xcomet.tsv"

python create_gold_tsv.py ^
--gold_json "C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\wmt25-genmt-humeval.jsonl" ^
--gold_tsv "C:\Users\Benjamin Pong\OneDrive\Documents\Machine Translation Metrics\WMT2026\ref\task2_refs.tsv"
