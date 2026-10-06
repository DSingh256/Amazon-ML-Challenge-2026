"""
All settings of the pipeline in one place.

Change numbers here instead of inside the code.
"""
from dataclasses import dataclass, field

SEED = 42


@dataclass
class Retriever:
    """One way of searching for candidates.

    name        : short label, also used for column names in the output
    text_column : which cleaned text column to compare (see normalize.py)
    k_forward   : for each Source 1 record, keep this many best records
                  from Source 2 and again this many from Source 3
    k_reverse   : for each Source 2/3 record, keep this many best Source 1
                  records (0 = do not search in this direction)
    """
    name: str
    text_column: str
    k_forward: int
    k_reverse: int
    fields: tuple = ()      # text columns whose pieces are chosen separately (see max_query_ngrams)


@dataclass
class BlockingConfig:
    # The searches we run. On the sample, "name_addr" alone found 99.6% of the
    # true pairs; the extra "name" and "addr" searches added only +0.2% but
    # tripled the time, so they are off. (To turn one back on, add e.g.
    #   Retriever("name", "name_text", k_forward=10, k_reverse=2)  )
    retrievers: list = field(default_factory=lambda: [
        Retriever("name_addr", "name_addr_text", k_forward=15, k_reverse=10),
    ])
    # Reverse k: 3 -> 10 found many more pairs (a Source 1 record with 5+ look-alike
    # neighbours was cut off at 3); 20 added nothing more. The search work does not depend on k.

    # Text is cut into overlapping 3-letter pieces ("delta" -> "del", "elt", "lta").
    ngram_range: tuple = (3, 3)

    # Each query only uses its most informative pieces (the rarest ones).
    # Fewer pieces = faster search.
    max_query_ngrams: int = 30
    # With Retriever.fields the pieces are chosen inside every field: this many for the
    # first field (name) and for the second (address); a short or missing field leaves its
    # room to the other one.
    field_quota: tuple = (14, 16)

    # Reverse search (Source 2/3 record -> its Source 1 record): score = this share x "how many of
    # the query's pieces the Source 1 record contains" + the rest x cosine. 0 = plain cosine.
    # Dense sample (reverse k 10): 0.7 finds the short Source 2/3 texts (name only) that
    # cosine ranks below long Source 1 records. Costs no search work.
    reverse_coverage: float = 0.7

    # Very common pieces (like "str" from "street") make the search slow and say
    # little. A piece found in more than this share of the documents is dropped
    # from the query ...
    max_ngram_share: float = 0.001
    # ... but every query always keeps at least this many of its rarest pieces,
    # so short texts still find something.
    # This is the setting that matters most. Dense sample (all records of a few cities, as
    # crowded as the full data), with reverse k 10 and coverage 0.7:
    #   12 -> pair recall 0.9816 (work 1x) | 16 -> 0.9873 (1.9x) | 20 -> 0.9893 (3.4x)
    #   24 -> 0.9901 (5.7x) | 30 -> 0.9906 (11.4x)
    # The work per query x document is the same on the sample and the full data, so the
    # ratios hold there too: 20 is ~7 h for the biggest country of train (one notebook).
    min_query_ngrams: int = 20

    # How much work one matrix multiplication may do at a time (memory stays small:
    # only the top k of every query is kept).
    work_budget: int = 1_000_000_000

    # Documents are searched in blocks of this many (None = all at once). Same result,
    # but faster: the work area of one block fits in the CPU cache.
    # Kaggle benchmark, 20k India queries x 2.3M docs (4 CPUs), million work units / second:
    #   all docs at once 49-53 | blocks of 500k 116 | 250k 161 | 100k 202
    doc_block_size: int = 100_000

    # CPU threads for the search (only used with the sparse_dot_topn package).
    n_threads: int = 4

    # Final number of candidates kept per Source 1 record. The challenge ranks smaller
    # candidate sets higher. Dense sample (same model), holdout macro F0.5:
    #   30 -> 0.9818 | 20 -> 0.9796 | 15 -> 0.9800 | 10 -> 0.9782   (noise about +-0.0015)
    max_candidates: int = 15

    # Unsearched-best rescue: in addition to the top `max_candidates`, keep the best 1..`rescue`
    # pairs per record that scored DIRECTLY below the last kept candidate (no new search: the
    # raw top-k result already contains them). The model decides, guided by the `search_margin`
    # feature: a true copy is very rarely the worst-looking pair of its own record, so the
    # pairs that score below EVERY kept candidate of a record are almost never matches - and
    # among the ones that look better, real but hard matches hide (native-script names,
    # addresses reduced to a city; see Documentation_template.md, "remaining blocking misses").
    # 0 = off: the SUBMITTED outputs were produced without rescue (30 candidates per record).
    # A post-submission experiment (see Documentation_template.md, Appendix C); measure with the
    # blocking report ("PAIR RECALL" should rise more than the candidate count does) before use.
    rescue: int = 0


BLOCKING = BlockingConfig()

# Part of the training Source 1 records used to learn the model (all of them
# still take part in blocking, so the "competition" features look the same as
# on test). 10% of these records are held out for measuring.
TRAIN_FRACTION = 0.25
HOLDOUT_PERCENT = 10


@dataclass
class CrossEncoderConfig:
    """The "LLM layer" (cross_encoder.py): a small language model reads both records."""
    # MIT licence, 278M parameters, 100 languages (reads Hindi / Kannada / Telugu script too)
    model: str = "FacebookAI/xlm-roberta-base"
    max_tokens: int = 96            # name + address of both records; longer texts are cut
    # It learns from the candidates of this many training Source 1 records that the LightGBM
    # models never use: the first max_learning_rank candidates of each (+ its true matches),
    # the far ones are easy non-matches. v18: 40,000 x all 15 -> 150,000 x 6; v19: 300,000 x 6.
    train_records: int = 300_000
    max_learning_rank: int = 6
    epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 3e-5
    infer_batch: int = 512
    # Only the pairs stage 1 is unsure about get a score: |p1 - 0.5| < max_certainty
    # (0.499 = p1 between 0.001 and 0.999), and at most score_budget test pairs (the most unsure).
    # v20: 0.49 -> 0.499: the LLM layer fixed many France errors (LB 0.968 -> 0.975), so it
    # also reads pairs stage 1 is quite (but not fully) sure about.
    max_certainty: float = 0.499
    score_budget: int = 6_000_000


CROSS_ENCODER = CrossEncoderConfig()
