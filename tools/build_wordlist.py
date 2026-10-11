"""
Build a word-frequency list from modern Tamil text and merge it with the classical
(Project Madurai) vocabulary. Output feeds the synthetic generator (--vocab) and, later,
lexicon-constrained decoding and the corrector.

  # Tamil Wikipedia dump (streamed; no need to unzip the .bz2)
  .venv\\Scripts\\python tools\\build_wordlist.py ^
      --wiki external\\tawiki\\tawiki-latest-pages-articles.xml.bz2 ^
      --merge data\\text_corpus\\tamil_correction_vocabulary.json ^
      --out data\\text_corpus\\tamil_words_merged.json

  # any folder of .txt files (news, books you have rights to use, ...)
  .venv\\Scripts\\python tools\\build_wordlist.py --txt-dir some\\folder --out data\\text_corpus\\extra_words.json

Only words made entirely of characters in src/charset.py are kept (after NFC normalisation).
"""
import argparse
import bz2
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import charset as cs  # noqa: E402

TAMIL_WORD = re.compile(r"[஀-௿]+")
# crude wiki-markup stripping: templates, tables, refs, links, tags
_MARKUP = [re.compile(p, re.S) for p in (r"\{\{.*?\}\}", r"\{\|.*?\|\}", r"<ref[^>]*>.*?</ref>", r"<[^>]+>",
                                          r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\[https?://[^\]]*\]")]


def clean_wiki(text):
    for i, rx in enumerate(_MARKUP):
        text = rx.sub(r"\1" if i == 4 else " ", text)
    return text


def iter_wiki_text(path):
    """Yield the <text> body of each page from a MediaWiki XML(.bz2) dump, streaming."""
    opener = bz2.open if str(path).endswith(".bz2") else open
    buf, inside = [], False
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            while line:                       # a line may close one page and open the next
                if not inside:
                    i = line.find("<text")
                    if i < 0:
                        break
                    k = line.find(">", i)
                    if k < 0:
                        break
                    inside, line = True, line[k + 1:]
                j = line.find("</text>")
                if j < 0:
                    buf.append(line)
                    break
                buf.append(line[:j])
                yield "".join(buf)
                buf, inside, line = [], False, line[j + 7:]


def count_words(texts, counter, min_len=1, max_len=40):
    for t in texts:
        for w in TAMIL_WORD.findall(cs.normalize(t)):
            if min_len <= len(w) <= max_len and cs.is_encodable(w):
                counter[w] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", help="tawiki-*-pages-articles.xml(.bz2)")
    ap.add_argument("--txt-dir", help="folder of .txt files (recursive)")
    ap.add_argument("--merge", nargs="*", default=[], help="existing {word: freq} JSON files to merge in")
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    c = Counter()
    if a.wiki:
        n = 0
        for body in iter_wiki_text(a.wiki):
            count_words([clean_wiki(body)], c)
            n += 1
            if n % 20000 == 0:
                print(f"  {n} pages, {len(c)} distinct words", flush=True)
        print(f"wiki: {n} pages, {len(c)} distinct words")
    if a.txt_dir:
        files = list(Path(a.txt_dir).rglob("*.txt"))
        for p in files:
            count_words([p.read_text(encoding="utf-8", errors="ignore")], c)
        print(f"txt: {len(files)} files, {len(c)} distinct words so far")
    modern = {w: f for w, f in c.items() if f >= a.min_freq}
    merged = Counter(modern)
    for m in a.merge:
        with open(m, encoding="utf-8") as fh:
            for w, f in json.load(fh).items():
                w = cs.normalize(w)
                if cs.is_encodable(w):
                    merged[w] += int(f)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(dict(merged.most_common()), fh, ensure_ascii=False)
    lens = [len(w) for w in merged]
    print(f"wrote {a.out}: {len(merged):,} words (modern-only {len(modern):,}); "
          f"median length {sorted(lens)[len(lens)//2] if lens else 0}, words >= 16 chars: {sum(l >= 16 for l in lens):,}")


if __name__ == "__main__":
    main()
