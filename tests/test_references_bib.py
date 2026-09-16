"""Structural checks on the shipped paper/references.bib (live-audited entries)."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BIB_PATH = REPO_ROOT / "paper" / "references.bib"


def _read_braced(text: str, start: int) -> tuple[str, int]:
    depth = 0
    i = start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start + 1 : i], i + 1
        i += 1
    return text[start + 1 :], len(text)


def parse_bib(text: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    i = 0
    while i < len(text):
        at = text.find("@", i)
        if at < 0:
            break
        m = re.match(r"@(\w+)\s*\{", text[at:], re.I)
        if not m:
            i = at + 1
            continue
        entry_type = m.group(1).lower()
        j = at + m.end()
        key_end = text.find(",", j)
        if key_end < 0:
            break
        key = text[j:key_end].strip()
        j = key_end + 1
        fields: dict[str, str] = {"_type": entry_type, "_key": key}
        depth = 1
        while j < len(text) and depth > 0:
            c = text[j]
            if c == "{":
                depth += 1
                j += 1
                continue
            if c == "}":
                depth -= 1
                j += 1
                continue
            if depth == 1:
                fm = re.match(r"\s*([A-Za-z][A-Za-z0-9_-]*)\s*=\s*", text[j:])
                if fm:
                    fname = fm.group(1).lower()
                    j = j + fm.end()
                    if j < len(text) and text[j] == "{":
                        val, j = _read_braced(text, j)
                        fields[fname] = val
                    continue
            j += 1
        entries.append(fields)
        i = j
    return entries


class ReferencesBibTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = BIB_PATH.read_text(encoding="utf-8")
        cls.entries = parse_bib(cls.text)
        cls.by_key = {e["_key"]: e for e in cls.entries}

    def test_bib_file_is_the_shipped_paper_bibliography(self) -> None:
        self.assertTrue(BIB_PATH.is_file())
        self.assertGreater(len(self.entries), 50)
        self.assertEqual(len(self.entries), len(self.by_key), "duplicate cite keys")

    def test_every_entry_has_core_fields_and_a_live_identifier(self) -> None:
        for entry in self.entries:
            key = entry["_key"]
            self.assertTrue(entry.get("title"), key)
            self.assertTrue(entry.get("author"), key)
            self.assertTrue(entry.get("year"), key)
            self.assertTrue(re.fullmatch(r"\d{4}", entry["year"]), f"{key} year={entry['year']}")
            doi = entry.get("doi")
            url = entry.get("url")
            eprint = entry.get("eprint")
            self.assertTrue(doi or url or eprint, f"{key} has no doi/url/eprint")
            if doi:
                self.assertRegex(doi, r"^10\.\S+/\S+$", f"{key} doi={doi}")

    def test_corrected_records_match_live_authorities(self) -> None:
        spirtes = self.by_key["spirtes2000causation"]
        self.assertEqual(spirtes["year"], "2001")
        self.assertEqual(spirtes["doi"].lower(), "10.7551/mitpress/1754.001.0001")

        hollmann = self.by_key["hollmann2025tabpfn"]
        self.assertEqual(hollmann["number"], "8045")
        self.assertEqual(hollmann["volume"], "637")
        self.assertIn("319", hollmann["pages"])
        self.assertIn("326", hollmann["pages"])

        ma = self.by_key["ma2025causalfm"]
        self.assertRegex(ma["pages"], r"79065\s*(to|--|-)\s*79098")

        thumm = self.by_key["thumm2026causaltimeprior"]
        self.assertEqual(thumm["eprint"], "2603.11090")
        self.assertEqual(thumm["doi"], "10.48550/arXiv.2603.11090")

        ospc = self.by_key["ospc2026correction"]
        self.assertNotIn("volume", ospc)
        self.assertNotEqual(ospc.get("volume"), "306")
        self.assertEqual(ospc["doi"], "10.48550/arXiv.2603.12037")

        zheng = self.by_key["zheng2018notears"]
        self.assertIn("Ravikumar, Pradeep K", zheng["author"])
        self.assertIn("Xing, Eric", zheng["author"])
        self.assertNotIn("Eric P.", zheng["author"])

    def test_no_unregistered_pmlr_volume_is_cited(self) -> None:
        for entry in self.entries:
            if entry.get("volume") == "306" and "Machine Learning Research" in (
                entry.get("series") or ""
            ):
                self.fail(f"{entry['_key']} cites unpublished PMLR volume 306")


if __name__ == "__main__":
    unittest.main()
