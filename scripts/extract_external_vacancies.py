"""Per-step energies of the chalcogenide vacancy relaxations of Mosquera-Lois et al.

Source: Zenodo record 10.5281/zenodo.10579527, file defect_relaxations.tar.gz
(20.6 GB). The archive is streamed once; every vasprun.xml segment of every
distortion is parsed for its ionic-step energies, electronic step counts and,
for the Unperturbed and rattled runs, the first and last atomic positions.

    python scripts/extract_external_vacancies.py defect_relaxations.tar.gz segments.csv

The output is reduced to one row per vacancy by
`relaxfree.external.summarise_segments`.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import re
import sys
import tarfile
import xml.etree.ElementTree as ET

SEGMENT = re.compile(
    r"^(?P<top>[^/]+)/(?P<host>[^/]+)/(?P<defect>[^/]+)/(?P<dist>[^/]+)/(?P<file>vasprun\.xml(?:_.+\.gz)?)$"
)
KEEP_POSITIONS = ("Unperturbed", "Bond_Distortion_0.0%")
FIELDS = ["top", "host", "defect", "distortion", "file", "n_steps", "natoms", "complete", "nelm",
          "e_first", "e_last", "energies", "n_scsteps", "first_lattice", "first_frac", "last_frac"]


def _vectors(node) -> list[list[float]]:
    return [[float(x) for x in v.text.split()] for v in node]


def parse_vasprun(data: bytes) -> dict:
    """Energies per ionic step; a run that stopped mid-write keeps the steps read so far."""
    st = {"e": [], "nsc": [], "nelm": None, "na": None, "lat": None, "f0": None, "f1": None, "complete": True}
    try:
        for _, el in ET.iterparse(io.BytesIO(data), events=("end",)):
            if el.tag == "i" and el.get("name") == "NELM" and st["nelm"] is None:
                st["nelm"] = int(float(el.text))
            elif el.tag == "atominfo" and el.find("atoms") is not None:
                st["na"] = int(el.find("atoms").text)
            elif el.tag == "structure" and el.get("name") == "initialpos":
                st["lat"] = _vectors(el.find("crystal/varray[@name='basis']"))
                st["f0"] = _vectors(el.find("varray[@name='positions']"))
            elif el.tag == "calculation":
                e = el.find("energy/i[@name='e_fr_energy']")
                if e is not None:
                    try:
                        st["e"].append(float(e.text))
                    except ValueError:  # VASP prints asterisks on overflow
                        st["e"].append(float("nan"))
                    st["nsc"].append(len(el.findall("scstep")))
                pos = el.find("structure/varray[@name='positions']")
                if pos is not None:
                    st["f1"] = _vectors(pos)
                el.clear()
    except ET.ParseError:
        st["complete"] = False
    return st


def main(src: str, out_csv: str) -> None:
    with tarfile.open(src, "r|gz") as tf, open(out_csv, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(FIELDS)
        for member in tf:
            m = SEGMENT.match(member.name) if member.isfile() else None
            if not m:
                continue
            data = tf.extractfile(member).read()
            if m["file"].endswith(".gz") and data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)  # a few archived segments were stored uncompressed
            st = parse_vasprun(data)
            keep = m["dist"] in KEEP_POSITIONS
            e = st["e"]
            writer.writerow([
                m["top"], m["host"], m["defect"], m["dist"], m["file"], len(e), st["na"], int(st["complete"]),
                st["nelm"], e[0] if e else "", e[-1] if e else "", json.dumps(e), json.dumps(st["nsc"]),
                json.dumps(st["lat"]) if keep and st["lat"] else "",
                json.dumps(st["f0"]) if keep and st["f0"] else "",
                json.dumps(st["f1"]) if keep and st["f1"] else "",
            ])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
