#!/usr/bin/env python3
"""Inline the MAE figures and generate the tables into report_mae.html."""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figs"

CONFIG_ROWS = [
    ("base",   "hres",                      8,  "0.25°", "baseline（三個變因共用）", True),
    ("res0.5", "hres",                      8,  "0.5°",  "解析度", False),
    ("res1.5", "hres",                      8,  "1.5°",  "解析度", False),
    ("w4",     "hres",                      4,  "0.25°", "厚度", False),
    ("w12",    "hres",                     12,  "0.25°", "厚度", False),
    ("w16",    "hres",                     16,  "0.25°", "厚度", False),
    ("gt",     "ground truth（逐時）",        8,  "0.25°", "來源", False),
    ("gt6",    "ground truth（6 小時一格）",   8,  "0.25°", "來源", False),
]
GROUPS = [("baseline", ["base"]), ("解析度", ["res0.5", "res1.5"]),
          ("厚度", ["w4", "w12", "w16"]), ("來源", ["gt", "gt6"])]
LABEL = {"base": "base — hres / w8 / 0.25°", "res0.5": "res0.5", "res1.5": "res1.5",
         "w4": "w4", "w12": "w12", "w16": "w16",
         "gt": "gt — ERA5 逐時", "gt6": "gt6 — ERA5，6 小時一格"}


def config_table():
    return "\n".join(
        f'<tr{" class=\"base\"" if is_base else ""}><td><code>{name}</code></td><td>{src}</td>'
        f'<td class="n2">{w}</td><td class="n2">{res}</td><td>{factor}</td></tr>'
        for name, src, w, res, factor, is_base in CONFIG_ROWS)


def summary_table(s):
    def cell(key, field, fmt=".1f"):
        v = s.get(key, {}).get(field)
        return f'<td class="n2">{v:{fmt}}</td>' if v is not None else '<td class="n2">—</td>'

    out = []
    for gname, names in GROUPS:
        out.append(f'<tr class="grp"><td colspan="11">{gname}</td></tr>')
        for n in names:
            cls = ' class="base"' if n == "base" else ""
            row = [f'<tr{cls}><td>{LABEL[n]}</td>']
            for view in ("avg", "00z"):
                k = f"{view}/{n}"
                for f in ("msl@24h", "msl@72h", "msl@168h", "msl_mean", "z_500_mean"):
                    row.append(cell(k, f))
            row.append("</tr>")
            out.append("".join(row))
    return "\n".join(out)


def main():
    s = json.load(open(DATA / "summary_mae.json"))
    html = (HERE / "report_mae_template.html").read_text()
    html = html.replace("__CONFIG_TABLE__", config_table())
    html = html.replace("__SUMMARY_TABLE__", summary_table(s))

    def sub(m):
        p = FIGS / f"{m.group(1)}.svg"
        if not p.exists():
            raise SystemExit(f"missing figure: {p}")
        return p.read_text().strip()

    html = re.sub(r"__FIG:([a-z0-9_]+)__", sub, html)
    left = re.findall(r"__[A-Z0-9_:]+__", html)
    if left:
        raise SystemExit(f"unsubstituted: {left}")

    out = HERE / "report_mae.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
