#!/usr/bin/env python3
"""Inline the figures and generate the data-driven tables/readouts into report.html."""

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGS = HERE / "figs"

CONFIG_ROWS = [
    ("base",   "hres",         8,  "0.25°", "baseline（三個變因共用）", True),
    ("res0.5", "hres",         8,  "0.5°",  "解析度", False),
    ("res1.5", "hres",         8,  "1.5°",  "解析度", False),
    ("w4",     "hres",         4,  "0.25°", "厚度", False),
    ("w12",    "hres",        12,  "0.25°", "厚度", False),
    ("w16",    "hres",        16,  "0.25°", "厚度", False),
    ("gt",     "ground truth（逐時）", 8,  "0.25°", "來源", False),
    ("gt6",    "ground truth（6 小時一格）", 8, "0.25°", "來源", False),
]
GROUPS = [("baseline", ["base"]), ("解析度", ["res0.5", "res1.5"]),
          ("厚度", ["w4", "w12", "w16"]), ("來源", ["gt", "gt6"])]
LABEL = {"base": "base — hres / w8 / 0.25°", "res0.5": "res0.5", "res1.5": "res1.5",
         "w4": "w4", "w12": "w12", "w16": "w16", "gt": "gt — ERA5 逐時", "gt6": "gt6 — ERA5，6 小時一格"}


def config_table():
    out = []
    for name, src, w, res, factor, is_base in CONFIG_ROWS:
        cls = ' class="base"' if is_base else ""
        out.append(f'<tr{cls}><td><code>{name}</code></td><td>{src}</td>'
                   f'<td class="n2">{w}</td><td class="n2">{res}</td><td>{factor}</td></tr>')
    return "\n".join(out)


def summary_table(s):
    def cell(key, field, fmt):
        v = s.get(key, {}).get(field)
        return f'<td class="n2">{v:{fmt}}</td>' if v is not None else '<td class="n2">—</td>'

    out = []
    for gname, names in GROUPS:
        out.append(f'<tr class="grp"><td colspan="11">{gname}</td></tr>')
        for n in names:
            cls = ' class="base"' if n == "base" else ""
            row = [f'<tr{cls}><td>{LABEL[n]}</td>']
            for scale in ("day", "month"):
                k = f"{scale}/{n}"
                row.append(cell(k, "mae_int_msl_overall", ".1f"))
                row.append(cell(k, "mae_int_z_overall", ".1f"))
                row.append(cell(k, "dcos_overall", ".4f"))
                row.append(cell(k, "burst_over_rest_msl", ".3f"))
                row.append(cell(k, "clock_peak_to_trough_msl", ".2f"))
            row.append("</tr>")
            out.append("".join(row))
    return "\n".join(out)


def readout(s, factor):
    """Short, purely factual readout, generated from summary.json so it cannot drift."""
    def val(scale, n, f):
        return s.get(f"{scale}/{n}", {}).get(f)

    def line(label, names, labels, field, fmt):
        bits = []
        for n, lb in zip(names, labels):
            d, m = val("day", n, field), val("month", n, field)
            ds = f"{d:{fmt}}" if d is not None else "—"
            ms = f"{m:{fmt}}" if m is not None else "—"
            bits.append(f'{lb} <span class="n">{ds}</span> / <span class="n">{ms}</span>')
        return f"<li>{label}：" + "，".join(bits) + "</li>"

    spec = {
        "resolution": (["base", "res0.5", "res1.5"], ["0.25°", "0.5°", "1.5°"]),
        "width": (["w4", "base", "w12", "w16"], ["4", "8", "12", "16"]),
        "source": (["base", "gt", "gt6"], ["hres", "gt 逐時", "gt6 六小時"]),
    }[factor]
    names, labels = spec
    html = ["<p>下面每一項都是「單日 / 月平均」：</p>", "<ul>"]
    html.append(line("內部 MAE(msl)", names, labels, "mae_int_msl_overall", ".1f"))
    html.append(line("內部 MAE(z)", names, labels, "mae_int_z_overall", ".1f"))
    html.append(line("1 − cos", names, labels, "dcos_overall", ".4f"))
    html.append(line("burst/rest", names, labels, "burst_over_rest_msl", ".3f"))
    html.append(line("peak/trough", names, labels, "clock_peak_to_trough_msl", ".2f"))
    html.append("</ul>")
    return "\n".join(html)


def main():
    s = json.load(open(DATA / "summary.json"))
    html = (HERE / "report_template.html").read_text()

    html = html.replace("__CONFIG_TABLE__", config_table())
    html = html.replace("__SUMMARY_TABLE__", summary_table(s))
    for f in ("resolution", "width", "source"):
        html = html.replace(f"__READ_{f}__", readout(s, f))

    wave = (HERE / "wavenumber_readout.html")
    html = html.replace("__READ_wavenumber__",
                        wave.read_text() if wave.exists() else "")

    def sub(m):
        p = FIGS / f"{m.group(1)}.svg"
        if not p.exists():
            raise SystemExit(f"missing figure: {p}")
        return p.read_text().strip()

    html = re.sub(r"__FIG:([a-z0-9_]+)__", sub, html)
    left = re.findall(r"__[A-Z0-9_:]+__", html)
    if left:
        raise SystemExit(f"unsubstituted: {left}")

    out = HERE / "report.html"
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
