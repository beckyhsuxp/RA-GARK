#!/usr/bin/env python3
from __future__ import annotations

import copy
import re
import struct
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional
from xml.etree import ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
SOURCE_MD = ROOT / "Document" / "notes" / "SLIDES.md"
TEMPLATE_PPTX = ROOT / "Document" / "graph" / "gragh.pptx"
OUTPUT_PPTX = ROOT / "Document" / "notes" / "SLIDES.pptx"

P = "http://schemas.openxmlformats.org/presentationml/2006/main"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT = "http://schemas.openxmlformats.org/package/2006/content-types"
EP = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
VT = "http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes"

NS = {"p": P, "a": A, "r": R, "pr": PKG_REL, "ct": CONTENT, "ep": EP, "vt": VT}

for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)


def qn(prefix: str, tag: str) -> str:
    return f"{{{NS[prefix]}}}{tag}"


EMU = 914400
SLIDE_W = 12192000
SLIDE_H = 6858000

HEADER_H = int(0.86 * EMU)
LEFT_MARGIN = int(0.62 * EMU)
RIGHT_MARGIN = int(0.62 * EMU)
TOP_MARGIN = int(0.18 * EMU)
BODY_TOP = int(1.12 * EMU)
BODY_BOTTOM = int(0.45 * EMU)

HEADER_FILL = "16324F"
ACCENT_FILL = "2B8AA3"
TITLE_COLOR = "FFFFFF"
BODY_COLOR = "1C2430"
MUTED_COLOR = "6A7380"


@dataclass
class Para:
    text: str
    bold: bool = False
    mono: bool = False
    size: Optional[int] = None


@dataclass
class SlideSpec:
    number: int
    title: str
    paras: List[Para]
    image: Optional[Path] = None


def emu(inches: float) -> int:
    return int(inches * EMU)


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Unsupported image format: {path}")
    return struct.unpack(">II", header[16:24])


def parse_slides(md_text: str) -> list[tuple[int, str, list[str]]]:
    heading_re = re.compile(r"^## Slide (\d+) — (.+)$")
    slides: list[tuple[int, str, list[str]]] = []
    current: tuple[int, str, list[str]] | None = None

    for line in md_text.splitlines():
        m = heading_re.match(line)
        if m:
            if current is not None:
                slides.append(current)
            current = (int(m.group(1)), m.group(2).strip(), [])
            continue
        if current is None:
            continue
        current[2].append(line)

    if current is not None:
        slides.append(current)
    return slides


def clean_slide_body(lines: list[str]) -> tuple[list[Para], Optional[Path]]:
    paras: list[Para] = []
    image: Optional[Path] = None
    in_code = False

    for raw in lines:
        s = raw.rstrip("\n")
        stripped = s.strip()

        if not stripped or stripped == "---":
            paras.append(Para(text=""))
            continue

        if stripped.startswith("```"):
            in_code = not in_code
            continue

        if stripped == "**圖片**":
            continue

        if stripped.startswith("`") and stripped.endswith("`") and "thesis/img/" in stripped:
            rel = stripped.strip("`")
            image = ROOT / "Document" / rel
            continue

        if in_code:
            paras.append(Para(text=s, mono=True))
            continue

        text = re.sub(r"\*\*(.+?)\*\*", r"\1", stripped)
        text = text.replace("\t", "    ")

        if text.startswith("- "):
            text = "• " + text[2:]

        mono = text.startswith("|") or "|" in text or text.startswith("```")
        if mono:
            paras.append(Para(text=text, mono=True))
        elif text:
            bold = bool(re.fullmatch(r"[^|]+\|[^|]+", text)) is False and raw.strip().startswith("**") and raw.strip().endswith("**")
            paras.append(Para(text=text, bold=bold))
        else:
            paras.append(Para(text=""))

    while paras and not paras[0].text:
        paras.pop(0)
    while paras and not paras[-1].text:
        paras.pop()

    return paras, image


def build_specs() -> list[SlideSpec]:
    md_text = SOURCE_MD.read_text(encoding="utf-8")
    parsed = parse_slides(md_text)
    specs: list[SlideSpec] = []

    for number, title, lines in parsed:
        paras, image = clean_slide_body(lines)
        if number in {12, 18, 22, 23, 31} and image is None:
            raise ValueError(f"Slide {number} expected an image but none was parsed.")
        specs.append(SlideSpec(number=number, title=title, paras=paras, image=image))

    return specs


def make_sp_tree() -> ET.Element:
    sp_tree = ET.Element(qn("p", "spTree"))

    grp = ET.SubElement(sp_tree, qn("p", "nvGrpSpPr"))
    ET.SubElement(grp, qn("p", "cNvPr"), id="1", name="")
    ET.SubElement(grp, qn("p", "cNvGrpSpPr"))
    ET.SubElement(grp, qn("p", "nvPr"))

    grp_pr = ET.SubElement(sp_tree, qn("p", "grpSpPr"))
    xfrm = ET.SubElement(grp_pr, qn("a", "xfrm"))
    ET.SubElement(xfrm, qn("a", "off"), x="0", y="0")
    ET.SubElement(xfrm, qn("a", "ext"), cx="0", cy="0")
    ET.SubElement(xfrm, qn("a", "chOff"), x="0", y="0")
    ET.SubElement(xfrm, qn("a", "chExt"), cx="0", cy="0")

    return sp_tree


def add_rect(sp_tree: ET.Element, shape_id: int, name: str, x: int, y: int, w: int, h: int,
             fill: str, line: Optional[str] = None, radius: bool = False) -> None:
    sp = ET.SubElement(sp_tree, qn("p", "sp"))
    nv = ET.SubElement(sp, qn("p", "nvSpPr"))
    ET.SubElement(nv, qn("p", "cNvPr"), id=str(shape_id), name=name)
    ET.SubElement(nv, qn("p", "cNvSpPr"), txBox="1")
    ET.SubElement(nv, qn("p", "nvPr"))
    sp_pr = ET.SubElement(sp, qn("p", "spPr"))
    xfrm = ET.SubElement(sp_pr, qn("a", "xfrm"))
    ET.SubElement(xfrm, qn("a", "off"), x=str(x), y=str(y))
    ET.SubElement(xfrm, qn("a", "ext"), cx=str(w), cy=str(h))
    geom = ET.SubElement(sp_pr, qn("a", "prstGeom"), prst="roundRect" if radius else "rect")
    ET.SubElement(geom, qn("a", "avLst"))
    if fill:
        solid = ET.SubElement(sp_pr, qn("a", "solidFill"))
        ET.SubElement(solid, qn("a", "srgbClr"), val=fill)
    else:
        ET.SubElement(sp_pr, qn("a", "noFill"))
    ln = ET.SubElement(sp_pr, qn("a", "ln"))
    if line:
        solid = ET.SubElement(ln, qn("a", "solidFill"))
        ET.SubElement(solid, qn("a", "srgbClr"), val=line)
    else:
        ET.SubElement(ln, qn("a", "noFill"))
    tx = ET.SubElement(sp, qn("p", "txBody"))
    ET.SubElement(tx, qn("a", "bodyPr"), wrap="square")
    ET.SubElement(tx, qn("a", "lstStyle"))
    p = ET.SubElement(tx, qn("a", "p"))
    ET.SubElement(p, qn("a", "endParaRPr"))


def add_textbox(
    sp_tree: ET.Element,
    shape_id: int,
    name: str,
    x: int,
    y: int,
    w: int,
    h: int,
    paras: Iterable[Para],
    *,
    default_size: int = 18,
    default_color: str = BODY_COLOR,
    fit_to_box: bool = False,
) -> None:
    sp = ET.SubElement(sp_tree, qn("p", "sp"))
    nv = ET.SubElement(sp, qn("p", "nvSpPr"))
    ET.SubElement(nv, qn("p", "cNvPr"), id=str(shape_id), name=name)
    ET.SubElement(nv, qn("p", "cNvSpPr"), txBox="1")
    ET.SubElement(nv, qn("p", "nvPr"))

    sp_pr = ET.SubElement(sp, qn("p", "spPr"))
    xfrm = ET.SubElement(sp_pr, qn("a", "xfrm"))
    ET.SubElement(xfrm, qn("a", "off"), x=str(x), y=str(y))
    ET.SubElement(xfrm, qn("a", "ext"), cx=str(w), cy=str(h))
    geom = ET.SubElement(sp_pr, qn("a", "prstGeom"), prst="rect")
    ET.SubElement(geom, qn("a", "avLst"))
    ET.SubElement(sp_pr, qn("a", "noFill"))
    ln = ET.SubElement(sp_pr, qn("a", "ln"))
    ET.SubElement(ln, qn("a", "noFill"))

    tx = ET.SubElement(sp, qn("p", "txBody"))
    ET.SubElement(tx, qn("a", "bodyPr"), wrap="square", anchor="t", lIns="0", rIns="0", tIns="0", bIns="0")
    ET.SubElement(tx, qn("a", "lstStyle"))

    for para in paras:
        p = ET.SubElement(tx, qn("a", "p"))
        if not para.text:
            ET.SubElement(p, qn("a", "endParaRPr"), sz=str(default_size * 100))
            continue
        size = para.size or default_size
        r = ET.SubElement(p, qn("a", "r"))
        rPr = ET.SubElement(r, qn("a", "rPr"), sz=str(size * 100))
        if para.bold:
            rPr.set("b", "1")
        if para.mono:
            rPr.set("lang", "en-US")
            latin = ET.SubElement(rPr, qn("a", "latin"), typeface="Courier New")
            ET.SubElement(rPr, qn("a", "ea"), typeface="Courier New")
            ET.SubElement(rPr, qn("a", "cs"), typeface="Courier New")
        else:
            rPr.set("lang", "zh-TW")
        fill = ET.SubElement(rPr, qn("a", "solidFill"))
        ET.SubElement(fill, qn("a", "srgbClr"), val=default_color)
        t = ET.SubElement(r, qn("a", "t"))
        t.text = para.text
        ET.SubElement(p, qn("a", "endParaRPr"), sz=str(size * 100))


def add_picture(
    sp_tree: ET.Element,
    shape_id: int,
    name: str,
    rel_id: str,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    pic = ET.SubElement(sp_tree, qn("p", "pic"))
    nv = ET.SubElement(pic, qn("p", "nvPicPr"))
    ET.SubElement(nv, qn("p", "cNvPr"), id=str(shape_id), name=name)
    ET.SubElement(nv, qn("p", "cNvPicPr"))
    ET.SubElement(nv, qn("p", "nvPr"))

    blip_fill = ET.SubElement(pic, qn("p", "blipFill"))
    ET.SubElement(blip_fill, qn("a", "blip"), {qn("r", "embed"): rel_id})
    stretch = ET.SubElement(blip_fill, qn("a", "stretch"))
    ET.SubElement(stretch, qn("a", "fillRect"))

    sp_pr = ET.SubElement(pic, qn("p", "spPr"))
    xfrm = ET.SubElement(sp_pr, qn("a", "xfrm"))
    ET.SubElement(xfrm, qn("a", "off"), x=str(x), y=str(y))
    ET.SubElement(xfrm, qn("a", "ext"), cx=str(w), cy=str(h))
    geom = ET.SubElement(sp_pr, qn("a", "prstGeom"), prst="rect")
    ET.SubElement(geom, qn("a", "avLst"))


def choose_body_size(spec: SlideSpec) -> int:
    if spec.number == 1:
        return 18
    if spec.number in {29, 30}:
        return 11
    if spec.number in {3, 9, 19, 25}:
        return 15
    if spec.number in {12, 18, 22, 23, 31}:
        return 15
    non_empty = sum(1 for p in spec.paras if p.text.strip())
    if non_empty > 18:
        return 13
    if non_empty > 12:
        return 15
    return 18


def make_title_slide(spec: SlideSpec) -> ET.Element:
    root = ET.Element(qn("p", "sld"))
    cSld = ET.SubElement(root, qn("p", "cSld"))
    sp_tree = make_sp_tree()
    cSld.append(sp_tree)
    ET.SubElement(root, qn("p", "clrMapOvr"))
    root.find(qn("p", "clrMapOvr")).append(ET.Element(qn("a", "masterClrMapping")))

    add_rect(sp_tree, 2, "Background", 0, 0, SLIDE_W, SLIDE_H, "F8F7F3")
    add_rect(sp_tree, 3, "AccentBar", 0, 0, emu(0.48), SLIDE_H, HEADER_FILL)
    add_rect(sp_tree, 4, "AccentBand", emu(1.0), emu(0.9), emu(10.8), emu(0.14), ACCENT_FILL)

    title_lines: list[Para] = []
    for p in spec.paras:
        if p.text:
            title_lines.append(p)

    # Main title, subtitle, Chinese subtitle, tags, and main idea block.
    positions = [
        (emu(1.0), emu(1.1), emu(11.0), emu(0.72), 34),
        (emu(1.0), emu(1.9), emu(11.0), emu(0.42), 20),
        (emu(1.0), emu(2.35), emu(11.0), emu(0.42), 20),
        (emu(1.0), emu(3.0), emu(11.0), emu(0.32), 15),
    ]
    for idx, (x, y, w, h, size) in enumerate(positions):
        if idx >= len(title_lines):
            break
        add_textbox(
            sp_tree,
            10 + idx,
            f"TitleLine{idx+1}",
            x,
            y,
            w,
            h,
            [Para(title_lines[idx].text, bold=True if idx == 0 else False)],
            default_size=size,
            default_color=HEADER_FILL if idx == 0 else BODY_COLOR,
        )

    if len(title_lines) > 4:
        # Group the remaining lines into a highlighted takeaway box.
        box_lines = [p for p in title_lines[4:]]
        add_rect(sp_tree, 20, "TakeawayBox", emu(0.95), emu(4.05), emu(11.7), emu(1.1), "EAF3F5", "B7D8DF", radius=True)
        add_textbox(
            sp_tree,
            21,
            "TakeawayText",
            emu(1.2),
            emu(4.22),
            emu(11.1),
            emu(0.75),
            box_lines,
            default_size=18,
            default_color=HEADER_FILL,
            fit_to_box=True,
        )

    footer = ET.SubElement(sp_tree, qn("p", "sp"))
    nv = ET.SubElement(footer, qn("p", "nvSpPr"))
    ET.SubElement(nv, qn("p", "cNvPr"), id="30", name="SlideNumber")
    ET.SubElement(nv, qn("p", "cNvSpPr"), txBox="1")
    ET.SubElement(nv, qn("p", "nvPr"))
    sp_pr = ET.SubElement(footer, qn("p", "spPr"))
    ET.SubElement(sp_pr, qn("a", "noFill"))
    ln = ET.SubElement(sp_pr, qn("a", "ln"))
    ET.SubElement(ln, qn("a", "noFill"))
    tx = ET.SubElement(footer, qn("p", "txBody"))
    ET.SubElement(tx, qn("a", "bodyPr"))
    ET.SubElement(tx, qn("a", "lstStyle"))
    p = ET.SubElement(tx, qn("a", "p"))
    r = ET.SubElement(p, qn("a", "r"))
    rPr = ET.SubElement(r, qn("a", "rPr"), sz="1050", lang="en-US")
    fill = ET.SubElement(rPr, qn("a", "solidFill"))
    ET.SubElement(fill, qn("a", "srgbClr"), val=MUTED_COLOR)
    t = ET.SubElement(r, qn("a", "t"))
    t.text = f"{spec.number:02d}"
    ET.SubElement(p, qn("a", "endParaRPr"), sz="1050")
    # Position the footer via the shape transform.
    xfrm = ET.SubElement(sp_pr, qn("a", "xfrm"))
    ET.SubElement(xfrm, qn("a", "off"), x=str(emu(12.1)), y=str(emu(6.85)))
    ET.SubElement(xfrm, qn("a", "ext"), cx=str(emu(0.6)), cy=str(emu(0.24)))

    return root


def make_regular_slide(spec: SlideSpec, image_rel: Optional[str] = None) -> ET.Element:
    root = ET.Element(qn("p", "sld"))
    cSld = ET.SubElement(root, qn("p", "cSld"))
    sp_tree = make_sp_tree()
    cSld.append(sp_tree)
    clr = ET.SubElement(root, qn("p", "clrMapOvr"))
    ET.SubElement(clr, qn("a", "masterClrMapping"))

    add_rect(sp_tree, 2, "Header", 0, 0, SLIDE_W, HEADER_H, HEADER_FILL)
    add_rect(sp_tree, 3, "HeaderAccent", 0, HEADER_H - emu(0.06), SLIDE_W, emu(0.06), ACCENT_FILL)

    add_textbox(
        sp_tree,
        4,
        "SlideTitle",
        LEFT_MARGIN,
        TOP_MARGIN,
        emu(11.5),
        emu(0.46),
        [Para(spec.title, bold=True)],
        default_size=24,
        default_color=TITLE_COLOR,
    )

    body_top = BODY_TOP
    body_h = SLIDE_H - BODY_TOP - BODY_BOTTOM
    if image_rel is not None:
        # Reserve space for the image in the upper half of the slide.
        img_x = emu(0.9)
        img_y = emu(1.22)
        img_w = emu(11.5)
        img_h = emu(3.35)
        add_picture(sp_tree, 5, "SlideImage", image_rel, img_x, img_y, img_w, img_h)
        body_top = emu(4.85)
        body_h = SLIDE_H - body_top - BODY_BOTTOM

    font_size = choose_body_size(spec)
    if image_rel is not None:
        font_size = min(font_size, 15)

    body_lines = spec.paras
    body_box = ET.SubElement(sp_tree, qn("p", "sp"))
    nv = ET.SubElement(body_box, qn("p", "nvSpPr"))
    ET.SubElement(nv, qn("p", "cNvPr"), id="6", name="Body")
    ET.SubElement(nv, qn("p", "cNvSpPr"), txBox="1")
    ET.SubElement(nv, qn("p", "nvPr"))
    sp_pr = ET.SubElement(body_box, qn("p", "spPr"))
    xfrm = ET.SubElement(sp_pr, qn("a", "xfrm"))
    ET.SubElement(xfrm, qn("a", "off"), x=str(LEFT_MARGIN), y=str(body_top))
    ET.SubElement(xfrm, qn("a", "ext"), cx=str(emu(12.0) - LEFT_MARGIN - RIGHT_MARGIN), cy=str(body_h))
    geom = ET.SubElement(sp_pr, qn("a", "prstGeom"), prst="rect")
    ET.SubElement(geom, qn("a", "avLst"))
    ET.SubElement(sp_pr, qn("a", "noFill"))
    ln = ET.SubElement(sp_pr, qn("a", "ln"))
    ET.SubElement(ln, qn("a", "noFill"))
    tx = ET.SubElement(body_box, qn("p", "txBody"))
    ET.SubElement(tx, qn("a", "bodyPr"), wrap="square", anchor="t", lIns="0", rIns="0", tIns="0", bIns="0")
    ET.SubElement(tx, qn("a", "lstStyle"))

    for para in body_lines:
        p = ET.SubElement(tx, qn("a", "p"))
        if not para.text:
            ET.SubElement(p, qn("a", "endParaRPr"), sz=str(font_size * 100))
            continue
        size = para.size or font_size
        r = ET.SubElement(p, qn("a", "r"))
        rPr = ET.SubElement(r, qn("a", "rPr"), sz=str(size * 100), lang="en-US" if para.mono else "zh-TW")
        if para.bold:
            rPr.set("b", "1")
        fill = ET.SubElement(rPr, qn("a", "solidFill"))
        ET.SubElement(fill, qn("a", "srgbClr"), val=BODY_COLOR)
        if para.mono:
            ET.SubElement(rPr, qn("a", "latin"), typeface="Courier New")
            ET.SubElement(rPr, qn("a", "ea"), typeface="Courier New")
            ET.SubElement(rPr, qn("a", "cs"), typeface="Courier New")
        t = ET.SubElement(r, qn("a", "t"))
        t.text = para.text
        ET.SubElement(p, qn("a", "endParaRPr"), sz=str(size * 100))

    footer = ET.SubElement(sp_tree, qn("p", "sp"))
    nv = ET.SubElement(footer, qn("p", "nvSpPr"))
    ET.SubElement(nv, qn("p", "cNvPr"), id="30", name="SlideNumber")
    ET.SubElement(nv, qn("p", "cNvSpPr"), txBox="1")
    ET.SubElement(nv, qn("p", "nvPr"))
    sp_pr = ET.SubElement(footer, qn("p", "spPr"))
    xfrm = ET.SubElement(sp_pr, qn("a", "xfrm"))
    ET.SubElement(xfrm, qn("a", "off"), x=str(emu(12.1)), y=str(emu(6.86)))
    ET.SubElement(xfrm, qn("a", "ext"), cx=str(emu(0.55)), cy=str(emu(0.22)))
    ET.SubElement(sp_pr, qn("a", "noFill"))
    ln = ET.SubElement(sp_pr, qn("a", "ln"))
    ET.SubElement(ln, qn("a", "noFill"))
    tx = ET.SubElement(footer, qn("p", "txBody"))
    ET.SubElement(tx, qn("a", "bodyPr"))
    ET.SubElement(tx, qn("a", "lstStyle"))
    p = ET.SubElement(tx, qn("a", "p"))
    r = ET.SubElement(p, qn("a", "r"))
    rPr = ET.SubElement(r, qn("a", "rPr"), sz="900", lang="en-US")
    fill = ET.SubElement(rPr, qn("a", "solidFill"))
    ET.SubElement(fill, qn("a", "srgbClr"), val="8A93A0")
    t = ET.SubElement(r, qn("a", "t"))
    t.text = f"{spec.number:02d}"
    ET.SubElement(p, qn("a", "endParaRPr"), sz="900")

    return root


def slide_relationship_xml(image_names: list[str]) -> bytes:
    rels = ET.Element(qn("pr", "Relationships"))
    ET.SubElement(
        rels,
        qn("pr", "Relationship"),
        Id="rId1",
        Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout",
        Target="../slideLayouts/slideLayout7.xml",
    )
    for idx, img_name in enumerate(image_names, start=2):
        ET.SubElement(
            rels,
            qn("pr", "Relationship"),
            Id=f"rId{idx}",
            Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
            Target=f"../media/{img_name}",
        )
    return ET.tostring(rels, encoding="utf-8", xml_declaration=True)


def presentation_xml(slides_count: int) -> bytes:
    root = ET.fromstring(_read_zip_member(TEMPLATE_PPTX, "ppt/presentation.xml"))
    sld_id_lst = root.find(qn("p", "sldIdLst"))
    if sld_id_lst is None:
        raise RuntimeError("Missing slide id list in template presentation.xml")

    for child in list(sld_id_lst):
        sld_id_lst.remove(child)

    rel_map = {}
    # First six slides reuse the template's relationship ids rId2..rId7.
    for idx in range(1, min(slides_count, 6) + 1):
        rel_map[idx] = f"rId{idx + 1}"
    next_rel = 13
    for idx in range(7, slides_count + 1):
        rel_map[idx] = f"rId{next_rel}"
        next_rel += 1

    for idx in range(1, slides_count + 1):
        sld_id = ET.SubElement(sld_id_lst, qn("p", "sldId"))
        sld_id.set("id", str(255 + idx))
        sld_id.set(qn("r", "id"), rel_map[idx])

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def presentation_rels_xml(slides_count: int) -> bytes:
    rels = ET.fromstring(_read_zip_member(TEMPLATE_PPTX, "ppt/_rels/presentation.xml.rels"))
    # Remove any existing slide relationships; keep the core package metadata.
    for child in list(rels):
        if child.attrib.get("Type", "").endswith("/slide"):
            rels.remove(child)

    for idx in range(1, min(slides_count, 6) + 1):
        ET.SubElement(
            rels,
            qn("pr", "Relationship"),
            Id=f"rId{idx + 1}",
            Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide",
            Target=f"slides/slide{idx}.xml",
        )

    next_rel = 13
    for idx in range(7, slides_count + 1):
        ET.SubElement(
            rels,
            qn("pr", "Relationship"),
            Id=f"rId{next_rel}",
            Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide",
            Target=f"slides/slide{idx}.xml",
        )
        next_rel += 1

    return ET.tostring(rels, encoding="utf-8", xml_declaration=True)


def content_types_xml(slides_count: int) -> bytes:
    root = ET.fromstring(_read_zip_member(TEMPLATE_PPTX, "[Content_Types].xml"))
    for child in list(root):
        if child.tag.endswith("Override") and child.attrib.get("PartName", "").startswith("/ppt/slides/slide"):
            root.remove(child)

    defaults = {child.attrib.get("Extension") for child in root if child.tag.endswith("Default")}
    if "png" not in defaults:
        ET.SubElement(root, qn("ct", "Default"), Extension="png", ContentType="image/png")

    for idx in range(1, slides_count + 1):
        ET.SubElement(
            root,
            qn("ct", "Override"),
            PartName=f"/ppt/slides/slide{idx}.xml",
            ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
        )

    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def app_xml(slides_count: int) -> bytes:
    root = ET.fromstring(_read_zip_member(TEMPLATE_PPTX, "docProps/app.xml"))
    ns = {"ep": EP, "vt": VT}
    slides_el = root.find(f"{{{EP}}}Slides")
    if slides_el is not None:
        slides_el.text = str(slides_count)
    # Update the slide title count entry in HeadingPairs if present.
    heading_pairs = root.find(f"{{{EP}}}HeadingPairs")
    if heading_pairs is not None:
        ints = heading_pairs.findall(f".//{{{VT}}}i4")
        if ints:
            ints[-1].text = str(slides_count)
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def core_xml() -> bytes:
    return _read_zip_member(TEMPLATE_PPTX, "docProps/core.xml")


def _read_zip_member(path: Path, member: str) -> bytes:
    with zipfile.ZipFile(path) as zf:
        return zf.read(member)


def image_target_name(slide_number: int) -> Optional[str]:
    if slide_number not in {12, 18, 22, 23, 31}:
        return None
    return {
        12: "image101.png",
        18: "image102.png",
        22: "image103.png",
        23: "image104.png",
        31: "image105.png",
    }[slide_number]


def build() -> None:
    specs = build_specs()
    spec_map = {spec.number: spec for spec in specs}
    if len(spec_map) != 32:
        raise RuntimeError(f"Expected 32 slides, found {len(spec_map)}")

    slide_xml: dict[int, bytes] = {}
    slide_rels: dict[int, bytes] = {}
    media_files: dict[str, Path] = {}

    for idx in range(1, 33):
        spec = spec_map[idx]
        image_name = image_target_name(idx)
        if idx == 1:
            slide = make_title_slide(spec)
            slide_xml[idx] = ET.tostring(slide, encoding="utf-8", xml_declaration=True)
            slide_rels[idx] = slide_relationship_xml([])
        else:
            if image_name and spec.image is not None:
                media_files[image_name] = spec.image
            slide = make_regular_slide(spec, image_rel=f"rId2" if image_name else None)
            slide_xml[idx] = ET.tostring(slide, encoding="utf-8", xml_declaration=True)
            slide_rels[idx] = slide_relationship_xml([image_name] if image_name else [])

    with zipfile.ZipFile(TEMPLATE_PPTX, "r") as zin, zipfile.ZipFile(OUTPUT_PPTX, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "ppt/presentation.xml":
                data = presentation_xml(32)
            elif item.filename == "ppt/_rels/presentation.xml.rels":
                data = presentation_rels_xml(32)
            elif item.filename == "[Content_Types].xml":
                data = content_types_xml(32)
            elif item.filename == "docProps/app.xml":
                data = app_xml(32)
            elif item.filename == "docProps/core.xml":
                data = core_xml()
            elif item.filename == "ppt/slides/slide1.xml":
                data = slide_xml[1]
            elif item.filename == "ppt/slides/slide2.xml":
                data = slide_xml[2]
            elif item.filename == "ppt/slides/slide3.xml":
                data = slide_xml[3]
            elif item.filename == "ppt/slides/slide4.xml":
                data = slide_xml[4]
            elif item.filename == "ppt/slides/slide5.xml":
                data = slide_xml[5]
            elif item.filename == "ppt/slides/slide6.xml":
                data = slide_xml[6]
            elif item.filename == "ppt/slides/_rels/slide1.xml.rels":
                data = slide_rels[1]
            elif item.filename == "ppt/slides/_rels/slide2.xml.rels":
                data = slide_rels[2]
            elif item.filename == "ppt/slides/_rels/slide3.xml.rels":
                data = slide_rels[3]
            elif item.filename == "ppt/slides/_rels/slide4.xml.rels":
                data = slide_rels[4]
            elif item.filename == "ppt/slides/_rels/slide5.xml.rels":
                data = slide_rels[5]
            elif item.filename == "ppt/slides/_rels/slide6.xml.rels":
                data = slide_rels[6]
            zout.writestr(item, data)

        # Add slide 7..32 and their rels.
        for idx in range(7, 33):
            zout.writestr(f"ppt/slides/slide{idx}.xml", slide_xml[idx])
            zout.writestr(f"ppt/slides/_rels/slide{idx}.xml.rels", slide_rels[idx])

        # Add embedded images.
        for media_name, src in media_files.items():
            zout.writestr(f"ppt/media/{media_name}", src.read_bytes())

    print(f"Wrote {OUTPUT_PPTX}")


if __name__ == "__main__":
    build()
