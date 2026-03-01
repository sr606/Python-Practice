import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from graphviz import Source
from graphviz.backend import ExecutableNotFound


INPUT_FOLDER = "data/input"
OUTPUT_FOLDER = "data/output"


@dataclass
class Transformation:
    output_link: str
    target_col: str
    expression: str
    is_modified: bool
    issue: str = ""


@dataclass
class Constraint:
    name: str
    expression: str
    issue: str = ""


@dataclass
class JoinBlock:
    join_type: str
    table_name: str
    alias: str
    raw_join: str
    join_condition: str
    resolved_condition: str


@dataclass
class Stage:
    raw_header: str
    name: str
    stage_type: str
    block: str
    inputs: List[Tuple[str, str, str]] = field(default_factory=list)
    outputs: List[Tuple[str, str]] = field(default_factory=list)
    transformations: List[Transformation] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)
    stage_vars_defined: Dict[str, str] = field(default_factory=dict)
    stage_vars_used: List[str] = field(default_factory=list)
    output_links: List[str] = field(default_factory=list)
    alias_map: Dict[str, str] = field(default_factory=dict)
    join_blocks: List[JoinBlock] = field(default_factory=list)
    source_entities: List[Tuple[str, str]] = field(default_factory=list)
    has_explicit_join_stage: bool = False
    has_lookup: bool = False


def sanitize_id(name: str, fallback: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", name.strip())
    token = re.sub(r"_+", "_", token).strip("_")
    if not token:
        token = fallback
    if token[0].isdigit():
        token = f"n_{token}"
    return token


def dot_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def parse_stage_blocks(text: str) -> List[Tuple[str, str]]:
    stages: List[Tuple[str, str]] = []

    header_matches = list(
        re.finditer(
            r"(?ms)^//\s*---\s*\[(?P<header>.*?)\]\s*\[Lines.*?\]\s*---\s*(?P<body>.*?)(?=^//\s*---\s*\[|\Z)",
            text,
        )
    )
    for m in header_matches:
        stages.append((m.group("header"), m.group("body").strip()))

    if stages:
        return stages

    xml_matches = list(
        re.finditer(
            r"(?is)<Stage\b[^>]*>(?P<body>.*?)</Stage>",
            text,
        )
    )
    for idx, m in enumerate(xml_matches, 1):
        stages.append((f"Stage_{idx}", m.group("body").strip()))

    return stages


def parse_inputs(block: str) -> List[Tuple[str, str, str]]:
    results = []
    for m in re.finditer(
        r"Input:\s*.*?dataset_(\d+)\s*\(([^)]+)\)(?:\s*\(Link:\s*([^)]+)\))?",
        block,
        re.IGNORECASE,
    ):
        dataset_id = f"dataset_{m.group(1)}"
        dataset_name = m.group(2).strip()
        link_name = (m.group(3) or "").strip()
        results.append((dataset_id, dataset_name, link_name))
    return results


def parse_outputs(block: str) -> List[Tuple[str, str]]:
    results = []
    for m in re.finditer(
        r"Output:\s*.*?dataset_(\d+)\s*\(([^)]+)\)",
        block,
        re.IGNORECASE,
    ):
        dataset_id = f"dataset_{m.group(1)}"
        dataset_name = m.group(2).strip()
        results.append((dataset_id, dataset_name))
    return results


def parse_stage_variables(block: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    match = re.search(
        r"(?is)Stage Variables:\s*(.*?)(?=^\s*(Constraint\s*\(|StageType:|Transformations:|Output:|Input:|Link File|//\s*---|\Z))",
        block,
        re.MULTILINE,
    )
    if not match:
        return out
    for line in match.group(1).splitlines():
        m = re.search(r"StageVar\s+([A-Za-z_]\w*)\s*=\s*(.+)", line.strip(), re.IGNORECASE)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def parse_constraints(block: str) -> List[Constraint]:
    out: List[Constraint] = []
    for m in re.finditer(r"Constraint\s*\(([^)]+)\)\s*:\s*(.+)", block, re.IGNORECASE):
        out.append(Constraint(name=m.group(1).strip(), expression=m.group(2).strip()))
    return out


def is_direct_copy(expr: str) -> bool:
    expr = expr.strip()
    return bool(re.fullmatch(r"[A-Za-z_]\w*\.[A-Za-z_]\w*", expr))


def parse_transformations(block: str) -> Tuple[List[Transformation], List[str]]:
    out: List[Transformation] = []
    output_links: List[str] = []
    match = re.search(
        r"(?is)Transformations:\s*(.*?)(?=^\s*(Output:\s*|//\s*---|\Z))",
        block,
        re.MULTILINE,
    )
    if not match:
        return out, output_links

    current_output = "Unknown"
    lines = [ln.rstrip() for ln in match.group(1).splitlines() if ln.strip()]
    for line in lines:
        marker = re.match(r"\s*--\s*Output:\s*(.*?)\s*--\s*$", line, re.IGNORECASE)
        if marker:
            current_output = marker.group(1).strip() or "Unknown"
            if current_output not in output_links:
                output_links.append(current_output)
            continue
        m = re.match(r"\s*([A-Za-z_]\w*)\s*=\s*(.+)$", line)
        if m:
            target = m.group(1).strip()
            expr = m.group(2).strip()
            out.append(
                Transformation(
                    output_link=current_output,
                    target_col=target,
                    expression=expr,
                    is_modified=(not is_direct_copy(expr)),
                )
            )
        else:
            out.append(
                Transformation(
                    output_link=current_output,
                    target_col="Unknown",
                    expression=line.strip(),
                    is_modified=False,
                    issue="Unresolved Mapping",
                )
            )
    return out, output_links


def extract_from_block(sql_text: str) -> str:
    m = re.search(
        r"(?is)\bfrom\b\s*(.*?)(?=\b(?:left|right|full|inner|outer|cross)?\s*join\b|\bwhere\b|\bgroup\s+by\b|\border\s+by\b|;|\Z)",
        sql_text,
    )
    return m.group(1).strip() if m else ""


def build_alias_map(sql_text: str) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    from_block = extract_from_block(sql_text)
    if from_block:
        m = re.match(r"(?is)([A-Za-z0-9_.$#]+)\s+([A-Za-z_]\w*)\b", from_block.strip())
        if m:
            alias_map[m.group(2).lower()] = m.group(1)

    for jb in parse_structured_joins(sql_text, {}):
        if jb.alias:
            alias_map[jb.alias.lower()] = jb.table_name
    return alias_map


def resolve_aliases(expr: str, alias_map: Dict[str, str]) -> str:
    resolved = expr
    for alias, table in sorted(alias_map.items(), key=lambda x: len(x[0]), reverse=True):
        if table.lower() in {"subquery", "unknown", "rawjoin"}:
            continue
        resolved = re.sub(
            rf"\b{re.escape(alias)}\.",
            f"{table}.",
            resolved,
            flags=re.IGNORECASE,
        )
    return resolved


def _find_top_level_keyword(sql: str, keyword_regex: str, start_idx: int = 0) -> int:
    depth = 0
    idx = start_idx
    pat = re.compile(keyword_regex, re.IGNORECASE)
    while idx < len(sql):
        ch = sql[idx]
        if ch == "(":
            depth += 1
            idx += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            idx += 1
            continue
        if depth == 0:
            m = pat.match(sql, idx)
            if m:
                return idx
        idx += 1
    return -1


def _extract_join_segments(sql_text: str) -> List[str]:
    segments: List[str] = []
    from_idx = _find_top_level_keyword(sql_text, r"\bfrom\b")
    if from_idx == -1:
        return segments

    cursor = from_idx
    join_pat = r"\b(?:left|right|full|inner|cross)(?:\s+outer)?\s+join\b|\bjoin\b"
    stop_pat = r"\bwhere\b|\bgroup\s+by\b|\border\s+by\b|;"

    while True:
        jidx = _find_top_level_keyword(sql_text, join_pat, cursor)
        if jidx == -1:
            break
        next_join = _find_top_level_keyword(sql_text, join_pat, jidx + 1)
        next_stop = _find_top_level_keyword(sql_text, stop_pat, jidx + 1)
        end_candidates = [x for x in [next_join, next_stop] if x != -1]
        end = min(end_candidates) if end_candidates else len(sql_text)
        segment = sql_text[jidx:end].strip()
        if segment:
            segments.append(re.sub(r"\s+", " ", segment))
        cursor = end

    return segments


def _parse_join_target(rest: str) -> Tuple[str, str, str]:
    rest = rest.strip()
    rest = re.sub(r"(?is)^/\*.*?\*/\s*", "", rest).strip()
    rest = re.sub(r"(?im)^--.*?$", "", rest).strip()
    if not rest:
        return "RawJoin", "", ""

    if rest.startswith("("):
        depth = 0
        end_idx = -1
        for i, ch in enumerate(rest):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    end_idx = i
                    break
        if end_idx == -1:
            return "RawJoin", "", rest
        after = rest[end_idx + 1 :].lstrip()
        alias_m = re.match(r"([A-Za-z_]\w*)\b(.*)$", after, re.IGNORECASE)
        if alias_m:
            alias = alias_m.group(1)
            return "Subquery", alias, alias_m.group(2).lstrip()
        return "Subquery", "", after

    token_m = re.match(r"([A-Za-z0-9_.$#]+)\b(.*)$", rest, re.IGNORECASE)
    if not token_m:
        return "RawJoin", "", rest
    table_name = token_m.group(1)
    after = token_m.group(2).lstrip()
    alias_m = re.match(r"([A-Za-z_]\w*)\b(.*)$", after, re.IGNORECASE)
    if alias_m and alias_m.group(1).lower() != "on":
        return table_name, alias_m.group(1), alias_m.group(2).lstrip()
    return table_name, "", after


def parse_structured_joins(sql_text: str, alias_map: Dict[str, str]) -> List[JoinBlock]:
    joins: List[JoinBlock] = []
    join_segments = _extract_join_segments(sql_text)
    for raw_segment in join_segments:
        jm = re.match(
            r"(?is)(?P<jtype>(?:left|right|full|inner|cross)(?:\s+outer)?\s+join|join)\s+(?P<rest>.*)$",
            raw_segment,
        )
        if not jm:
            continue
        join_type = jm.group("jtype").upper()
        table_name, parsed_alias, tail = _parse_join_target(jm.group("rest"))
        alias = parsed_alias or table_name

        on_m = re.search(r"(?is)\bon\b\s*(.+)$", tail)
        join_condition = re.sub(r"\s+", " ", on_m.group(1)).strip() if on_m else raw_segment
        if table_name == "RawJoin":
            join_condition = raw_segment
            alias = "RawJoin"
        resolved_condition = resolve_aliases(join_condition, alias_map)
        joins.append(
            JoinBlock(
                join_type=join_type,
                table_name=table_name,
                alias=alias,
                raw_join=raw_segment,
                join_condition=join_condition,
                resolved_condition=resolved_condition,
            )
        )
    return joins


def parse_source_entities(sql_text: str, join_blocks: List[JoinBlock]) -> List[Tuple[str, str]]:
    entities: List[Tuple[str, str]] = []
    seen = set()

    # Prefer physical schema.table names explicitly present in SQL FROM/JOIN clauses.
    for m in re.finditer(r"(?is)\b(from|join)\s+([A-Za-z0-9_.$#]+)", sql_text):
        relation_kw = m.group(1).upper()
        token = m.group(2).strip("()")
        if token.lower() == "select":
            continue
        if "." not in token:
            continue
        key = token.lower()
        if key not in seen:
            rel = "Source" if relation_kw == "FROM" else "JOIN"
            entities.append((token, rel))
            seen.add(key)

    # If physical tables were found, do not add alias-only fallback nodes.
    if entities:
        return entities

    from_block = extract_from_block(sql_text)
    if from_block and not entities:
        table_name, alias, _ = _parse_join_target(from_block)
        display = alias if table_name in {"Subquery", "RawJoin"} and alias else table_name
        display = display if display not in {"Subquery", "RawJoin"} else "SourceBlock"
        key = display.lower()
        if key not in seen:
            entities.append((display, "Source"))
            seen.add(key)

    for jb in join_blocks:
        display = jb.table_name
        if display in {"Subquery", "RawJoin"}:
            display = jb.alias if jb.alias and jb.alias not in {"Subquery", "RawJoin"} else "JoinBlock"
        rel = "LEFT JOIN" if "LEFT" in jb.join_type else "JOIN"
        key = display.lower()
        if key not in seen:
            entities.append((display, rel))
            seen.add(key)

    return entities


def parse_stages(text: str) -> List[Stage]:
    parsed: List[Stage] = []
    for header, block in parse_stage_blocks(text):
        name = header.split(":", 1)[1].strip() if ":" in header else header.strip()
        stage_type_match = re.search(r"StageType:\s*([^\s]+)", block)
        stage_type = stage_type_match.group(1).strip() if stage_type_match else "Unknown"
        sql_match = re.search(
            r"(?is)SQL:\s*(.*?)(?=^\s*(StageType:|Output:|Input:|Transformations:|Constraint\s*\(|//\s*---|\Z))",
            block,
            re.MULTILINE,
        )
        sql_text = sql_match.group(1).strip() if sql_match else ""

        transformations, output_links = parse_transformations(block)
        alias_map = build_alias_map(sql_text)
        join_blocks = parse_structured_joins(sql_text, alias_map)

        stage = Stage(
            raw_header=header,
            name=name,
            stage_type=stage_type,
            block=block,
            inputs=parse_inputs(block),
            outputs=parse_outputs(block),
            transformations=transformations,
            constraints=parse_constraints(block),
            stage_vars_defined=parse_stage_variables(block),
            output_links=output_links,
            alias_map=alias_map,
            join_blocks=join_blocks,
            source_entities=parse_source_entities(sql_text, join_blocks),
            has_explicit_join_stage=("join" in stage_type.lower()),
            has_lookup=("lookup" in stage_type.lower() and re.search(r"\breference\b", block, re.IGNORECASE) is not None),
        )
        parsed.append(stage)
    return parsed


def extract_logic_notes(stage: Stage) -> List[str]:
    notes: List[str] = []
    for t in stage.transformations:
        expr_l = t.expression.lower()
        if "vehicle_sk=-99" in expr_l and "then" in expr_l:
            note = "Vehicle SK fallback if source SK = -99"
            if note not in notes:
                notes.append(note)
        if "supp_sk=-99" in expr_l and "'unknown'" in expr_l:
            note = "SUPP_SK -99 mapped to Unknown"
            if note not in notes:
                notes.append(note)
        if "supp_sk=-97" in expr_l and ("duplicates" in expr_l or "'duplicates'" in expr_l):
            note = "SUPP_SK -97 mapped to Duplicates"
            if note not in notes:
                notes.append(note)
    for c in stage.constraints:
        notes.append(f"Constraint: {c.expression}")
    return notes


def classify_stage_role(stage: Stage) -> str:
    if stage.has_lookup:
        return "lookup"
    if not stage.inputs:
        return "source"
    if not stage.outputs:
        return "target"
    return "processing"


def detect_undefined_references(stage: Stage) -> None:
    defined_links = {lnk for _, _, lnk in stage.inputs if lnk}
    defined_vars = set(stage.stage_vars_defined.keys())
    used_stage_vars = set()

    for tr in stage.transformations:
        for token in re.findall(r"\b([A-Za-z_]\w*)\.[A-Za-z_]\w*\b", tr.expression):
            if token not in defined_links:
                if tr.issue:
                    tr.issue = f"{tr.issue}; Undefined Reference"
                else:
                    tr.issue = "Undefined Reference"
        for sv in re.findall(r"\bsv[A-Za-z_]\w*\b", tr.expression):
            used_stage_vars.add(sv)

    for c in stage.constraints:
        for sv in re.findall(r"\bsv[A-Za-z_]\w*\b", c.expression):
            used_stage_vars.add(sv)

    stage.stage_vars_used = sorted(used_stage_vars)

    for sv in stage.stage_vars_used:
        if sv not in defined_vars:
            for c in stage.constraints:
                if sv in c.expression:
                    c.issue = "Undefined Variable"


def longest_chain(edges: List[Tuple[str, str]]) -> int:
    children: Dict[str, List[str]] = {}
    for src, dst in edges:
        children.setdefault(src, []).append(dst)
    memo: Dict[str, int] = {}

    def depth(node: str, trail: set) -> int:
        if node in memo:
            return memo[node]
        if node in trail:
            return 0
        trail.add(node)
        best = 1
        for nxt in children.get(node, []):
            best = max(best, 1 + depth(nxt, trail))
        trail.remove(node)
        memo[node] = best
        return best

    return max((depth(n, set()) for n in children), default=0)


def build_global_links(stages: List[Stage]) -> List[Tuple[str, str, str]]:
    producers: Dict[str, str] = {}
    for st in stages:
        for _, ds_name in st.outputs:
            producers[ds_name] = st.name

    edges: List[Tuple[str, str, str]] = []
    for st in stages:
        for _, ds_name, _ in st.inputs:
            src = producers.get(ds_name, "Unknown")
            edges.append((src, st.name, ds_name))
    return edges


def compute_complexity(stages: List[Stage]) -> Tuple[int, int]:
    stage_count = len(stages)
    join_count = sum(len(s.join_blocks) + (1 if s.has_explicit_join_stage else 0) for s in stages)
    transformation_count = sum(1 for s in stages for t in s.transformations if t.is_modified)
    constraint_count = sum(len(s.constraints) for s in stages)
    transformed_columns = sum(1 for s in stages for t in s.transformations if t.is_modified)
    return stage_count + join_count + transformation_count + constraint_count, transformed_columns


def node_shape(stage: Stage) -> str:
    role = classify_stage_role(stage)
    stype = stage.stage_type.lower()
    if "seqfile" in stype:
        return "folder"
    if role == "source" or role == "target":
        return "cylinder"
    if role == "lookup":
        return "component"
    if "join" in stype:
        return "box"
    return "box"


def get_constraint_routes(stage: Stage, constraint_name: str) -> Tuple[str, str]:
    # Prefer transformation output link names for route targets.
    explicit_outputs = [o for o in stage.output_links if o and o != "Unknown"]
    if explicit_outputs:
        yes_target = constraint_name if constraint_name in explicit_outputs else explicit_outputs[0]
        no_candidates = [o for o in explicit_outputs if o != yes_target]
        no_target = no_candidates[0] if no_candidates else "Unknown"
        return yes_target, no_target

    # Fallback to stage dataset outputs if no transformation link labels are present.
    ds_outputs = [name for _, name in stage.outputs]
    if ds_outputs:
        yes_target = ds_outputs[0]
        no_target = ds_outputs[1] if len(ds_outputs) > 1 else "Unknown"
        return yes_target, no_target
    return "Unknown", "Unknown"


def build_high_level_dot(graph_name: str, stages: List[Stage], rankdir: str, links: List[Tuple[str, str, str]]) -> str:
    stage_by_name = {s.name: s for s in stages}
    downstream: Dict[str, List[Tuple[str, str]]] = {}
    for src, dst, lbl in links:
        downstream.setdefault(src, []).append((dst, lbl))

    transformer_stage = next(
        (s for s in stages if "transformer" in s.stage_type.lower() or s.name.lower().startswith("tfm_")),
        next((s for s in stages if classify_stage_role(s) == "processing"), stages[0]),
    )
    source_stage = next(
        (s for s in stages if classify_stage_role(s) == "source" and s.name != "HF_SMR_VEHICLE_DIM"),
        next((s for s in stages if classify_stage_role(s) == "source"), transformer_stage),
    )

    input_link_to_stage: Dict[str, str] = {}
    for s in stages:
        for _, _, link_name in s.inputs:
            if link_name:
                input_link_to_stage[link_name] = s.name

    lookup_stage_names = set()
    source_to_transform_link = "Unknown"
    lookup_links: List[Tuple[str, str]] = []
    for _, _, link_name in transformer_stage.inputs:
        producer_name = None
        for src, dst, lbl in links:
            if dst == transformer_stage.name and lbl:
                producer_name = src
        # Map lookup style links first.
        if link_name and link_name.lower().startswith("lk"):
            if producer_name and producer_name in stage_by_name:
                lookup_stage_names.add(producer_name)
                lookup_links.append((producer_name, link_name))
            continue
        if link_name and source_to_transform_link == "Unknown":
            source_to_transform_link = link_name

    for s in stages:
        if s.has_lookup:
            lookup_stage_names.add(s.name)

    lookup_stages = [stage_by_name[n] for n in lookup_stage_names if n in stage_by_name and n != transformer_stage.name]

    yes_stage_name = None
    no_stage_name = None
    constraint_expr = ""
    if transformer_stage.constraints:
        c = transformer_stage.constraints[0]
        constraint_expr = c.expression
        yes_out, no_out = get_constraint_routes(transformer_stage, c.name)
        yes_stage_name = input_link_to_stage.get(yes_out)
        no_stage_name = input_link_to_stage.get(no_out)

    transformer_children = [dst for dst, _ in downstream.get(transformer_stage.name, [])]
    if not yes_stage_name and transformer_children:
        yes_stage_name = transformer_children[0]
    if not no_stage_name and len(transformer_children) > 1:
        no_stage_name = transformer_children[1]

    target_chain: List[str] = []
    if yes_stage_name:
        target_chain.append(yes_stage_name)
    if no_stage_name and no_stage_name not in target_chain:
        target_chain.append(no_stage_name)
    # Include full downstream target flow (e.g., HF_FACT -> Ora_StgVehicleOffRoad).
    q = list(target_chain)
    seen = set(target_chain)
    while q:
        cur = q.pop(0)
        for nxt, _ in downstream.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
                target_chain.append(nxt)

    success_stage_name = yes_stage_name
    exception_stage_name = no_stage_name
    if yes_stage_name and no_stage_name:
        yes_stage = stage_by_name.get(yes_stage_name)
        no_stage = stage_by_name.get(no_stage_name)
        yes_is_exception = yes_stage and ("seqfile" in yes_stage.stage_type.lower() or "exception" in yes_stage.name.lower())
        no_is_exception = no_stage and ("seqfile" in no_stage.stage_type.lower() or "exception" in no_stage.name.lower())
        if yes_is_exception and not no_is_exception:
            success_stage_name = no_stage_name
            exception_stage_name = yes_stage_name
        elif no_is_exception and not yes_is_exception:
            success_stage_name = yes_stage_name
            exception_stage_name = no_stage_name

    source_note = f"Enriched via {max(len(source_stage.source_entities)-1, 0)} joins"
    source_label = f"{source_stage.name}\\n({source_note})"

    logic_notes = extract_logic_notes(transformer_stage)
    transformer_lines = [transformer_stage.name]
    for n in logic_notes[:3]:
        transformer_lines.append(f"- {n}")

    transformer_label = "\\n".join(transformer_lines)
    transformer_label = "\\n".join(transformer_lines)
    lines = [
        f'digraph {sanitize_id(graph_name, "lineage_graph")} {{',
        "rankdir=LR;",
        "nodesep=0.5;",
        "ranksep=1.0;",
        'fontsize=10;',
        'fontname="Arial";',
        f'label="{dot_escape(graph_name)} Data Lineage Diagram";',
        'labelloc="t";',
        'node [shape=box, style="filled", fontname="Arial", fontsize=10];',
        'edge [fontname="Arial", fontsize=8];',
        "",
        "subgraph cluster_0 {",
        'label="Source Layer";',
        "style=dashed;",
        "color=grey;",
        f'Source_DB [label="{dot_escape(source_label)}", fillcolor="#FFD1DC"];',
        "}",
        "",
        "subgraph cluster_1 {",
        'label="Reference Data";',
        "style=dashed;",
        "color=blue;",
    ]
    for i, lk in enumerate(lookup_stages, 1):
        lines.append(
            f'Lookup_{i} [label="{dot_escape(lk.name)}", fillcolor="#E1F5FE", shape=cylinder];'
        )
    lines.extend(
        [
            "}",
            "",
            "subgraph cluster_2 {",
            'label="Business Logic Stage";',
            "style=solid;",
            f'Transformer [label="{dot_escape(transformer_label)}", fillcolor="#FFF9C4"];',
            "}",
            "",
            "subgraph cluster_3 {",
            'label="Target Layer";',
            "style=dashed;",
            "color=green;",
        ]
    )

    target_node_by_stage: Dict[str, str] = {}
    t_idx = 1
    for name in target_chain:
        stage = stage_by_name.get(name)
        if not stage:
            continue
        nid = f"Target_{t_idx}"
        t_idx += 1
        target_node_by_stage[name] = nid
        shape = "note" if ("seqfile" in stage.stage_type.lower() or "exception" in stage.name.lower()) else "cylinder"
        fill = "#FFCCBC" if shape == "note" else "#C8E6C9"
        lines.append(f'{nid} [label="{dot_escape(stage.name)}", fillcolor="{fill}", shape={shape}];')
    lines.append("}")
    lines.append("")

    lines.append(f'Source_DB -> Transformer [label="{dot_escape(source_to_transform_link)}"];')
    for i, (_, link_name) in enumerate(lookup_links, 1):
        if i <= len(lookup_stages):
            cond = ""
            for t in transformer_stage.transformations:
                if "vehicle_sk=-99" in t.expression.lower():
                    cond = "\\n(If SK = -99)"
                    break
            lines.append(
                f'Lookup_{i} -> Transformer [label="{dot_escape(link_name + cond)}", style=dotted];'
            )

    if success_stage_name and success_stage_name in target_node_by_stage:
        label = "Success Path"
        if constraint_expr:
            label = f"Success Path\\n({constraint_expr} false)"
        lines.append(
            f'Transformer -> {target_node_by_stage[success_stage_name]} [label="{dot_escape(label)}", color=blue];'
        )
    if exception_stage_name and exception_stage_name in target_node_by_stage:
        label = "Exception Path"
        if constraint_expr:
            label = f"Exception Path\\n({constraint_expr} true)"
        lines.append(
            f'Transformer -> {target_node_by_stage[exception_stage_name]} [label="{dot_escape(label)}", color=red, fontcolor=red];'
        )

    for src_name, src_node in target_node_by_stage.items():
        for dst_name, _ in downstream.get(src_name, []):
            if dst_name in target_node_by_stage:
                lines.append(f"{src_node} -> {target_node_by_stage[dst_name]};")

    # Footnote block in the left corner.
    if logic_notes:
        note_text = "Logic Fixes:\\n" + "\\n".join([f"{i+1}. {n}" for i, n in enumerate(logic_notes[:4])])
        lines.append(f'Note1 [shape=plaintext, label="{dot_escape(note_text)}", fontsize=8];')
        lines.append("Note1 -> Source_DB [style=invis];")

    lines.append("}")
    return "\n".join(lines)


def build_detailed_dot(
    graph_name: str,
    stages: List[Stage],
    rankdir: str,
    links: List[Tuple[str, str, str]],
    transformed_columns: int,
) -> str:
    stage_by_name = {s.name: s for s in stages}
    downstream: Dict[str, List[Tuple[str, str]]] = {}
    for src, dst, lbl in links:
        downstream.setdefault(src, []).append((dst, lbl))

    transformer_stage = next(
        (s for s in stages if "transformer" in s.stage_type.lower() or s.name.lower().startswith("tfm_")),
        next((s for s in stages if classify_stage_role(s) == "processing"), stages[0]),
    )
    source_stage = next(
        (s for s in stages if classify_stage_role(s) == "source" and s.name != "HF_SMR_VEHICLE_DIM"),
        next((s for s in stages if classify_stage_role(s) == "source"), transformer_stage),
    )

    input_link_to_stage: Dict[str, str] = {}
    for s in stages:
        for _, _, link_name in s.inputs:
            if link_name:
                input_link_to_stage[link_name] = s.name

    lookup_stage_names = set()
    source_to_transform_link = "Unknown"
    lookup_links: List[Tuple[str, str]] = []
    for _, _, link_name in transformer_stage.inputs:
        producer_name = None
        for src, dst, lbl in links:
            if dst == transformer_stage.name and lbl:
                producer_name = src
        if link_name and link_name.lower().startswith("lk"):
            if producer_name and producer_name in stage_by_name:
                lookup_stage_names.add(producer_name)
                lookup_links.append((producer_name, link_name))
            continue
        if link_name and source_to_transform_link == "Unknown":
            source_to_transform_link = link_name

    for s in stages:
        if s.has_lookup:
            lookup_stage_names.add(s.name)
    lookup_stages = [stage_by_name[n] for n in lookup_stage_names if n in stage_by_name and n != transformer_stage.name]

    yes_stage_name = None
    no_stage_name = None
    constraint_expr = ""
    if transformer_stage.constraints:
        c = transformer_stage.constraints[0]
        constraint_expr = c.expression
        yes_out, no_out = get_constraint_routes(transformer_stage, c.name)
        yes_stage_name = input_link_to_stage.get(yes_out)
        no_stage_name = input_link_to_stage.get(no_out)

    transformer_children = [dst for dst, _ in downstream.get(transformer_stage.name, [])]
    if not yes_stage_name and transformer_children:
        yes_stage_name = transformer_children[0]
    if not no_stage_name and len(transformer_children) > 1:
        no_stage_name = transformer_children[1]

    success_stage_name = yes_stage_name
    exception_stage_name = no_stage_name
    if yes_stage_name and no_stage_name:
        yes_stage = stage_by_name.get(yes_stage_name)
        no_stage = stage_by_name.get(no_stage_name)
        yes_is_exception = yes_stage and ("seqfile" in yes_stage.stage_type.lower() or "exception" in yes_stage.name.lower())
        no_is_exception = no_stage and ("seqfile" in no_stage.stage_type.lower() or "exception" in no_stage.name.lower())
        if yes_is_exception and not no_is_exception:
            success_stage_name = no_stage_name
            exception_stage_name = yes_stage_name
        elif no_is_exception and not yes_is_exception:
            success_stage_name = yes_stage_name
            exception_stage_name = no_stage_name

    source_note = f"Enriched via {max(len(source_stage.source_entities)-1, 0)} joins"
    source_label = f"{source_stage.name}\\n({source_note})"
    logic_notes = extract_logic_notes(transformer_stage)
    transformer_lines = [transformer_stage.name] + [f"- {n}" for n in logic_notes[:3]]
    transformer_label = "\\n".join(transformer_lines)

    lines = [
        f'digraph {sanitize_id(graph_name + "_detailed", "lineage_detailed")} {{',
        "rankdir=LR;",
        "nodesep=0.5;",
        "ranksep=1.0;",
        'fontsize=10;',
        'fontname="Arial";',
        f'label="{dot_escape(graph_name)} Detailed Lineage Diagram";',
        'labelloc="t";',
        'node [shape=box, style="filled", fontname="Arial", fontsize=10];',
        'edge [fontname="Arial", fontsize=8];',
        "",
        "subgraph cluster_0 {",
        'label="Source Layer";',
        "style=dashed;",
        "color=grey;",
        f'Source_DB [label="{dot_escape(source_label)}", fillcolor="#FFD1DC"];',
    ]
    for i, (entity, rel) in enumerate(source_stage.source_entities, 1):
        src_ent_id = f"SourceEnt_{i}"
        lines.append(f'{src_ent_id} [label="{dot_escape(entity)}", shape=cylinder, fillcolor="#F8D7DA"];')
        lines.append(f'{src_ent_id} -> Source_DB [label="{dot_escape(rel)}"];')
    lines.extend(
        [
            "}",
            "",
            "subgraph cluster_1 {",
            'label="Reference Data";',
            "style=dashed;",
            "color=blue;",
        ]
    )
    for i, lk in enumerate(lookup_stages, 1):
        lines.append(f'Lookup_{i} [label="{dot_escape(lk.name)}", fillcolor="#E1F5FE", shape=cylinder];')
    lines.extend(
        [
            "}",
            "",
            "subgraph cluster_2 {",
            'label="Business Logic Stage";',
            "style=solid;",
            "color=black;",
            f'Transformer [label="{dot_escape(transformer_label)}", fillcolor="#FFF9C4"];',
        ]
    )

    # Join blocks
    for idx, jb in enumerate(source_stage.join_blocks, 1):
        table_label = jb.table_name
        alias_label = jb.alias
        if table_label in {"Subquery", "RawJoin"}:
            table_label = "Subquery" if table_label == "Subquery" else "Join Block"
        if alias_label in {"Subquery", "RawJoin", "Unknown"}:
            alias_label = ""
        join_label = f"{jb.join_type} {table_label}".strip()
        if alias_label:
            join_label = f"{join_label} as {alias_label}"
        jnid = f"Join_{idx}"
        lines.append(f'{jnid} [label="{dot_escape(join_label)}", fillcolor="#FFE6CC"];')

    # Stage vars
    for idx, sv in enumerate(transformer_stage.stage_vars_used, 1):
        vid = f"SV_{idx}"
        if sv in transformer_stage.stage_vars_defined:
            lines.append(f'{vid} [label="{dot_escape(sv)}", shape=ellipse, fillcolor="#E8DAEF"];')
        else:
            lines.append(f'{vid} [label="Undefined Variable", shape=ellipse, fillcolor="#F5B7B1"];')

    # Constraints
    for cidx, c in enumerate(transformer_stage.constraints, 1):
        cnid = f"Constraint_{cidx}"
        expr = c.expression if len(c.expression) <= 140 else c.expression[:137] + "..."
        lines.append(f'{cnid} [label="{dot_escape(expr)}", shape=diamond, fillcolor="#FADBD8"];')

    # Mappings
    mapped_count = sum(1 for t in transformer_stage.transformations if t.target_col != "Unknown")
    modified_count = sum(1 for t in transformer_stage.transformations if t.is_modified)
    if mapped_count > 15:
        lines.append(
            f'MapSummary [label="{dot_escape(transformer_stage.name)} ({mapped_count} columns mapped)", fillcolor="#FCF3CF"];'
        )
        unique_expr: Dict[str, List[str]] = {}
        for t in transformer_stage.transformations:
            if t.is_modified:
                unique_expr.setdefault(t.expression, []).append(t.target_col)
        for eidx, (expr, cols) in enumerate(unique_expr.items(), 1):
            enid = f"Expr_{eidx}"
            edge_label = expr if len(expr) <= 140 else (expr[:137] + "...")
            lines.append(f'{enid} [label="Applied to {len(cols)} columns", fillcolor="#FEF9E7"];')
            lines.append(f'MapSummary -> {enid} [label="Transformation: {dot_escape(edge_label)}"];')
        direct_copy_count = mapped_count - modified_count
        if direct_copy_count > 0:
            lines.append(f'DirectCopy [label="Direct copies: {direct_copy_count}", fillcolor="#FEF9E7"];')
            lines.append('MapSummary -> DirectCopy [label="Transformation: Direct copy mappings"];')
    elif transformed_columns > 25 and modified_count > 0:
        lines.append(
            f'MapSummary [label="Multiple column transformations ({modified_count})", fillcolor="#FCF3CF"];'
        )
    else:
        unique_expr: Dict[str, List[str]] = {}
        for t in transformer_stage.transformations:
            if t.is_modified:
                unique_expr.setdefault(t.expression, []).append(t.target_col)
        for eidx, (expr, cols) in enumerate(unique_expr.items(), 1):
            mnid = f"Map_{eidx}"
            edge_label = expr if len(expr) <= 140 else (expr[:137] + "...")
            lines.append(f'{mnid} [label="Applied to {len(cols)} columns", fillcolor="#FCF3CF"];')
            lines.append(f'Transformer -> {mnid} [label="Transformation: {dot_escape(edge_label)}"];')

    lines.extend(
        [
            "}",
            "",
            "subgraph cluster_3 {",
            'label="Target Layer";',
            "style=dashed;",
            "color=green;",
        ]
    )

    target_nodes: Dict[str, str] = {}
    target_order: List[str] = []
    for n in [success_stage_name, exception_stage_name]:
        if n and n not in target_order:
            target_order.append(n)
    q = list(target_order)
    seen = set(target_order)
    while q:
        cur = q.pop(0)
        for nxt, _ in downstream.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
                target_order.append(nxt)

    t_idx = 1
    for name in target_order:
        if not name or name in target_nodes:
            continue
        stage = stage_by_name.get(name)
        if not stage:
            continue
        nid = f"Target_{t_idx}"
        t_idx += 1
        target_nodes[name] = nid
        is_exception = "seqfile" in stage.stage_type.lower() or "exception" in stage.name.lower()
        shape = "note" if is_exception else "cylinder"
        fill = "#FFCCBC" if is_exception else "#C8E6C9"
        lines.append(f'{nid} [label="{dot_escape(stage.name)}", fillcolor="{fill}", shape={shape}];')
    lines.append("}")
    lines.append("")

    # Core flow edges
    lines.append(f'Source_DB -> Transformer [label="{dot_escape(source_to_transform_link)}"];')
    for i, (_, link_name) in enumerate(lookup_links, 1):
        if i <= len(lookup_stages):
            cond = ""
            for t in transformer_stage.transformations:
                if "vehicle_sk=-99" in t.expression.lower():
                    cond = "\\n(If SK = -99)"
                    break
            lines.append(f'Lookup_{i} -> Transformer [style=dashed, label="Lookup: {dot_escape(link_name + cond)}"];')

    for idx, jb in enumerate(source_stage.join_blocks, 1):
        cond = jb.join_condition if jb.table_name in {"Subquery", "RawJoin"} else (jb.resolved_condition or jb.raw_join)
        lines.append(f'Source_DB -> Join_{idx} [label="Join: {dot_escape(cond)}"];')

    for idx, sv in enumerate(transformer_stage.stage_vars_used, 1):
        lines.append(f'SV_{idx} -> Transformer [label="Transformation"];')
        for cidx, c in enumerate(transformer_stage.constraints, 1):
            if re.search(rf"\b{re.escape(sv)}\b", c.expression):
                lines.append(f'SV_{idx} -> Constraint_{cidx} [label="Constraint"];')

    if mapped_count > 15 or (transformed_columns > 25 and modified_count > 0):
        lines.append('Transformer -> MapSummary [label="Transformation"];')

    for cidx, c in enumerate(transformer_stage.constraints, 1):
        lines.append(f'Transformer -> Constraint_{cidx} [label="Constraint"];')
        yes_target, no_target = get_constraint_routes(transformer_stage, c.name)
        yes_stage = input_link_to_stage.get(yes_target, success_stage_name)
        no_stage = input_link_to_stage.get(no_target, exception_stage_name)
        if yes_stage in target_nodes:
            lines.append(f'Constraint_{cidx} -> {target_nodes[yes_stage]} [label="Yes"];')
        if no_stage in target_nodes:
            lines.append(f'Constraint_{cidx} -> {target_nodes[no_stage]} [label="No"];')

    # Target chaining (e.g., hash -> oracle)
    for src_name, src_node in target_nodes.items():
        for dst_name, _ in downstream.get(src_name, []):
            if dst_name in target_nodes:
                lines.append(f"{src_node} -> {target_nodes[dst_name]} [label=\"Transformation\"];")

    if logic_notes:
        note_text = "Logic Fixes:\\n" + "\\n".join([f"{i+1}. {n}" for i, n in enumerate(logic_notes[:4])])
        lines.append(f'Note1 [shape=plaintext, label="{dot_escape(note_text)}", fontsize=8];')
        lines.append("Note1 -> Source_DB [style=invis];")

    lines.append("}")
    return "\n".join(lines)


def extract_lineage(text: str, file_name: str) -> str:
    stages = parse_stages(text)
    if not stages:
        return "BEGIN_OUTPUT\nParsing error: Unable to extract deterministic lineage from input.\nEND_OUTPUT"

    for s in stages:
        detect_undefined_references(s)

    complexity_score, transformed_columns = compute_complexity(stages)
    if complexity_score > 150:
        return "Job too complex for single detailed diagram. Recommend stage-wise lineage generation."

    links = build_global_links(stages)
    chain_length = longest_chain([(a, b) for a, b, _ in links])
    rankdir = "TB" if (len(stages) > 10 or chain_length > 8) else "LR"

    graph_name = sanitize_id(os.path.splitext(file_name)[0], "lineage")
    high = build_high_level_dot(graph_name, stages, rankdir, links)
    detailed = build_detailed_dot(graph_name, stages, rankdir, links, transformed_columns)
    return "\n".join(
        [
            "BEGIN_OUTPUT",
            "",
            "HIGH LEVEL DOT",
            high,
            "DETAILED DOT",
            detailed,
            "END_OUTPUT",
        ]
    )


def extract_dot_sections(contract_output: str) -> Tuple[str, str]:
    high_match = re.search(r"(?s)HIGH LEVEL DOT\s*(digraph.*?)(?=\nDETAILED DOT\b)", contract_output)
    detailed_match = re.search(r"(?s)DETAILED DOT\s*(digraph.*?)(?=\nEND_OUTPUT\b)", contract_output)
    if not high_match or not detailed_match:
        raise ValueError("DOT sections missing from output contract.")
    return high_match.group(1).strip(), detailed_match.group(1).strip()


def render_pdf(dot_code: str, output_stem: str) -> None:
    Source(dot_code).render(output_stem, format="pdf", cleanup=True)


def process_file(file_path: str, file_name: str) -> None:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        pseudo = f.read()
    output = extract_lineage(pseudo, file_name)
    base = sanitize_id(os.path.splitext(file_name)[0], "lineage")
    output_path = os.path.join(OUTPUT_FOLDER, f"{base}.dot.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(output)

    if output.startswith("BEGIN_OUTPUT"):
        try:
            high_dot, detailed_dot = extract_dot_sections(output)

            high_dot_path = os.path.join(OUTPUT_FOLDER, f"{base}_high_level.dot")
            detailed_dot_path = os.path.join(OUTPUT_FOLDER, f"{base}_detailed.dot")
            hybrid_dot_path = os.path.join(OUTPUT_FOLDER, f"{base}_hybrid.dot")
            with open(high_dot_path, "w", encoding="utf-8") as f:
                f.write(high_dot)
            with open(detailed_dot_path, "w", encoding="utf-8") as f:
                f.write(detailed_dot)
            # Hybrid output uses the detailed architecture-style DOT as requested.
            with open(hybrid_dot_path, "w", encoding="utf-8") as f:
                f.write(detailed_dot)

            render_pdf(high_dot, os.path.join(OUTPUT_FOLDER, f"{base}_high_level"))
            render_pdf(detailed_dot, os.path.join(OUTPUT_FOLDER, f"{base}_detailed"))
            render_pdf(detailed_dot, os.path.join(OUTPUT_FOLDER, f"{base}_hybrid"))
        except ExecutableNotFound:
            print("[WARN] Graphviz 'dot' executable not found. Install Graphviz and add it to PATH to generate PDF.")
        except Exception as ex:
            print(f"[WARN] PDF render failed: {ex}")

    print(f"Processed {file_name} -> {output_path}")


def run_agent() -> None:
    os.makedirs(INPUT_FOLDER, exist_ok=True)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".txt")])
    if not files:
        print(f"No input files found in {INPUT_FOLDER}")
        return

    for file_name in files:
        file_path = os.path.join(INPUT_FOLDER, file_name)
        try:
            process_file(file_path, file_name)
        except Exception:
            fallback = "BEGIN_OUTPUT\nParsing error: Unable to extract deterministic lineage from input.\nEND_OUTPUT"
            base = sanitize_id(os.path.splitext(file_name)[0], "lineage")
            output_path = os.path.join(OUTPUT_FOLDER, f"{base}.dot.txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(fallback)
            print(f"Processed {file_name} -> {output_path}")


if __name__ == "__main__":
    run_agent()
