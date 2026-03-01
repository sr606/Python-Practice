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
    node_ids = {s.name: sanitize_id(s.name, f"stage_{i}") for i, s in enumerate(stages, 1)}
    source_entity_nodes: Dict[str, str] = {}
    for s in stages:
        for entity, _ in s.source_entities:
            source_entity_nodes[entity] = sanitize_id(f"src_{entity}", f"src_{len(source_entity_nodes)+1}")

    def stage_summary_label(stage: Stage) -> str:
        parts = [stage.name]
        if stage.join_blocks:
            parts.append(f"Joins: {len(stage.join_blocks)}")
        if stage.transformations:
            mapped = sum(1 for t in stage.transformations if t.target_col != "Unknown")
            if mapped:
                parts.append(f"Mapped columns: {mapped}")
        if stage.constraints:
            parts.append(f"Constraints: {len(stage.constraints)}")
        return "\\n".join(parts)

    lines = [
        f'digraph {sanitize_id(graph_name, "lineage_graph")} {{',
        f"rankdir={rankdir};",
        'fontsize=10;',
        'fontname="Arial";',
        'node [fontname="Arial", fontsize=9, style=filled, fillcolor="white"];',
        'edge [fontname="Arial", fontsize=8];',
        "",
        "subgraph cluster_source {",
        'label="STEP 1 - SOURCE SYSTEMS";',
        "fontsize=10;",
        "style=filled,rounded;",
        'color="#CFE2F3";',
        'fillcolor="#CFE2F3";',
    ]
    for entity, nid in source_entity_nodes.items():
        lines.append(f'{nid} [label="{dot_escape(entity)}", shape=cylinder];')
    for s in stages:
        if classify_stage_role(s) == "source":
            lines.append(f'{node_ids[s.name]} [label="{dot_escape(stage_summary_label(s))}", shape={node_shape(s)}];')
    lines.extend(
        [
            "}",
            "",
            "subgraph cluster_processing {",
            'label="STEP 2 - INTEGRATION AND RULES";',
            "fontsize=10;",
            "style=filled,rounded;",
            'color="#FFF2CC";',
            'fillcolor="#FFF2CC";',
        ]
    )
    for s in stages:
        if classify_stage_role(s) in {"processing", "lookup"}:
            lines.append(f'{node_ids[s.name]} [label="{dot_escape(stage_summary_label(s))}", shape={node_shape(s)}];')
            if s.constraints:
                for cidx, c in enumerate(s.constraints, 1):
                    dec_id = f"{node_ids[s.name]}_decision_{cidx}"
                    lines.append(f'{dec_id} [label="{dot_escape(c.expression)}", shape=diamond, fillcolor="white"];')
    lines.extend(
        [
            "}",
            "",
            "subgraph cluster_target {",
            'label="STEP 3 - TARGET OUTPUT";',
            "fontsize=10;",
            "style=filled,rounded;",
            'color="#E2D5F9";',
            'fillcolor="#E2D5F9";',
        ]
    )
    for s in stages:
        if classify_stage_role(s) == "target":
            lines.append(f'{node_ids[s.name]} [label="{dot_escape(stage_summary_label(s))}", shape={node_shape(s)}];')
    lines.append("}")

    for s in stages:
        sid = node_ids[s.name]
        for entity, rel in s.source_entities:
            src_node = source_entity_nodes.get(entity)
            if src_node:
                lines.append(f'{src_node} -> {sid} [label="{dot_escape(rel)}"];')

    for src, dst, label in links:
        src_id = node_ids.get(src, "Unknown")
        dst_id = node_ids.get(dst, "Unknown")
        if src_id == "Unknown":
            lines.append('Unknown [label="Unknown", shape=cylinder];')
            src_id = "Unknown"
        lines.append(f'{src_id} -> {dst_id} [label="Transformation: {dot_escape(label)}"];')

    for s in stages:
        sid = node_ids[s.name]
        for cidx, c in enumerate(s.constraints, 1):
            dec_id = f"{sid}_decision_{cidx}"
            yes_target, no_target = get_constraint_routes(s, c.name)
            yes_id = f"{sid}_hl_yes_{sanitize_id(yes_target, 'yes')}"
            no_id = f"{sid}_hl_no_{sanitize_id(no_target, 'no')}"
            lines.append(f'{yes_id} [label="{dot_escape(yes_target)}", shape=folder];')
            lines.append(f'{no_id} [label="{dot_escape(no_target)}", shape=folder];')
            lines.append(f'{sid} -> {dec_id} [label="Constraint"];')
            lines.append(f'{dec_id} -> {yes_id} [label="Yes"];')
            lines.append(f'{dec_id} -> {no_id} [label="No"];')

    lines.append("}")
    return "\n".join(lines)


def build_detailed_dot(
    graph_name: str,
    stages: List[Stage],
    rankdir: str,
    links: List[Tuple[str, str, str]],
    transformed_columns: int,
) -> str:
    node_ids = {s.name: sanitize_id(s.name, f"stage_{i}") for i, s in enumerate(stages, 1)}
    lines = [
        f'digraph {sanitize_id(graph_name + "_detailed", "lineage_detailed")} {{',
        f"rankdir={rankdir};",
        'fontsize=10;',
        'fontname="Arial";',
        'node [fontname="Arial"];',
        'edge [fontname="Arial"];',
    ]

    for s in stages:
        lines.append(f'{node_ids[s.name]} [label="{dot_escape(s.name)}", shape={node_shape(s)}];')

    for src, dst, dataset in links:
        src_id = node_ids.get(src, "Unknown")
        dst_id = node_ids.get(dst, "Unknown")
        if src_id == "Unknown":
            lines.append('Unknown [label="Unknown", shape=cylinder];')
        lines.append(f'{src_id} -> {dst_id} [label="Transformation: {dot_escape(dataset)}"];')

    for s in stages:
        sid = node_ids[s.name]
        if s.has_lookup:
            lines.append(f'{sid} -> {sid} [style=dashed, label="Lookup"];')
        for idx, jb in enumerate(s.join_blocks, 1):
            jnid = f"{sid}_join_{idx}"
            table_label = jb.table_name
            alias_label = jb.alias
            if table_label in {"Subquery", "RawJoin"}:
                table_label = "Subquery" if table_label == "Subquery" else "Join Block"
            if alias_label in {"Subquery", "RawJoin", "Unknown"}:
                alias_label = ""
            join_label = f"{jb.join_type} {table_label}".strip()
            if alias_label:
                join_label = f"{join_label} as {alias_label}"
            lines.append(f'{jnid} [label="{dot_escape(join_label)}", shape=box];')
            if jb.table_name in {"Subquery", "RawJoin"}:
                cond = jb.join_condition if jb.join_condition else jb.raw_join
            else:
                cond = jb.resolved_condition if jb.resolved_condition else jb.raw_join
            lines.append(f'{sid} -> {jnid} [label="Join: {dot_escape(cond)}"];')
        for cidx, c in enumerate(s.constraints, 1):
            cnid = f"{sid}_constraint_{cidx}"
            expr = c.expression
            if len(expr) > 140:
                truncated = expr[:137] + "..."
                lines.append(f'{cnid} [label="{dot_escape(truncated)}", shape=diamond];')
                lines.append(f'{sid} -> {cnid} [label="Constraint: {dot_escape(expr)}"];')
            else:
                lines.append(f'{cnid} [label="{dot_escape(expr)}", shape=diamond];')
                lines.append(f'{sid} -> {cnid} [label="Constraint"];')

            yes_target, no_target = get_constraint_routes(s, c.name)
            yes_id = f"{sid}_out_yes_{sanitize_id(yes_target, 'Unknown')}"
            no_id = f"{sid}_out_no_{sanitize_id(no_target, 'Unknown')}"
            lines.append(f'{yes_id} [label="Output: {dot_escape(yes_target)}", shape=box];')
            lines.append(f'{no_id} [label="Output: {dot_escape(no_target)}", shape=box];')
            lines.append(f'{cnid} -> {yes_id} [label="Yes"];')
            lines.append(f'{cnid} -> {no_id} [label="No"];')

            if c.issue:
                issue_id = f"{cnid}_issue"
                lines.append(f'{issue_id} [label="{dot_escape(c.issue)}", shape=diamond];')
                lines.append(f'{cnid} -> {issue_id} [label="Constraint"];')

        mapped_count = sum(1 for t in s.transformations if t.target_col != "Unknown")
        modified_count = sum(1 for t in s.transformations if t.is_modified)
        if mapped_count > 15:
            summary_id = f"{sid}_summary"
            lines.append(f'{summary_id} [label="{dot_escape(s.name)} ({mapped_count} columns mapped)", shape=box];')
            lines.append(f'{sid} -> {summary_id} [label="Transformation"];')
            unique_expr: Dict[str, List[str]] = {}
            for t in s.transformations:
                if t.is_modified:
                    unique_expr.setdefault(t.expression, []).append(t.target_col)
            for eidx, (expr, cols) in enumerate(unique_expr.items(), 1):
                expr_node = f"{sid}_expr_{eidx}"
                lines.append(f'{expr_node} [label="Applied to {len(cols)} columns", shape=box];')
                edge_label = expr if len(expr) <= 140 else (expr[:137] + "...")
                lines.append(f'{summary_id} -> {expr_node} [label="Transformation: {dot_escape(edge_label)}"];')
            direct_copy_count = mapped_count - modified_count
            if direct_copy_count > 0:
                copy_node = f"{sid}_direct_copy_summary"
                lines.append(f'{copy_node} [label="Direct copies: {direct_copy_count}", shape=box];')
                lines.append(f'{summary_id} -> {copy_node} [label="Transformation: Direct copy mappings"];')
        elif transformed_columns > 25:
            if modified_count > 0:
                lines.append(f'{sid} -> {sid} [label="Multiple column transformations ({modified_count})"];')
        else:
            unique_expr: Dict[str, List[str]] = {}
            unresolved_count = 0
            for t in s.transformations:
                if t.issue == "Unresolved Mapping":
                    unresolved_count += 1
                if not t.is_modified:
                    continue
                unique_expr.setdefault(t.expression, []).append(t.target_col)

            for eidx, (expr, cols) in enumerate(unique_expr.items(), 1):
                mid = f"{sid}_map_{eidx}"
                if len(cols) > 1:
                    target_label = f"Applied to {len(cols)} columns"
                elif len(s.transformations) <= 8:
                    target_label = f"{s.name}|{cols[0]}"
                else:
                    target_label = f"{s.name} ({len(s.transformations)} columns transformed)"
                shape = "record" if len(s.transformations) <= 8 else "box"
                if len(expr) > 140:
                    lines.append(f'{mid} [label="{dot_escape(target_label)}", shape={shape}];')
                    lines.append(f'{sid} -> {mid} [label="Transformation: {dot_escape(expr)}"];')
                else:
                    edge_label = expr if len(cols) == 1 else f"Applied to {len(cols)} columns"
                    lines.append(f'{mid} [label="{dot_escape(target_label)}", shape={shape}];')
                    lines.append(f'{sid} -> {mid} [label="Transformation: {dot_escape(edge_label)}"];')

            for t in s.transformations:
                if t.issue and t.issue != "Unresolved Mapping":
                    issue_id = f"{sid}_issue_{sanitize_id(t.target_col, 'unknown')}"
                    lines.append(f'{issue_id} [label="{dot_escape(t.issue)}", shape=diamond];')
                    lines.append(f'{sid} -> {issue_id} [label="Transformation"];')
            if unresolved_count:
                um_id = f"{sid}_unresolved"
                lines.append(f'{um_id} [label="Unresolved Mapping", shape=diamond];')
                lines.append(f'{sid} -> {um_id} [label="Transformation"];')

        defined = set(s.stage_vars_defined.keys())
        constraint_ids: Dict[str, str] = {}
        for cidx, c in enumerate(s.constraints, 1):
            constraint_ids[c.expression] = f"{sid}_constraint_{cidx}"
        for sv in s.stage_vars_used:
            if sv in defined:
                vid = f"{sid}_{sanitize_id(sv, 'sv')}"
                lines.append(f'{vid} [label="{dot_escape(sv)}", shape=ellipse];')
                lines.append(f'{vid} -> {sid} [label="Stage Variable"];')
                for c in s.constraints:
                    if re.search(rf"\b{re.escape(sv)}\b", c.expression):
                        lines.append(f'{vid} -> {constraint_ids[c.expression]} [label="Constraint"];')
            else:
                vid = f"{sid}_{sanitize_id(sv, 'sv')}_undef"
                lines.append(f'{vid} [label="Undefined Variable", shape=ellipse];')
                lines.append(f'{vid} -> {sid} [label="Stage Variable"];')
                for c in s.constraints:
                    if re.search(rf"\b{re.escape(sv)}\b", c.expression):
                        lines.append(f'{vid} -> {constraint_ids[c.expression]} [label="Constraint"];')

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
            with open(high_dot_path, "w", encoding="utf-8") as f:
                f.write(high_dot)
            with open(detailed_dot_path, "w", encoding="utf-8") as f:
                f.write(detailed_dot)

            render_pdf(high_dot, os.path.join(OUTPUT_FOLDER, f"{base}_high_level"))
            render_pdf(detailed_dot, os.path.join(OUTPUT_FOLDER, f"{base}_detailed"))
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
