import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from graphviz import Source
from graphviz.backend import ExecutableNotFound
try:
    import sqlparse
    from sqlparse import sql as sql_ast
    from sqlparse import tokens as sql_tokens
except Exception:
    sqlparse = None
    sql_ast = None
    sql_tokens = None


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
class SQLColumnLineage:
    target_column: str
    expression: str
    source_columns: List[str] = field(default_factory=list)
    issue: str = ""


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
    sql_column_lineage: List[SQLColumnLineage] = field(default_factory=list)
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


def _split_top_level_csv(text: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(text):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            parts.append(text[start:i].strip())
            start = i + 1
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_select_segment_with_sqlparse(sql_text: str) -> str:
    if not sqlparse:
        return ""
    try:
        statements = sqlparse.parse(sql_text)
        if not statements:
            return ""
        stmt = statements[0]
        in_select = False
        chunks: List[str] = []
        for tok in stmt.tokens:
            if tok.ttype in sql_tokens.Whitespace:
                continue
            if not in_select:
                if tok.ttype in sql_tokens.DML and tok.normalized == "SELECT":
                    in_select = True
                continue
            if tok.ttype in sql_tokens.Keyword and tok.normalized == "FROM":
                break
            chunks.append(str(tok))
        return "".join(chunks).strip()
    except Exception:
        return ""


def parse_sql_column_lineage(sql_text: str, alias_map: Dict[str, str]) -> List[SQLColumnLineage]:
    if not sql_text:
        return []
    select_segment = _extract_select_segment_with_sqlparse(sql_text)
    if not select_segment:
        # Fallback to deterministic top-level extraction.
        m = re.search(r"(?is)\bselect\b(.*?)(?=\bfrom\b)", sql_text)
        select_segment = m.group(1).strip() if m else ""
    if not select_segment:
        return []

    items = _split_top_level_csv(select_segment)
    results: List[SQLColumnLineage] = []

    for item in items:
        item_clean = re.sub(r"(?is)^/\*.*?\*/\s*", "", item)
        item_clean = re.sub(r"\s+", " ", item_clean).strip()
        if not item_clean:
            continue

        alias_match = re.match(r"(?is)^(.*?)(?:\s+AS\s+([A-Za-z_]\w*)|\s+([A-Za-z_]\w*))\s*$", item_clean)
        target_col = "Unknown"
        expr = item_clean
        if alias_match:
            expr = alias_match.group(1).strip()
            target_col = (alias_match.group(2) or alias_match.group(3) or "Unknown").strip()
        else:
            # If no alias, keep raw target when expression is a direct column.
            direct = re.match(r"(?is)^([A-Za-z_]\w*)\.([A-Za-z_]\w*)$", item_clean)
            if direct:
                target_col = direct.group(2)
            else:
                bare = re.match(r"(?is)^([A-Za-z_]\w*)$", item_clean)
                if bare:
                    target_col = bare.group(1)

        source_cols: List[str] = []
        for am, cm in re.findall(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b", expr):
            table_or_alias = alias_map.get(am.lower(), am)
            source_cols.append(f"{table_or_alias}.{cm}")

        issue = ""
        if target_col == "Unknown":
            issue = "Unresolved Mapping"
        elif not source_cols and re.search(r"\b[A-Za-z_]\w*\.[A-Za-z_]\w*\b", expr) is not None:
            issue = "Undefined Reference"

        results.append(
            SQLColumnLineage(
                target_column=target_col,
                expression=expr,
                source_columns=sorted(set(source_cols)),
                issue=issue,
            )
        )

    return results


def parse_sql_ast(sql_text: str) -> Tuple[Dict[str, str], List[JoinBlock], List[Tuple[str, str]], List[SQLColumnLineage]]:
    # sqlparse is token-based AST. Use it when present; deterministic fallback otherwise.
    normalized_sql = sql_text
    if sqlparse:
        try:
            normalized_sql = sqlparse.format(sql_text, strip_comments=True, reindent=False, keyword_case="upper")
        except Exception:
            normalized_sql = sql_text

    alias_map = build_alias_map(normalized_sql)
    join_blocks = parse_structured_joins(normalized_sql, alias_map)
    source_entities = parse_source_entities(normalized_sql, join_blocks)
    col_lineage = parse_sql_column_lineage(normalized_sql, alias_map)
    return alias_map, join_blocks, source_entities, col_lineage


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
        alias_map, join_blocks, source_entities, sql_col_lineage = parse_sql_ast(sql_text)

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
            source_entities=source_entities,
            sql_column_lineage=sql_col_lineage,
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


def business_rule_summaries(stage: Stage) -> List[str]:
    rules: List[str] = []
    for t in stage.transformations:
        expr_l = t.expression.lower()
        if "vehicle_sk=-99" in expr_l and "lkvehicledim" in expr_l:
            text = "If Vehicle Key is missing, derive it from Vehicle Dimension"
            if text not in rules:
                rules.append(text)
        if "supp_sk=-99" in expr_l and "'unknown'" in expr_l:
            text = "Unmapped suppliers are classified as 'Unknown'"
            if text not in rules:
                rules.append(text)
        if "supp_sk=-97" in expr_l and ("duplicates" in expr_l or "'duplicates'" in expr_l):
            text = "Duplicate suppliers are classified as 'Duplicates'"
            if text not in rules:
                rules.append(text)
    if stage.constraints:
        text = "Records failing supplier validation are routed to Exception output"
        if text not in rules:
            rules.append(text)
    return rules


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
    downstream: Dict[str, List[str]] = {}
    for src, dst, _ in links:
        downstream.setdefault(src, []).append(dst)

    transformer = next(
        (s for s in stages if "transformer" in s.stage_type.lower() or s.name.lower().startswith("tfm_")),
        next((s for s in stages if classify_stage_role(s) == "processing"), stages[0]),
    )
    source = next(
        (s for s in stages if classify_stage_role(s) == "source" and s.name != "HF_SMR_VEHICLE_DIM"),
        next((s for s in stages if classify_stage_role(s) == "source"), transformer),
    )
    lookups = [s for s in stages if s.name == "HF_SMR_VEHICLE_DIM" or s.has_lookup]

    input_to_stage: Dict[str, str] = {}
    for s in stages:
        for _, _, link_name in s.inputs:
            if link_name:
                input_to_stage[link_name] = s.name

    yes_stage = None
    no_stage = None
    if transformer.constraints:
        yes_out, no_out = get_constraint_routes(transformer, transformer.constraints[0].name)
        yes_stage = input_to_stage.get(yes_out)
        no_stage = input_to_stage.get(no_out)
    children = downstream.get(transformer.name, [])
    if not yes_stage and children:
        yes_stage = children[0]
    if not no_stage and len(children) > 1:
        no_stage = children[1]

    # Map Yes to success and No to exception in business view.
    success_stage = yes_stage
    exception_stage = no_stage
    ys = stage_by_name.get(yes_stage or "")
    ns = stage_by_name.get(no_stage or "")
    if ys and ("exception" in ys.name.lower() or "seqfile" in ys.stage_type.lower()):
        success_stage, exception_stage = no_stage, yes_stage
    elif ns and not ("exception" in (ys.name.lower() if ys else "") or "seqfile" in (ys.stage_type.lower() if ys else "")) and (
        "exception" in ns.name.lower() or "seqfile" in ns.stage_type.lower()
    ):
        success_stage, exception_stage = yes_stage, no_stage

    rules = business_rule_summaries(transformer)
    mapped_cols = sum(1 for t in transformer.transformations if t.target_col != "Unknown")
    business_rules_count = sum(1 for t in transformer.transformations if t.is_modified)
    join_count = max(len(source.source_entities) - 1, len([1 for _, rel in source.source_entities if rel == "JOIN"]))
    source_count = len(source.source_entities)

    source_label = f"{source.name}\\n(Enriched via {join_count} joins)"
    transformer_label = "\\n".join([transformer.name] + [f"- {r}" for r in rules[:2]])

    lines = [
        f'digraph {sanitize_id(graph_name, "lineage_graph")} {{',
        "rankdir=LR;",
        "nodesep=0.5;",
        "ranksep=1.0;",
        'fontname="Arial";',
        'fontsize=10;',
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
    for i, lk in enumerate(lookups, 1):
        lines.append(f'Lookup_{i} [label="{dot_escape(lk.name)}", fillcolor="#E1F5FE", shape=cylinder];')
    lines.extend(
        [
            "}",
            "",
            "subgraph cluster_2 {",
            'label="Business Logic Stage";',
            "style=solid;",
            f'Transformer [label="{dot_escape(transformer_label)}", fillcolor="#FFF9C4"];',
            'Decision [shape=diamond, label="Supplier Valid?", fillcolor="#FADBD8"];',
            "}",
            "",
            "subgraph cluster_3 {",
            'label="Target Layer";',
            "style=dashed;",
            "color=green;",
        ]
    )

    target_ids: Dict[str, str] = {}
    idx = 1
    for name in [success_stage, exception_stage]:
        if name and name not in target_ids and name in stage_by_name:
            st = stage_by_name[name]
            nid = f"Target_{idx}"
            idx += 1
            target_ids[name] = nid
            is_exc = "exception" in st.name.lower() or "seqfile" in st.stage_type.lower()
            shape = "note" if is_exc else "cylinder"
            fill = "#FFCCBC" if is_exc else "#C8E6C9"
            lines.append(f'{nid} [label="{dot_escape(st.name)}", fillcolor="{fill}", shape={shape}];')

    final_stage = None
    if success_stage:
        for nxt in downstream.get(success_stage, []):
            if nxt in stage_by_name and nxt not in target_ids:
                final_stage = nxt
                break
    if final_stage:
        target_ids[final_stage] = f"Target_{idx}"
        lines.append(f'Target_{idx} [label="{dot_escape(final_stage)}", fillcolor="#C8E6C9", shape=cylinder];')
    lines.append("}")
    lines.append("")

    lines.append(f'Source_DB -> Transformer [label="{mapped_cols} columns"];')
    if lookups:
        lines.append('Lookup_1 -> Transformer [label="If Vehicle Key is missing, derive it from Vehicle Dimension", style=dotted];')
    lines.append('Transformer -> Decision [label="Supplier validation"];')
    if success_stage and success_stage in target_ids:
        lines.append(f'Decision -> {target_ids[success_stage]} [label="Yes", color=blue];')
    if exception_stage and exception_stage in target_ids:
        lines.append(f'Decision -> {target_ids[exception_stage]} [label="No", color=red, fontcolor=red];')
    if success_stage and final_stage and success_stage in target_ids and final_stage in target_ids:
        lines.append(f'{target_ids[success_stage]} -> {target_ids[final_stage]} [label="Transformation"];')

    summary = (
        "Job Summary:\\n"
        f"- {source_count} Source Tables\\n"
        f"- {join_count} Joins\\n"
        f"- {mapped_cols} Columns Processed\\n"
        f"- {business_rules_count} Business Rules Applied\\n"
        f"- {len(transformer.constraints)} Decision Split"
    )
    lines.append(f'Note1 [shape=plaintext, label="{dot_escape(summary)}", fontsize=8];')
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
    # Detailed view intentionally keeps the same architecture-focused style as high-level.
    # This avoids technical clutter while retaining deterministic lineage flow.
    return build_high_level_dot(graph_name + "_detailed", stages, rankdir, links)


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
