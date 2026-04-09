#!/usr/bin/env python3
"""Headless Radare2 ACFG extractor.

This script mirrors the JSON structure produced by the Gencoding / Genius IDA
workflow, but uses Radare2 via r2pipe and emits 11-dimensional block features.

Expected output schema:
{
    "function_edges": [[src_indices], [dst_indices]],
    "acfg_list": [
        {
            "block_number": N,
            "block_edges": [[src_indices], [dst_indices]],
            "block_features": [[11-dim vector], ...]
        }
    ],
    "function_names": ["..."],
    "hash": "...",
    "function_number": N
}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import r2pipe
except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard for runtime use
    raise SystemExit(
        "r2pipe is required. Install it in the Python environment before running this extractor."
    ) from exc


CALL_TYPES = {"call", "icall", "rcall", "ucall"}
TRANSFER_TYPES = {"jmp", "ujmp", "cjmp", "switch", "branch", "goto"}
ARITH_TYPES = {
    "add",
    "sub",
    "mul",
    "div",
    "mod",
    "inc",
    "dec",
    "neg",
    "adc",
    "sbb",
    "lea",
    "fadd",
    "fsub",
    "fmul",
    "fdiv",
}
LOGIC_TYPES = {"and", "or", "xor", "not", "shl", "shr", "sal", "sar", "rol", "ror", "test", "bit", "logic"}
COMPARE_TYPES = {"cmp", "cmps", "scas", "testcmp", "ucomiss", "ucomisd", "fcmp", "tst"}
MOVE_TYPES = {"mov", "movs", "movzx", "movsx", "xchg", "push", "pop", "cmov", "load", "store", "set"}
TERMINATION_TYPES = {"ret", "retn", "retf", "iret", "hlt", "syscall", "sysenter", "sysret", "int", "trap", "ud2", "leave"}
DATA_TYPES = {"data", "invalid", "unk", "ascii", "string", "byte", "word", "dword", "qword", "dqword"}
DATA_MNEMONICS = {"db", "dw", "dd", "dq", "dt", "dc", "ascii", "string", "byte", "word", "dword", "qword"}

# Feature order is fixed to match the expected 11-dimensional block vector:
# [
#   0: calls,
#   1: transfer instructions,
#   2: arithmetic instructions,
#   3: logic instructions,
#   4: compare instructions,
#   5: move instructions,
#   6: termination instructions,
#   7: data declarations,
#   8: total instructions,
#   9: string/integer constants,
#   10: offspring (successor count)
# ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ACFGs from a PE binary using Radare2.")
    parser.add_argument("--binary_path", required=True, help="Path to the PE binary to analyze")
    parser.add_argument("--output_path", help="Output .json file path, or a directory to place the result in")
    return parser.parse_args()


def safe_json_command(r2: Any, command: str) -> Any:
    """Run a Radare2 JSON command and return decoded JSON or None."""
    try:
        if hasattr(r2, "cmdj"):
            result = r2.cmdj(command)
            if result is not None:
                return result
    except Exception:
        pass

    try:
        raw = r2.cmd(command)
    except Exception:
        return None

    if not raw:
        return None

    try:
        return json.loads(raw)
    except Exception:
        return None


def normalize_graph_output(result: Any) -> List[Dict[str, Any]]:
    if result is None:
        return []
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if isinstance(result, dict):
        if "blocks" in result or "nodes" in result or "edges" in result:
            return [result]
        if len(result) == 1:
            value = next(iter(result.values()))
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def get_function_name(function_record: Dict[str, Any]) -> str:
    name = function_record.get("name") or function_record.get("realname") or function_record.get("title")
    if isinstance(name, str) and name.strip():
        return name.strip()

    offset = function_record.get("offset")
    if offset is None:
        offset = function_record.get("addr")
    if offset is None:
        offset = function_record.get("from")

    if isinstance(offset, int):
        return f"sub_{offset:x}"
    if isinstance(offset, str) and offset:
        return offset
    return "sub_unknown"


def get_function_offset(function_record: Dict[str, Any]) -> Optional[int]:
    for key in ("addr", "offset", "from", "vaddr"):
        value = function_record.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value, 0)
            except Exception:
                continue
    return None


def get_block_offset(block: Dict[str, Any]) -> Optional[int]:
    for key in ("addr", "offset", "from", "start"):
        value = block.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value, 0)
            except Exception:
                continue
    return None


def get_block_size(block: Dict[str, Any]) -> int:
    for key in ("size", "length", "bytes"):
        value = block.get(key)
        if isinstance(value, int) and value > 0:
            return value
        if isinstance(value, str):
            try:
                parsed = int(value, 0)
            except Exception:
                continue
            if parsed > 0:
                return parsed
    return 0


def extract_blocks(function_graph: Dict[str, Any]) -> List[Dict[str, Any]]:
    blocks = function_graph.get("blocks")
    if isinstance(blocks, list):
        return [block for block in blocks if isinstance(block, dict)]

    # nodes = function_graph.get("nodes")
    # if isinstance(nodes, list):
    #     return [node for node in nodes if isinstance(node, dict)]

    return []


def resolve_graph_edges(graph: Dict[str, Any]) -> List[Tuple[Any, Any]]:
    edges: List[Tuple[Any, Any]] = []
    raw_edges = graph.get("imports")
    if not isinstance(raw_edges, list):
        return edges

    for edge in raw_edges:
        if edge is None or not isinstance(edge, str):
            print(f"[Warning] Unrecognized edge format: {edge}, skipping.")
            continue

        source = get_function_name(graph)
        target = edge.strip()

        if source is None or target is None:
            continue
        
        edges.append((source, target))
    return edges


def edge_node_label(node: Any, nodes_by_id: Dict[Any, Dict[str, Any]]) -> Optional[str]:
    if isinstance(node, str):
        return node
    if isinstance(node, int) and node in nodes_by_id:
        return get_function_name(nodes_by_id[node])
    if isinstance(node, dict):
        return get_function_name(node)
    return None


def collect_callgraph(r2: Any) -> Tuple[List[str], List[Tuple[str, str]]]:
    # agCj is the global call graph. agcj is function-scoped and only a fallback.
    graph_data = normalize_graph_output(safe_json_command(r2, "agCj"))
    if not graph_data:
        graph_data = normalize_graph_output(safe_json_command(r2, "agcj"))
    if not graph_data:
        return [], []

    node_names: List[str] = []
    seen_names = set()

    for graph in graph_data:
        # agCj commonly returns one record per function with {name, imports}.
        graph_name = get_function_name(graph)
        if graph_name and graph_name not in seen_names:
            seen_names.add(graph_name)
            node_names.append(graph_name)

    edges: List[Tuple[str, str]] = []
    seen_edges = set()
    for graph in graph_data:
        # Handle edge-list format (imports) when present.
        resolved_edges = resolve_graph_edges(graph)

        for source, target in resolved_edges:
            if source and source not in seen_names:
                seen_names.add(source)
                node_names.append(source)
            if target and target not in seen_names:
                seen_names.add(target)
                node_names.append(target)
            if source and target:
                edge = (source, target)
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                edges.append(edge)

    return node_names, edges


def is_call_instruction(op_type: str, mnemonic: str) -> bool:
    return op_type in CALL_TYPES or mnemonic in CALL_TYPES or mnemonic.startswith("call")


def is_control_flow_transfer(op_type: str, mnemonic: str) -> bool:
    if op_type in TRANSFER_TYPES:
        return True
    return mnemonic in TRANSFER_TYPES or (mnemonic.startswith("j") and mnemonic != "jmp")


def is_arithmetic_instruction(op_type: str, mnemonic: str) -> bool:
    return op_type in ARITH_TYPES or mnemonic in ARITH_TYPES


def is_logic_instruction(op_type: str, mnemonic: str) -> bool:
    return op_type in LOGIC_TYPES or mnemonic in LOGIC_TYPES


def is_compare_instruction(op_type: str, mnemonic: str) -> bool:
    return op_type in COMPARE_TYPES or mnemonic in COMPARE_TYPES or mnemonic.startswith("cmp")


def is_move_instruction(op_type: str, mnemonic: str) -> bool:
    return op_type in MOVE_TYPES or mnemonic in MOVE_TYPES or mnemonic.startswith("mov") or mnemonic.startswith("cmov")


def is_termination_instruction(op_type: str, mnemonic: str) -> bool:
    return op_type in TERMINATION_TYPES or mnemonic in TERMINATION_TYPES


def is_data_declaration(op_type: str, mnemonic: str) -> bool:
    if op_type in DATA_TYPES or mnemonic in DATA_MNEMONICS:
        return True
    return mnemonic.startswith("db") or mnemonic.startswith("dw") or mnemonic.startswith("dd") or mnemonic.startswith("dq")


def looks_like_constant(op: Dict[str, Any], op_type: str, mnemonic: str) -> bool:
    if is_call_instruction(op_type, mnemonic) or is_control_flow_transfer(op_type, mnemonic):
        return False

    for key in ("str", "string", "val", "ptr", "refptr", "imm"):
        value = op.get(key)
        if value is None:
            continue
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, int):
            return True
        if isinstance(value, list) and value:
            return True

    disasm = op.get("opcode") or op.get("disasm") or op.get("mnemonic") or ""
    if isinstance(disasm, str) and re.search(r"0x[0-9a-fA-F]+|\b\d+\b", disasm):
        return True

    operands = op.get("opex") or op.get("operands")
    if isinstance(operands, list):
        for operand in operands:
            if not isinstance(operand, dict):
                continue
            operand_type = str(operand.get("type") or operand.get("kind") or operand.get("class") or "").lower()
            if any(token in operand_type for token in ("imm", "int", "num", "str", "string", "const")):
                return True

    return False


def extract_instruction_entries(r2: Any, address: int) -> List[Dict[str, Any]]:
    result = safe_json_command(r2, f"aoj @ {address}")
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    if isinstance(result, dict):
        return [result]
    return []


def extract_instruction_at(r2: Any, address: int) -> Optional[Dict[str, Any]]:
    entries = extract_instruction_entries(r2, address)
    if entries:
        return entries[0]
    return None


def instruction_mnemonic(op: Dict[str, Any]) -> str:
    mnemonic = op.get("mnemonic") or op.get("opcode") or op.get("disasm") or ""
    if not isinstance(mnemonic, str):
        return ""
    mnemonic = mnemonic.strip().lower()
    if not mnemonic:
        return ""
    return mnemonic.split()[0]


def instruction_type(op: Dict[str, Any]) -> str:
    value = op.get("type") or op.get("ttype") or op.get("family") or ""
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def block_feature_vector(r2: Any, block: Dict[str, Any]) -> List[int]:
    start = get_block_offset(block)
    size = get_block_size(block)
    if start is None or size <= 0:
        return [0] * 11

    end = start + size
    # See feature order comment near constants above.
    counters = [0] * 11
    cursor = start
    visited_addrs = set()

    # Iterate through instructions in the block
    while cursor < end:
        if cursor in visited_addrs:
            break
        visited_addrs.add(cursor)

        op = extract_instruction_at(r2, cursor)
        if not op:
            # Treat undecodable bytes as data declarations to stay robust on
            # partially analyzed/misaligned blocks.
            counters[7] += 1
            cursor += 1
            continue

        op_type = instruction_type(op)
        mnemonic = instruction_mnemonic(op)

        counters[8] += 1    # Total number of instructions

        if is_call_instruction(op_type, mnemonic):
            counters[0] += 1
        if is_control_flow_transfer(op_type, mnemonic):
            counters[1] += 1
        if is_arithmetic_instruction(op_type, mnemonic):
            counters[2] += 1
        if is_logic_instruction(op_type, mnemonic):
            counters[3] += 1
        if is_compare_instruction(op_type, mnemonic):
            counters[4] += 1
        if is_move_instruction(op_type, mnemonic):
            counters[5] += 1
        if is_termination_instruction(op_type, mnemonic):
            counters[6] += 1
        if is_data_declaration(op_type, mnemonic):
            counters[7] += 1
        if looks_like_constant(op, op_type, mnemonic):
            counters[9] += 1

        size_value = op.get("size")
        if not isinstance(size_value, int) or size_value <= 0:
            cursor += 1
        else:
            cursor += size_value

    return counters


def build_block_graph(blocks: Sequence[Dict[str, Any]]) -> Tuple[List[List[int]], List[int]]:
    # Build a mapping from block offsets to their indices
    block_offsets: Dict[int, int] = {}
    for index, block in enumerate(blocks):
        offset = get_block_offset(block)
        if offset is not None:
            block_offsets[offset] = index

    edge_sources: List[int] = []
    edge_targets: List[int] = []
    offspring_sets: List[set] = [set() for _ in blocks]
    seen_edges: set = set()

    # For each block in the function
    for source_index, block in enumerate(blocks):
        # Prefer canonical CFG successors exposed by agfj.
        candidate_targets: List[int] = []

        # For each instruction in the block
        for op in block.get("ops", []):
            # Look for jump/fail targets in the instruction record
            for key in ("jump", "fail"):
                value = op.get(key)
                if isinstance(value, list):
                    candidate_targets.extend(item for item in value if isinstance(item, int))
                elif isinstance(value, int):
                    candidate_targets.append(value)
                elif isinstance(value, str):
                    try:
                        candidate_targets.append(int(value, 0))
                    except Exception:
                        continue

        for target_offset in candidate_targets:
            target_index = block_offsets.get(target_offset)
            if target_index is None:
                continue
            edge_key = (source_index, target_index)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            edge_sources.append(source_index)
            edge_targets.append(target_index)
            offspring_sets[source_index].add(target_index)      # Track unique successors for offspring count

    offspring = [len(values) for values in offspring_sets]
    return [edge_sources, edge_targets], offspring


def collect_local_functions(r2: Any, function_records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    locals_list: List[Dict[str, Any]] = []

    # Cycle through functions sorted by offset
    for function_record in sorted(function_records, key=lambda item: ((get_function_offset(item) or 0), get_function_name(item))):
        function_offset = get_function_offset(function_record)
        if function_offset is None:
            continue
        
        # Print progress for each function being processed, using the function name and offset for clarity.
        # print(f"Processing function {get_function_name(function_record)} at offset {function_offset:x}...")

        graph_data = normalize_graph_output(safe_json_command(r2, f"agfj @ {function_offset}"))
        if not graph_data:
            print(f"[Warning] No graph data for function at offset {function_offset:x}, skipping.")
            continue

        if graph_data[0].get("addr") != function_offset:
            print(f"[Warning] Graph data address {graph_data[0].get('addr')} does not match function offset {function_offset:x}, skipping.")
            continue

        graph = graph_data[0]
        blocks = extract_blocks(graph)
        if not blocks:
            print(f"[Warning] No blocks found for function at offset {function_offset:x}, skipping.")
            continue

        locals_list.append(
            {
                "name": get_function_name(function_record),
                "offset": function_offset,
                "graph": graph,
                "blocks": blocks,
            }
        )

    return locals_list


def build_function_index(local_functions: Sequence[Dict[str, Any]], callgraph_names: Sequence[str]) -> Tuple[List[str], Dict[str, int]]:
    ordered_names: List[str] = []
    seen = set()

    for item in local_functions:
        name = item["name"]
        if name not in seen:
            ordered_names.append(name)
            seen.add(name)

    for name in callgraph_names:
        if name not in seen:
            ordered_names.append(name)
            seen.add(name)

    name_to_index = {name: index for index, name in enumerate(ordered_names)}
    return ordered_names, name_to_index


def build_function_edges_from_callgraph(
    callgraph_edges: Sequence[Tuple[str, str]], function_index: Dict[str, int]
) -> List[List[int]]:
    edge_sources: List[int] = []
    edge_targets: List[int] = []
    seen = set()

    for source_name, target_name in callgraph_edges:
        source_index = function_index.get(source_name)
        target_index = function_index.get(target_name)
        if source_index is None or target_index is None:
            continue
        edge = (source_index, target_index)
        if edge in seen:
            continue
        seen.add(edge)
        edge_sources.append(source_index)
        edge_targets.append(target_index)

    return [edge_sources, edge_targets]


def build_callgraph_from_functions(local_functions: Sequence[Dict[str, Any]], function_index: Dict[str, int], r2: Any) -> List[List[int]]:
    edge_sources: List[int] = []
    edge_targets: List[int] = []
    seen = set()

    for caller in local_functions:
        caller_index = function_index.get(caller["name"])
        if caller_index is None:
            continue

        for block in caller["blocks"]:
            start = get_block_offset(block)
            size = get_block_size(block)
            if start is None or size <= 0:
                continue

            end = start + size
            cursor = start
            visited = set()
            while cursor < end and cursor not in visited:
                visited.add(cursor)
                op = extract_instruction_at(r2, cursor)
                if not op:
                    cursor += 1
                    continue

                op_type = instruction_type(op)
                mnemonic = instruction_mnemonic(op)
                # Fallback when agcj is unavailable: infer call edges by scanning
                # call instructions in local CFG blocks.
                if is_call_instruction(op_type, mnemonic):
                    target_name = None
                    for key in ("name", "symbol", "call", "refname", "str"):
                        value = op.get(key)
                        if isinstance(value, str) and value.strip():
                            target_name = value.strip()
                            break

                    if target_name is None:
                        opcode = op.get("opcode") or op.get("disasm") or ""
                        if isinstance(opcode, str):
                            match = re.search(r"([A-Za-z_.$?@][\w.$?@]*)$", opcode.strip())
                            if match:
                                target_name = match.group(1)

                    if target_name and target_name in function_index:
                        edge = (caller_index, function_index[target_name])
                        if edge not in seen:
                            seen.add(edge)
                            edge_sources.append(edge[0])
                            edge_targets.append(edge[1])

                size_value = op.get("size")
                cursor += size_value if isinstance(size_value, int) and size_value > 0 else 1

    return [edge_sources, edge_targets]


def parse_import_names(r2: Any) -> List[str]:
    imports = safe_json_command(r2, "iij")
    names: List[str] = []
    if not isinstance(imports, list):
        return names

    for item in imports:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("realname") or item.get("symbol") or item.get("libname")
        if isinstance(name, str) and name.strip() and name.strip() not in names:
            names.append(name.strip())
    return names


def build_acfg_list(r2: Any, local_functions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    acfg_list: List[Dict[str, Any]] = []

    for function_item in local_functions:
        blocks = function_item["blocks"]

        # This function returns the edges between blocks and the offspring count (number of unique successors) for each block
        # Format of block_edges: [[source_indices], [target_indices]]
        block_edges, offspring = build_block_graph(blocks)

        block_features: List[List[int]] = []

        # Cycle through each block in the function and build its feature vector
        for block in blocks:
            block_features.append(block_feature_vector(r2, block))  # Cycle through ops in the block to build the feature vector for this block.

        # Force the 11th feature to reflect graph-derived successor count.
        for index in range(min(len(block_features), len(offspring))):
            block_features[index][10] = offspring[index]

        acfg_list.append(
            {
                "block_number": len(blocks),            # Number of blocks in the function
                "block_edges": block_edges,             # Edges between blocks, represented as a pair of lists: [source_indices, target_indices]
                "block_features": block_features,       # List of feature vectors for each block
            }
        )

    return acfg_list


def sha256_of_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_output_path(binary_path: str, output_path: str) -> str:
    if os.path.isdir(output_path):
        base_name = os.path.basename(binary_path) or "output"
        return os.path.join(output_path, f"{base_name}.json")
    return output_path


class R2ACFGExtractor:
    """Programmatic extractor for building MalGraph ACFG payloads from binaries."""

    def __init__(self, binary_path: str, output_path: Optional[str] = None):
        self.binary_path = os.path.abspath(binary_path)
        self.output_path = output_path

    def extract(self) -> Dict[str, Any]:
        r2 = r2pipe.open(self.binary_path)
        try:
            # Full deep analysis before any extraction.
            r2.cmd("aaaa")

            function_records = safe_json_command(r2, "aflj") or []
            if not isinstance(function_records, list):
                function_records = []
                print("[Warning] No functions found in the binary.")

            # Local functions are those for which we can build CFG/ACFG.
            local_functions = collect_local_functions(r2, function_records)

            # Global callgraph edges and names collected from agCj.
            callgraph_names, callgraph_edges = collect_callgraph(r2)

            # Create a vocabulary of function names.
            import_names = parse_import_names(r2)

            # Combine callgraph names and import names to build a comprehensive function index.
            combined_callgraph_names: List[str] = []
            for name in callgraph_names + import_names:
                if name not in combined_callgraph_names:
                    combined_callgraph_names.append(name)

            # Build the dictionary mapping function names to indices.
            function_names, function_index = build_function_index(local_functions, combined_callgraph_names)

            # Prefer radare2 global callgraph, fallback to instruction-level recovery.
            if callgraph_edges:
                function_edges = build_function_edges_from_callgraph(callgraph_edges, function_index)
            else:
                function_edges = build_callgraph_from_functions(local_functions, function_index, r2)

            acfg_list = build_acfg_list(r2, local_functions)

            return {
                "function_edges": function_edges,
                "acfg_list": acfg_list,
                "function_names": function_names,
                "hash": sha256_of_file(self.binary_path),
                "function_number": len(function_names),
            }
        finally:
            try:
                r2.quit()
            except Exception:
                pass

    def run(self) -> str:
        payload = self.extract()

        if self.output_path:
            output_file = resolve_output_path(self.binary_path, os.path.abspath(self.output_path))
        else:
            output_file = f"{payload['hash']}.json"

        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True)

        return output_file


def main() -> int:
    args = parse_args()
    extractor = R2ACFGExtractor(binary_path=args.binary_path, output_path=args.output_path)
    output_file = extractor.run()
    print(f"Saved ACFG JSON to: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
