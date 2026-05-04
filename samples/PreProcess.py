import argparse
import hashlib
import json
import os, sys
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from concurrent.futures import ProcessPoolExecutor
from BuildExternalVocab import ExternalVocabBuilder
from Vocabulary import Vocab
import torch
from torch_geometric.data import Data
import r2pipe

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, *args, **kwargs):
        return iterable


CALL_MNEMONICS = {
    "call", "callq", "jal", "jalr", "bl", "blx", "icall", "rcall", "ucall"
}

TRANSFER_PREFIXES = (
    "j", "b", "loop"
)

TRANSFER_EXACT = {
    "jmp", "jmpq", "jz", "jnz", "je", "jne", "jg", "jge", "jl", "jle",
    "ja", "jae", "jb", "jbe", "jo", "jno", "js", "jns", "jp", "jnp",
    "retf", "iret", "iretd", "iretq"
}

ARITHMETIC_MNEMONICS = {
    "add", "sub", "mul", "div", "mod", "inc", "dec", "neg", "adc", "sbb", "lea", "fadd", "fsub", "fmul", "fdiv"
}

LOGIC_MNEMONICS = {
    "and", "or", "xor", "not", "shl", "shr", "sal", "sar", "rol", "ror", "test", "bit", "logic"
}

COMPARE_MNEMONICS = {
    "cmp", "cmps", "cmpsb", "cmpsd", "cmpsq", "cmpsw", "testcmp", "ucomi", "ucomisd",
    "ucomiss", "comiss", "comisd"
}

MOVE_MNEMONICS = {
    "mov", "movzx", "movsx", "movsxd", "xchg", "lea", "push", "pop", "cmov",
    "movaps", "movups", "movd", "movq", "movsb", "movsw", "movsd", "movsq"
}

TERMINATION_MNEMONICS = {
    "ret", "retn", "retf", "iret", "hlt", "syscall", "sysenter", "sysret", "int", "trap", "ud2", "leave"
}

DATA_DECLARATION_MNEMONICS = {
    "db", "dw", "dd", "dq", "dt", "do", "dy", "resb", "resw", "resd", "resq"
}

NUMBER_RE = re.compile(r"(?<![\w.])(0x[0-9a-fA-F]+|\d+)(?![\w.])")
SYM_RE = re.compile(r"(sym\.imp\.[\w.$@?]+|sym\.[\w.$@?]+|fcn\.[\w.$@?]+)")

KNOWN_LIBRARY_MARKERS = (
    ".dll",
    "kernel32",
    "user32",
    "advapi32",
    "ntdll",
)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_name(name: str) -> str:
    if not name:
        return "unknown"
    n = name.strip()
    for pref in ("sym.imp.", "imp.", "sym.", "fcn."):
        if n.startswith(pref):
            n = n[len(pref):]
    n = n.replace("__imp_", "")
    dll_pos = n.lower().find(".dll_")
    if dll_pos != -1:
        n = n[dll_pos + len(".dll_"):]
    return n


"""Check if a function metadata likely corresponds to an external/imported function. Return True if it is likely external, False if it is likely local."""
def is_external_meta(
    r2: Any,
    meta: Dict[str, Any],
    imported_names: Optional[Set[str]] = None,
    imported_offsets: Optional[Set[int]] = None,
) -> bool:
    
    # 1. Check common naming patterns and metadata flags
    name = str(meta.get("name", "")).lower()
    ftype = str(meta.get("type", "")).lower()
    if name.startswith(("sym.imp.", "imp.")) or ftype in {"imp", "import"}:
        return True
    if meta.get("is-imported") or meta.get("is_imported"):
        return True

    # 2. Check for known library markers in the name
    if any(marker in name for marker in KNOWN_LIBRARY_MARKERS):
        return True

    if imported_names is not None:
        raw_name = str(meta.get("name", ""))
        if raw_name in imported_names or norm_name(raw_name) in imported_names:
            return True

    # 3. Check if the function's address/offset matches any known imported offsets
    if imported_offsets is not None:
        for key in ("addr", "offset", "vaddr", "paddr", "plt"):
            off = meta.get(key)
            if isinstance(off, int) and off in imported_offsets:
                return True
            
    # 4. Heuristic: if the function is very small (e.g., <=2 instructions) and it calls another function
    if isinstance(meta.get("ninstrs"), int) and meta["ninstrs"] <= 2:
        try:
            ops = extract_function_pdfj(r2, meta.get("addr", 0)).get("ops", [])
        except Exception:
            return False
        
        if not ops:
            return False
        
        op = ops[0]
        disasm = str(op.get("disasm", "")).lower()

        # Check for a direct jump/call to an imported function
        head = disasm.split(" ", 1)[0]
        if head in CALL_MNEMONICS or "jmp" in head:
            # Check reference function name
            if "sym.imp" in disasm or any(marker in disasm for marker in KNOWN_LIBRARY_MARKERS):
                return True
            
            # Check reference function address
            target = op.get("jump") or op.get("ptr")
            if isinstance(target, int) and imported_offsets is not None and target in imported_offsets:
                return True
    return False


"""Build lookups for imported function names and offsets from the list of imports, to be used later for classification and edge resolution."""
def build_import_lookup(imports: List[Dict[str, Any]]) -> Tuple[Set[str], Set[int]]:
    imported_names: Set[str] = set()
    imported_offsets: Set[int] = set()

    # Extract names and offsets from imports, normalizing names
    for imp in imports:
        for key in ("name", "libname"):
            raw = imp.get(key)
            if isinstance(raw, str) and raw:
                imported_names.add(raw)
                imported_names.add(norm_name(raw))

        for key in ("addr", "offset", "vaddr", "paddr", "plt"):
            off = imp.get(key)
            if isinstance(off, int):
                imported_offsets.add(off)

    return imported_names, imported_offsets


def cmdj_safe(r2: Any, command: str) -> Any:
    try:
        out = r2.cmdj(command)
        return out if out is not None else []
    except Exception:
        return []


def cmd_safe(r2: Any, command: str) -> str:
    try:
        return r2.cmd(command)
    except Exception:
        return ""


def classify_mnemonic(mnem: str) -> Dict[str, int]:
    m = mnem.lower()
    call = int(m in CALL_MNEMONICS or m.startswith("call"))
    transfer = int((m in TRANSFER_EXACT or m.startswith(TRANSFER_PREFIXES)) and not call and m not in TERMINATION_MNEMONICS)
    arithmetic = int(m in ARITHMETIC_MNEMONICS)
    logic = int(m in LOGIC_MNEMONICS)
    compare = int(m in COMPARE_MNEMONICS or m.startswith("cmp"))
    move = int(m in MOVE_MNEMONICS or m.startswith("mov") or m.startswith("cmov"))
    termination = int(m in TERMINATION_MNEMONICS or m.startswith("ret"))
    data_decl = int(m in DATA_DECLARATION_MNEMONICS)
    return {
        "call": call,
        "transfer": transfer,
        "arithmetic": arithmetic,
        "logic": logic,
        "compare": compare,
        "move": move,
        "termination": termination,
        "data_decl": data_decl,
    }


def extract_constants_count(op: Dict[str, Any], strings_addrs: Set[int]) -> int:
    count = 0
    for key in ("val", "ptr", "refptr"):
        value = op.get(key)
        if isinstance(value, int):
            count += 1
            if value in strings_addrs:
                count += 1

    opcode = str(op.get("opcode", ""))
    count += len(NUMBER_RE.findall(opcode))

    if any(tag in opcode for tag in ("str.", ".ascii", ".utf", "wide")):
        count += 1
    return count


"""Given a list of edges in a directed graph and the number of nodes, compute the number of descendants for each node. This is used to capture the influence of a block on the control flow of a function."""
def descendants_count(edges: List[Tuple[int, int]], num_nodes: int) -> List[int]:
    # Set recursion limit higher to handle deep graphs
    sys.setrecursionlimit(max(20000, num_nodes + 500))

    adj: Dict[int, List[int]] = defaultdict(list)
    for u, v in edges:
        adj[u].append(v)

    memo: Dict[int, Set[int]] = {}

    def dfs(u: int, visiting: Set[int]) -> Set[int]:
        if u in memo:
            return memo[u]
        if u in visiting:
            return set()
        visiting.add(u)
        reach: Set[int] = set()
        for v in adj.get(u, []):
            reach.add(v)
            reach.update(dfs(v, visiting))
        visiting.remove(u)
        memo[u] = reach
        return reach

    out = [0] * num_nodes
    for i in range(num_nodes):
        out[i] = len(dfs(i, set()))
    return out


"""Resolve the callee of a call instruction to determine the target function name and whether it's external. This is used for edge extraction when building the call graph, especially as a fallback when `agCj` does not provide call graph edges."""
def resolve_callee(
    r2: Any,
    op: Dict[str, Any],
    all_funcs_by_offset: Dict[int, Dict[str, Any]],
    local_name_by_offset: Dict[int, str],
) -> Optional[Tuple[str, bool]]:
    jump = op.get("jump")
    if isinstance(jump, int):
        if jump in local_name_by_offset:
            return local_name_by_offset[jump], False
        if jump in all_funcs_by_offset:
            meta = all_funcs_by_offset[jump]
            return norm_name(str(meta.get("name", f"sub_{jump:x}"))), is_external_meta(r2, meta)

    opcode = str(op.get("opcode", ""))
    for sym in SYM_RE.findall(opcode):
        n = norm_name(sym)
        if n:
            if sym.startswith("sym.imp."):
                return n, True
            return n, False
    return None


"""Collect the call graph edges and node names from r2's `agCj` output, which provides a JSON representation of the call graph. This is used to build the function call graph for the PE file, and serves as the primary source of call graph edges if available."""
def collect_callgraph(r2: Any) -> Tuple[List[str], List[Tuple[str, str]]]:
    raw = cmdj_safe(r2, "agCj")
    if not isinstance(raw, list):
        return [], []

    node_names: List[str] = []
    seen_names: Set[str] = set()
    edges: List[Tuple[str, str]] = []
    seen_edges: Set[Tuple[str, str]] = set()

    # Iterate over the raw call graph entries
    for entry in raw:
        if not isinstance(entry, dict):
            continue

        # Get the source function name
        src_name = norm_name(str(entry.get("name", "")))
        if src_name and src_name not in seen_names:
            seen_names.add(src_name)
            node_names.append(src_name)

        # Get the list of imported/called functions (targets) from the current entry
        imports = entry.get("imports")
        if not isinstance(imports, list):
            continue

        # Iterate over the targets and extract their names, while building edges from the source function to each target function
        for target in imports:
            dst_name = ""
            if isinstance(target, str):
                dst_name = norm_name(target)
            elif isinstance(target, dict):
                for key in ("name", "target", "to", "dst"):
                    value = target.get(key)
                    if isinstance(value, str) and value:
                        dst_name = norm_name(value)
                        break

            if dst_name and dst_name not in seen_names:
                seen_names.add(dst_name)
                node_names.append(dst_name)

            # Build the edge from the source function to the target function
            if src_name and dst_name:
                edge = (src_name, dst_name)
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                edges.append(edge)

    return node_names, edges


"""Extract a function's PDF and its basic blocks, using `pdfj` for instructions and `agfj`/`afbj` for block structure. Returns a dictionary with keys 'ops' and 'blocks' containing the respective lists of instructions and blocks."""
def extract_function_pdfj(r2: Any, offset: int) -> Dict[str, Any]:
    if offset is None:
        return {}

    pdf = cmdj_safe(r2, f"pdfj @ {offset}")
    if not isinstance(pdf, dict):
        pdf = {}

    # `pdfj` returns instruction-level entries in `ops`; keep them as instructions.
    ops = list(pdf.get("ops") or [])
    if ops:
        pdf["ops"] = ops

    # Source real basic blocks from `agfj` and store them separately.
    blocks: List[Dict[str, Any]] = []
    agfj = cmdj_safe(r2, f"agfj @ {offset}")
    if isinstance(agfj, list):
        graph_entry: Optional[Dict[str, Any]] = None

        # Try to find the graph entry that matches the function offset (the one I am looking for)
        for entry in agfj:
            if not isinstance(entry, dict):
                continue
            if entry.get("addr") == offset:
                graph_entry = entry
                break
            if graph_entry is None:
                graph_entry = entry

        # Extract blocks from the identified graph entry
        if isinstance(graph_entry, dict):
            candidate = graph_entry.get("blocks")
            if isinstance(candidate, list):
                blocks = [b for b in candidate if isinstance(b, dict)]

    # Fallback for cases where agfj is unavailable or empty.
    if not blocks:
        afbj = cmdj_safe(r2, f"afbj @ {offset}")
        if isinstance(afbj, dict):
            for key in ("bbs", "basic_blocks", "blocks", "ops"):
                candidate = afbj.get(key)
                if isinstance(candidate, list):
                    blocks = [b for b in candidate if isinstance(b, dict)]
                    if blocks:
                        break
        elif isinstance(afbj, list):
            blocks = [b for b in afbj if isinstance(b, dict)]

    if blocks:
        pdf["blocks"] = blocks

    return pdf


def op_offset(op: Dict[str, Any]) -> Optional[int]:
    for key in ("offset", "addr", "address", "ea"):
        value = op.get(key)
        if isinstance(value, int):
            return value
    return None


def block_offset(block: Dict[str, Any]) -> Optional[int]:
    for key in ("offset", "addr", "address", "ea"):
        value = block.get(key)
        if isinstance(value, int):
            return value
    return None


def normalize_blocks_from_ops(ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_ops = [op for op in ops if isinstance(op, dict) and op_offset(op) is not None]
    if not normalized_ops:
        return []

    normalized_ops.sort(key=lambda op: op_offset(op) or 0)
    op_offsets = [op_offset(op) or 0 for op in normalized_ops]
    offset_set = set(op_offsets)
    leaders = {op_offsets[0]}

    for idx, op in enumerate(normalized_ops):
        for field in ("jump", "fail"):
            target = op.get(field)
            if isinstance(target, int) and target in offset_set:
                leaders.add(target)

        opcode = str(op.get("opcode", "")).strip().lower()
        mnem = op.get("mnemonic") or (opcode.split(" ", 1)[0] if opcode else "")
        cls = classify_mnemonic(str(mnem))
        if cls["transfer"] or cls["termination"]:
            if idx + 1 < len(normalized_ops):
                leaders.add(op_offsets[idx + 1])

    blocks: List[Dict[str, Any]] = []
    current_ops: List[Dict[str, Any]] = []
    for op in normalized_ops:
        off = op_offset(op)
        if off is None:
            continue
        if current_ops and off in leaders:
            blocks.append({"addr": op_offset(current_ops[0]), "ops": current_ops})
            current_ops = []
        current_ops.append(op)

    if current_ops:
        blocks.append({"addr": op_offset(current_ops[0]), "ops": current_ops})

    return blocks


def attach_ops_to_blocks(blocks: List[Dict[str, Any]], ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized_blocks = [block for block in blocks if isinstance(block, dict) and block_offset(block) is not None]
    normalized_ops = [op for op in ops if isinstance(op, dict) and op_offset(op) is not None]
    if not normalized_blocks or not normalized_ops:
        return normalized_blocks

    normalized_blocks.sort(key=lambda block: block_offset(block) or 0)
    normalized_ops.sort(key=lambda op: op_offset(op) or 0)

    for idx, block in enumerate(normalized_blocks):
        start = block_offset(block) or 0
        end = None
        size = block.get("size")
        if isinstance(size, int) and size > 0:
            end = start + size
        elif idx + 1 < len(normalized_blocks):
            next_start = block_offset(normalized_blocks[idx + 1])
            if isinstance(next_start, int):
                end = next_start

        block_ops = []
        for op in normalized_ops:
            off = op_offset(op)
            if off is None or off < start:
                continue
            if end is not None and off >= end:
                continue
            block_ops.append(op)
        if block_ops:
            block["ops"] = block_ops

    return normalized_blocks


"""Build an ACFG for a function based on its PDF and basic blocks, extracting features for each block and building edges based on jump/fail references. Returns a dictionary containing the ACFG structure and a list of blocks with attached operations."""
def build_acfg_for_function(
    pdf: Dict[str, Any],
    strings_addrs: Set[int],
    block_budget: int,
    imported_offsets: Set[int],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:

    # Get blocks and ops from the PDF
    blocks = list(pdf.get("blocks") or [])
    ops = list(pdf.get("ops") or [])
    # if blocks and ops:
    #     blocks = attach_ops_to_blocks(blocks, ops)
    # elif not blocks and ops:
    #     blocks = normalize_blocks_from_ops(ops)
    # if not blocks or block_budget <= 0:
    #     return None, blocks

    blocks = sorted(blocks, key=lambda b: int(b.get("addr", 0)))
    if len(blocks) > block_budget:
        blocks = blocks[:block_budget]

    # Build a mapping of block offsets to their indices for edge construction
    id_by_offset: Dict[int, int] = {}
    for idx, block in enumerate(blocks):
        off = block.get("addr")
        if isinstance(off, int):
            id_by_offset[off] = idx

    # Build edges based on jump/fail references in blocks
    edges: List[Tuple[int, int]] = []
    for block in blocks:
        src_off = block.get("addr")
        if src_off not in id_by_offset:
            continue
        src = id_by_offset[src_off]

        # Check jump and fail fields for potential edges
        for field in ("jump", "fail"):
            dst_off = block.get(field)
            if isinstance(dst_off, int) and dst_off in id_by_offset:
                edges.append((src, id_by_offset[dst_off]))


    # Deduplicate edges
    edges = list(set(edges))

    # Compute descendant counts for each block to use as a feature, which can help capture the block's influence on the control flow of the function.
    offspring = descendants_count(edges, len(blocks))

    # Extract features for each block based on its operations
    features: List[List[int]] = []
    api_calls = 0
    for idx, block in enumerate(blocks):
        call = transfer = arithmetic = logic = compare = move = termination = data_decl = total_inst = consts = 0
        ops = list(block.get("ops") or [])
        if not ops:
            ninstr = block.get("ninstrs")
            if isinstance(ninstr, int):
                total_inst = ninstr

        for op in ops:
            # Fallback: treat misaligned/undecodable operations as data declarations.
            if not op or op.get("opcode") == "invalid":
                data_decl += 1
                total_inst += 1
                continue

            # Classify based on mnemonic and opcode patterns
            opcode = str(op.get("opcode", "")).strip()
            mnem = op.get("mnemonic") or (opcode.split(" ", 1)[0] if opcode else "")
            cls = classify_mnemonic(mnem)
            call += cls["call"]
            transfer += cls["transfer"]
            arithmetic += cls["arithmetic"]
            logic += cls["logic"]
            compare += cls["compare"]
            move += cls["move"]
            termination += cls["termination"]
            data_decl += cls["data_decl"]
            consts += extract_constants_count(op, strings_addrs)
            total_inst += 1

            # Additional API call feature
            disasm = str(op.get("disasm", "")).lower()
            optype = str(op.get("type", "")).lower()

            if optype == "call":
                # Case 1: direct import call
                if "sym.imp." in disasm:
                    api_calls += 1
                else:
                    # Case 2: address-based detection
                    target = op.get("jump") or op.get("ptr")
                    if isinstance(target, int) and target in imported_offsets:
                        api_calls += 1

        features.append([
            call,
            transfer,
            arithmetic,
            logic,
            compare,
            move,
            termination,
            data_decl,
            total_inst,
            consts,
            offspring[idx],
            # api_calls,
        ])

    # Build the edge index for the ACFG, which represents the control flow between blocks based on jump/fail references.
    edge_index = [[], []]
    if edges:
        edge_index = [
            [u for u, _ in edges],
            [v for _, v in edges],
        ]

    return {
        "block_number": len(blocks),
        "block_edges": edge_index,
        "block_features": features,
    }, blocks


"""Main preprocessing function for a single PE file, extracting ACFGs, call graph edges, and function names."""
def preprocess_pe(
    pe_path: str,
    max_fcg_nodes: int = 3000,
    max_total_blocks: int = 10000,
) -> Dict[str, Any]:
    
    # Open the PE file with r2pipe and perform initial analysis
    r2 = r2pipe.open(pe_path, flags=["-2"])
    try:
        cmd_safe(r2, "e anal.jmptbl=true")
        cmd_safe(r2, "aaa")

        # Collect all functions, imports, and strings to build lookups for later use in classification and edge resolution
        all_funcs = list(cmdj_safe(r2, "aflj") or [])
        imports = list(cmdj_safe(r2, "iij") or [])
        strings = list(cmdj_safe(r2, "izzj") or [])
        imported_names, imported_offsets = build_import_lookup(imports)

        # Collect addresses of all strings
        strings_addrs: Set[int] = set()
        for s in strings:
            for key in ("vaddr", "paddr"):
                v = s.get(key)
                if isinstance(v, int):
                    strings_addrs.add(v)

        # Build a mapping of function offsets to their metadata
        all_funcs_by_offset: Dict[int, Dict[str, Any]] = {}
        for f in all_funcs:
            off = f.get("addr")
            if isinstance(off, int):
                all_funcs_by_offset[off] = f

        # Identify local functions by filtering out those that are likely external/imported based on metadata and naming patterns
        local_funcs = [
            f
            for f in all_funcs
            if not is_external_meta(
                r2,
                f,
                imported_names=imported_names,
                imported_offsets=imported_offsets,
            )
        ]
        local_funcs = sorted(local_funcs, key=lambda x: int(x.get("addr", 0)))

        # Limit the number of local functions to max_fcg_nodes
        if len(local_funcs) > max_fcg_nodes:
            local_funcs = local_funcs[:max_fcg_nodes]

        # Build a mapping of local function offsets to normalized names, and a list of local function names
        local_names: List[str] = []
        local_name_by_offset: Dict[int, str] = {}
        for f in local_funcs:
            off = int(f["addr"])
            name = norm_name(str(f.get("name", f"sub_{off:x}")))
            local_names.append(name)
            local_name_by_offset[off] = name

        acfg_list: List[Dict[str, Any]] = []
        fcg_edges_by_name: List[Tuple[str, str]] = []
        external_used: Set[str] = set(norm_name(str(imp.get("name", ""))) for imp in imports if imp.get("name"))

        total_blocks = 0
        kept_local_names: List[str] = []

        # Iterate over local functions, extract their ACFGs, and build edges based on call instructions, while respecting the block budget and collecting used external functions for later inclusion in the function list.
        for f in local_funcs:
            if total_blocks >= max_total_blocks:
                break

            # Get the function offset and corresponding name
            off = int(f["addr"])
            src_name = local_name_by_offset.get(off)
            if not src_name:
                continue

            # Extract the function's PDF and basic blocks, and build its ACFG
            pdf = extract_function_pdfj(r2, off)
            acfg, blocks = build_acfg_for_function(
                pdf=pdf,
                strings_addrs=strings_addrs,
                block_budget=max_total_blocks - total_blocks,
                imported_offsets=imported_offsets,
            )
            if acfg is None:
                continue

            acfg_list.append(acfg)
            kept_local_names.append(src_name)
            total_blocks += acfg["block_number"]

            # Fallback edge extraction
            # For each block and its operations, look for call instructions to resolve potential edges in the call graph. This is used as fallback in case `agCj` does not provide call graph edges.
            for block in blocks:
                for op in list(block.get("ops") or []):
                    # Extract the opcode and check if it's a call instruction
                    op_type = str(op.get("type", "")).lower()
                    opcode = str(op.get("opcode", "")).lower()
                    if "call" not in op_type and not any(m in opcode.split(" ", 1)[0] for m in CALL_MNEMONICS):
                        continue

                    # Resolve the callee of the call instruction to determine the target function name and whether it's external
                    resolved = resolve_callee(
                        op=op,
                        all_funcs_by_offset=all_funcs_by_offset,
                        local_name_by_offset=local_name_by_offset,
                        r2=r2,
                    )

                    if not resolved:
                        continue
                    dst_name, dst_is_external = resolved
                    if not dst_name:
                        continue
                    if dst_is_external or dst_name not in local_name_by_offset.values():
                        external_used.add(dst_name)

                    # If the callee is a local function add an edge from the caller to the callee
                    fcg_edges_by_name.append((src_name, dst_name))

        local_names = kept_local_names
        local_name_set = set(local_names)

        # Collect the call graph edges and node names from r2's `agCj` output
        callgraph_names, callgraph_edges = collect_callgraph(r2)

        external_names = [
            name for name in callgraph_names
            if name and name not in local_name_set
        ]

        # Add new external names from the call graph
        for name in sorted(external_used):
            if not name or name in local_name_set or name in external_names:
                continue
            external_names.append(name)

        # Limit the total number of functions (local + external) to max_fcg_nodes, prioritizing local functions and then filling remaining slots with external functions from the call graph.
        free_slots = max(0, max_fcg_nodes - len(local_names))
        function_names = local_names + external_names[:free_slots]
        name_to_idx = {name: i for i, name in enumerate(function_names)}

        # Build the final set of edges for the function call graph
        edge_set: Set[Tuple[int, int]] = set()
        for src, dst in callgraph_edges:
            if src in name_to_idx and dst in name_to_idx:
                edge_set.add((name_to_idx[src], name_to_idx[dst]))

        # If there are no valid edges using `agCj`, fall back to using edges collected from call instructions in the ACFGs
        if not edge_set:
            for src, dst in fcg_edges_by_name:
                if src in name_to_idx and dst in name_to_idx:
                    edge_set.add((name_to_idx[src], name_to_idx[dst]))

        function_edges = [[], []]
        if edge_set:
            ordered = sorted(edge_set)
            function_edges = [
                [u for u, _ in ordered],
                [v for _, v in ordered],
            ]

        return {
            "function_edges": function_edges,
            "acfg_list": acfg_list,
            "function_names": function_names,
            "hash": sha256_file(pe_path),
            "md5": md5_file(pe_path),
            "function_number": len(function_names),
        }
    finally:
        try:
            r2.quit()
        except Exception:
            pass


def process_one_pe_task(task: Tuple[str, int, int]) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    pe_path, max_fcg_nodes, max_total_blocks = task
    try:
        item = preprocess_pe(
            pe_path=pe_path,
            max_fcg_nodes=max_fcg_nodes,
            max_total_blocks=max_total_blocks,
        )
        return pe_path, item, None
    except Exception as exc:
        return pe_path, None, str(exc)


def per_pe_output_path(input_root: str, pe_path: str, output_root: str, single_file_input: bool) -> str:
    pe_abs = os.path.abspath(pe_path)
    rel_path = os.path.basename(pe_abs) if single_file_input else os.path.relpath(pe_abs, input_root)
    return os.path.join(output_root, os.path.splitext(rel_path)[0] + ".json")


def json_item_to_pyg_data(item: Dict[str, Any], label: Optional[int], vocab: Vocab) -> Any:
    """Helper placeholder for converting one JSON object into a PyG Data object."""
    try:
        import torch
        from torch_geometric.data import Data
    except Exception as exc:
        raise RuntimeError("torch and torch_geometric are required for PyG conversion") from exc

    acfg_data_list = []
    for one_acfg in item.get("acfg_list", []):
        x = torch.tensor(one_acfg["block_features"], dtype=torch.float)
        edge_index = torch.tensor(one_acfg["block_edges"], dtype=torch.long)
        acfg_data_list.append(Data(x=x, edge_index=edge_index))

    external_function_names = item.get("function_names", [])[len(acfg_data_list):]
    if vocab is not None:
        external_list = [vocab[name] for name in external_function_names]
    else:
        external_list = external_function_names

    data_kwargs = {
        "hash": item.get("hash"),
        "local_acfgs": acfg_data_list,
        "external_list": external_list,
        "function_edges": item.get("function_edges", [[], []]),
        "targets": label,
    }

    return Data(**data_kwargs)


def is_probable_pe_file(path: str) -> bool:
    """Fast PE check via DOS/PE signatures (MZ + PE\\0\\0)."""
    try:
        with open(path, "rb") as f:
            dos = f.read(64)
            if len(dos) < 64 or dos[:2] != b"MZ":
                return False
            pe_off = int.from_bytes(dos[0x3C:0x40], byteorder="little", signed=False)
            if pe_off <= 0:
                return False
            f.seek(pe_off)
            return f.read(4) == b"PE\x00\x00"
    except Exception:
        return False


def collect_pe_files(path: str, recursive: bool = True) -> List[str]:
    if os.path.isfile(path):
        return [path] if is_probable_pe_file(path) else []

    out: List[str] = []
    if recursive:
        for root, _, files in os.walk(path):
            for name in files:
                full = os.path.join(root, name)
                if is_probable_pe_file(full):
                    out.append(full)
    else:
        for name in os.listdir(path):
            full = os.path.join(path, name)
            if os.path.isfile(full) and is_probable_pe_file(full):
                out.append(full)
    return sorted(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Radare2-based MalGraph preprocessing for Windows PE files")
    parser.add_argument("--input", required=True, help="PE file path or directory containing PE files")
    # parser.add_argument("--output", default="sample.jsonl", help="Output jsonl file path")
    parser.add_argument("--max-fcg-nodes", type=int, default=3000, help="Maximum number of FCG nodes")
    parser.add_argument("--max-cfg-blocks", type=int, default=10000, help="Maximum number of basic blocks across all local CFGs")
    parser.add_argument("--workers", type=int, default=0, help="Number of worker processes (0 = auto)")
    parser.add_argument("--no-recursive", action="store_true", help="If input is a directory, only process files in that directory")
    # parser.add_argument("--per-pe-json", action="store_true", help="Write one JSON file per PE instead of a single combined JSONL file")
    parser.add_argument("--output-dir", default="", help="Directory for per-PE JSON files (used with --per-pe-json)")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output files")
    return parser.parse_args()


def main() -> None:
    # Parse command-line arguments
    args = parse_args()
    input_abs = os.path.abspath(args.input)
    input_is_dir = os.path.isdir(input_abs)
    input_root = input_abs if input_is_dir else os.path.dirname(input_abs)
    pe_files = collect_pe_files(args.input, recursive=not args.no_recursive)

    # If no PE files are found, create empty output and exit gracefully
    if not pe_files:
        print(f"[WARN] no PE files found in input: {args.input}")

        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)
        print(f"Saved 0 per-PE JSON files to {out_dir}")

        return

    # Initialize output variables
    out_dir = ""
    written_count = 0

    # Create output directory for per-PE JSON files
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    # If overwrite is not allowed, check for each pe_file whether the corresponding output already exists and skip processing if it does
    if not args.overwrite:
        original_count = len(pe_files)
        remaining_pe_files = []
        for pe in pe_files:
            out_path = per_pe_output_path(input_root, pe, out_dir, not input_is_dir)
            if os.path.exists(out_path):
                continue
            remaining_pe_files.append(pe)

        pe_files = remaining_pe_files

        print(f"[INFO] {original_count - len(pe_files)} files were skipped due to existing outputs.")


    if args.workers is not None and args.workers < 0:
        raise ValueError("--workers must be >= 0")

    # Determine number of worker processes to use
    auto_workers = max(1, min(len(pe_files), os.cpu_count() or 1))
    worker_count = args.workers if args.workers and args.workers > 0 else auto_workers

    # Process PE files in parallel using a process pool
    tasks = [(pe, args.max_fcg_nodes, args.max_cfg_blocks) for pe in pe_files]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for pe, item, error in tqdm(
            executor.map(process_one_pe_task, tasks),
            total=len(tasks),
            desc="Processing PE files",
        ):
            if error is not None or item is None:
                print(f"[WARN] failed to process {pe}: {error}")
                continue

            # Write output either as individual JSON files per PE
            out_path = per_pe_output_path(input_root, pe, out_dir, not input_is_dir)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(item, f, ensure_ascii=False)
            written_count += 1

    print(f"Saved {written_count} per-PE JSON files to {out_dir}\n")

    # Build the Vocab object - Decomment the following if you want to build the vocabulary
    # ExternalVocabBuilder(input_path=out_dir, output_file="train_external_function_name_vocab.jsonl").run()

    # Load the Vocab object
    vocabulary = Vocab(freq_file="train_external_function_name_vocab.jsonl")

    # Fix external function indexing in the JSON files and convert them to PyG objects
    def process_json_to_pyg(json_path, vocabulary):
        """Process a single JSON file and save as PyG data."""
        
        item = json.load(open(json_path, "r", encoding="utf-8")).get("function_names", [])
        if not item:
            print(f"[WARN] failed to fix function indexing for {json_path}. The function names list is empty!")
            return None

        # Check the parent folder to determine the label
        # Blacklist -> 0, Whitelist -> 1, otherwise skip PyG conversion for this file
        label = json_path.split(os.sep)[-2].lower()
        if label == "blacklist":
            label = 0
        elif label == "whitelist":
            label = 1
        else:
            print(f"[WARN] unable to determine label for {json_path}, skipping PyG conversion")
            return None

        torch_data = json_item_to_pyg_data(
            item=item,
            label=label,
            vocab=vocabulary,
        )

        # Save the PyG Data object as a .pt file with the same name as the JSON but with .pt extension and in another folder named "pyg_data" keep the same relative structure to the input folder
        pyg_out_dir = os.path.join(os.path.dirname(out_dir), "pyg_data")
        os.makedirs(pyg_out_dir, exist_ok=True)
        relative_path = os.path.relpath(json_path, out_dir)
        pyg_out_path = os.path.join(pyg_out_dir, os.path.splitext(relative_path)[0] + ".pt")
        os.makedirs(os.path.dirname(pyg_out_path), exist_ok=True)
        torch.save(torch_data, pyg_out_path)

        return pyg_out_path
    
    converted_files_count = 0
    for root, _, files in os.walk(out_dir):
        json_files = [os.path.join(root, name) for name in files if name.endswith(".json")]

        if not json_files:
            continue

        for json_file in tqdm(json_files, desc=f"Converting JSON to PyG"):
            pyg_path = process_json_to_pyg(json_file, vocabulary)
            if pyg_path:
                converted_files_count += 1

                
    print(f"Converted {converted_files_count} JSON files to PyG Data objects in {os.path.join(os.path.dirname(out_dir), 'pyg_data')}\n")

if __name__ == "__main__":
    main()