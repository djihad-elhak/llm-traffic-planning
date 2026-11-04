"""TNTP transportation network loader (Sioux Falls and similar benchmarks)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Link:
    init: int
    term: int
    capacity: float
    length: float
    free_flow_time: float
    b: float
    power: float
    toll: float = 0.0


@dataclass
class Network:
    name: str
    num_zones: int
    num_nodes: int
    links: list[Link]
    # demand[(origin, destination)] = trips
    demand: dict[tuple[int, int], float] = field(default_factory=dict)

    @property
    def zones(self) -> list[int]:
        return list(range(1, self.num_zones + 1))

    def link_index(self) -> dict[tuple[int, int], int]:
        return {(l.init, l.term): i for i, l in enumerate(self.links)}


def _read_metadata(lines: list[str]) -> tuple[dict[str, str], int]:
    meta: dict[str, str] = {}
    for i, line in enumerate(lines):
        if line.strip().startswith("<END OF METADATA>"):
            return meta, i + 1
        m = re.match(r"<([^>]+)>\s*(.*)", line.strip())
        if m:
            meta[m.group(1).strip().upper()] = m.group(2).strip()
    raise ValueError("No <END OF METADATA> marker found")


def load_network(net_path: str | Path, trips_path: str | Path, name: str) -> Network:
    net_lines = Path(net_path).read_text().splitlines()
    meta, start = _read_metadata(net_lines)
    num_zones = int(meta["NUMBER OF ZONES"])
    num_nodes = int(meta["NUMBER OF NODES"])

    links: list[Link] = []
    for line in net_lines[start:]:
        parts = line.strip().rstrip(";").split()
        if len(parts) < 7 or not parts[0].isdigit():
            continue  # skip blank lines and the "~ init_node ..." header row
        init, term = int(parts[0]), int(parts[1])
        links.append(
            Link(
                init=init,
                term=term,
                capacity=float(parts[2]),
                length=float(parts[3]),
                free_flow_time=float(parts[4]),
                b=float(parts[5]),
                power=float(parts[6]),
                toll=float(parts[7]) if len(parts) > 7 else 0.0,
            )
        )
    if len(links) != int(meta["NUMBER OF LINKS"]):
        raise ValueError(f"Expected {meta['NUMBER OF LINKS']} links, parsed {len(links)}")

    trip_lines = Path(trips_path).read_text().splitlines()
    _, tstart = _read_metadata(trip_lines)
    demand: dict[tuple[int, int], float] = {}
    origin = None
    for line in trip_lines[tstart:]:
        line = line.strip()
        if not line:
            continue
        m = re.match(r"Origin\s+(\d+)", line)
        if m:
            origin = int(m.group(1))
            continue
        for dest, value in re.findall(r"(\d+)\s*:\s*([\d.]+)\s*;", line):
            if float(value) > 0:
                demand[(origin, int(dest))] = float(value)

    return Network(name=name, num_zones=num_zones, num_nodes=num_nodes, links=links, demand=demand)
