#!/usr/bin/env python3
"""
Generate sam_road labels for the Manila satellite image dataset.

Inputs (from the road-width-detector-pipeline thesis repo):
  0_img-dataset/{clearregroad,wideroad,occludedroad}/*.png     -- RGB tiles
  0_img-dataset/{clearregroad_label,..._label}/*.png           -- road masks
  5_width-estimation/data/shapefiles/gis_osm_roads_free_1.shp -- OSM roads

Outputs (in --output_dir, default ./manila/):
  images/         {stem}.png
  gt_graph/       {stem}__gt_graph.p   (sat2graph adjacency dict, (row,col) keys)
  processed/      keypoint_mask_{stem}.png
                  road_mask_{stem}.png
  data_split.json

Usage (run from the sam_road project root):
  python generate_manila_labels.py --thesis_repo "C:/path/to/road-width-detector-pipeline"
"""

import argparse
import json
import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from sklearn.cluster import DBSCAN

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ── Default path to the thesis repo ────────────────────────────────────────────
# Override with --thesis_repo if your layout differs.
_DEFAULT_THESIS_REPO = Path(
    r'C:\Users\Philip\Documents\My Folders\Acads\Thesis\Repos\road-width-detector-pipeline'
)

# ── Geographic constants (from width-estimator.ipynb) ──────────────────────────
LAT_SIZE  = 0.0013364    # degrees of latitude  per 512-pixel tile
LONG_SIZE = 0.00137216   # degrees of longitude per 512-pixel tile
LON_SHIFT = -0.000053    # systematic alignment correction (see notebook)
PATCH_SIZE = 512

# ── Label-generation constants ─────────────────────────────────────────────────
KEYPOINT_RADIUS = 13     # px  (≈ baseline 8 px × 1.67 for 0.6 m/px)
MERGE_NODE_DIST = 3.0    # px  nearby nodes get merged into one

# OSM fclass values that are pedestrian-only — skip for road-graph purposes
EXCLUDE_FCLASS = {'footway', 'path', 'pedestrian', 'cycleway', 'steps', 'bridleway'}

IMAGE_CATEGORIES = ['clearregroad', 'wideroad', 'occludedroad']


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--thesis_repo', default=str(_DEFAULT_THESIS_REPO),
                   help='Root of road-width-detector-pipeline repo')
    p.add_argument('--output_dir',  default='./manila',
                   help='Output directory inside sam_road (default: ./manila)')
    p.add_argument('--img_dir',     default=None,
                   help='Override path to 0_img-dataset/ (default: <thesis_repo>/0_img-dataset)')
    p.add_argument('--shapefile',   default=None,
                   help='Override path to OSM shapefile (default: <thesis_repo>/5_width-estimation/...)')
    p.add_argument('--val_frac',    type=float, default=0.10)
    p.add_argument('--test_frac',   type=float, default=0.10)
    args = p.parse_args()

    thesis = Path(args.thesis_repo)
    if args.img_dir is None:
        args.img_dir = str(thesis / '0_img-dataset')
    if args.shapefile is None:
        args.shapefile = str(thesis / '5_width-estimation' / 'data' / 'shapefiles' /
                             'gis_osm_roads_free_1.shp')
    return args


# ── Geographic projection ──────────────────────────────────────────────────────

def get_bbox(lat: float, lon: float):
    """
    Compute the geographic bounding box (minx, miny, maxx, maxy) for a
    Manila tile whose filename encodes (lat, lon) as the tile center.

    Applies the LON_SHIFT alignment correction from width-estimator.ipynb.
    """
    minx = lon - LONG_SIZE / 2 - LON_SHIFT   # LON_SHIFT is negative → adds +0.000053
    maxx = lon + LONG_SIZE / 2 - LON_SHIFT
    miny = lat - LAT_SIZE / 2
    maxy = lat + LAT_SIZE / 2
    return minx, miny, maxx, maxy


def geo_to_pixel(lons, lats, minx, miny, maxx, maxy):
    """
    Project geographic (lon, lat) arrays into image pixel (row, col).
    row=0 is the northern (top) edge; col=0 is the western (left) edge.
    """
    col = (np.asarray(lons, dtype=np.float64) - minx) / (maxx - minx) * PATCH_SIZE
    row = (maxy - np.asarray(lats, dtype=np.float64)) / (maxy - miny) * PATCH_SIZE
    return row, col


# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph(patch_roads, minx, miny, maxx, maxy):
    """
    Project OSM LineStrings into pixel space and produce a sat2graph adjacency dict.

    Algorithm:
      1. Clip each OSM geometry to the tile bbox.
      2. Project vertices to (row, col) pixel coordinates.
      3. Merge nodes within MERGE_NODE_DIST pixels (handles shared OSM endpoints
         that differ by <1 px due to float encoding).
      4. Deduplicate and add reverse edges (undirected graph).
      5. Convert to sat2graph {(row,col): [(row,col),...]} with integer coordinates.

    Returns:
      adj_dict  : sat2graph adjacency dict  (empty dict if no roads in tile)
      nodes     : np.ndarray [N, 2] float   merged pixel (row, col) positions
      edges     : np.ndarray [E, 2] int     directed edges (both directions stored)
    """
    EMPTY = {}, np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)

    bbox_geom = box(minx, miny, maxx, maxy)
    raw_nodes = []
    raw_edges = []

    for geom in patch_roads.geometry:
        if geom is None or geom.is_empty:
            continue
        clipped = geom.intersection(bbox_geom)
        if clipped.is_empty:
            continue

        lines = []
        gtype = clipped.geom_type
        if gtype == 'LineString':
            lines = [clipped]
        elif gtype == 'MultiLineString':
            lines = list(clipped.geoms)
        elif gtype == 'GeometryCollection':
            for g in clipped.geoms:
                if g.geom_type == 'LineString':
                    lines.append(g)
                elif g.geom_type == 'MultiLineString':
                    lines.extend(list(g.geoms))

        for line in lines:
            coords = np.array(line.coords)   # shape (M, 2) = (lon, lat)
            if len(coords) < 2:
                continue
            rows, cols = geo_to_pixel(coords[:, 0], coords[:, 1], minx, miny, maxx, maxy)
            # Clip to valid pixel range
            rows = np.clip(rows, 0.0, PATCH_SIZE - 1e-6)
            cols = np.clip(cols, 0.0, PATCH_SIZE - 1e-6)

            base = len(raw_nodes)
            for r, c in zip(rows, cols):
                raw_nodes.append((float(r), float(c)))
            for i in range(len(rows) - 1):
                raw_edges.append((base + i, base + i + 1))

    if not raw_nodes:
        return EMPTY

    nodes = np.array(raw_nodes, dtype=np.float32)

    # Merge nearby nodes (handles floating-point duplicates at OSM intersections)
    clustering = DBSCAN(eps=MERGE_NODE_DIST, min_samples=1).fit(nodes)
    labels = clustering.labels_
    n_clusters = int(labels.max()) + 1

    centers = np.zeros((n_clusters, 2), dtype=np.float32)
    counts  = np.zeros(n_clusters,     dtype=np.float32)
    for i, nd in enumerate(nodes):
        c = labels[i]
        centers[c] += nd
        counts[c]  += 1
    centers /= counts[:, np.newaxis]

    # Build directed edge set (undirected: store both directions)
    edge_set = set()
    for s, d in raw_edges:
        ns, nd = int(labels[s]), int(labels[d])
        if ns != nd:
            edge_set.add((ns, nd))
            edge_set.add((nd, ns))

    if not edge_set:
        return EMPTY

    merged_edges = np.array(sorted(edge_set), dtype=np.int32)

    # Build sat2graph adjacency dict {(row,col): [(row,col), ...]}
    adj = [[] for _ in range(n_clusters)]
    for s, d in merged_edges:
        adj[s].append(d)

    int_centers = [(round(float(r)), round(float(c))) for r, c in centers]
    adj_dict = {int_centers[i]: [int_centers[j] for j in nbrs]
                for i, nbrs in enumerate(adj)}

    return adj_dict, centers, merged_edges


# ── Keypoint mask ──────────────────────────────────────────────────────────────

def make_keypoint_mask(nodes, edges, size=PATCH_SIZE, radius=KEYPOINT_RADIUS):
    """
    Draw white filled circles at every node whose undirected degree ≠ 2.
    Degree ≠ 2 means the node is an intersection (≥3), dead end (1),
    or isolated (0) — not a straight-road midpoint.

    nodes : [N, 2] float (row, col)
    edges : [E, 2] int, directed — both (s,d) and (d,s) are stored,
            so degree[s] = out-edge count from s = undirected degree of s.
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    if len(nodes) == 0:
        return mask

    n = len(nodes)
    degree = np.zeros(n, dtype=np.int32)
    for s, _ in edges:
        degree[s] += 1   # out-edge count = undirected degree (edges are doubled)

    for i, (r, c) in enumerate(nodes):
        if degree[i] != 2:
            row_i = int(round(float(r)))
            col_i = int(round(float(c)))
            if 0 <= row_i < size and 0 <= col_i < size:
                cv2.circle(mask, (col_i, row_i), radius, 255, -1)
    return mask


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out     = Path(args.output_dir)
    img_dir = Path(args.img_dir)

    (out / 'images').mkdir(parents=True, exist_ok=True)
    (out / 'gt_graph').mkdir(parents=True, exist_ok=True)
    (out / 'processed').mkdir(parents=True, exist_ok=True)

    # Load OSM shapefile
    print('Loading OSM shapefile …')
    osm = gpd.read_file(args.shapefile)
    if osm.crs is not None and osm.crs.to_epsg() != 4326:
        osm = osm.to_crs(epsg=4326)
    if 'fclass' in osm.columns:
        before = len(osm)
        osm = osm[~osm['fclass'].isin(EXCLUDE_FCLASS)]
        print(f'OSM roads: {len(osm)} features ({before - len(osm)} pedestrian-only removed)')
    else:
        print(f'OSM roads: {len(osm)} features (no fclass column — using all)')

    # Collect image/label pairs
    all_pairs = []
    for cat in IMAGE_CATEGORIES:
        cat_dir   = img_dir / cat
        label_dir = img_dir / f'{cat}_label'
        for img_path in sorted(cat_dir.glob('*.png')):
            lbl = label_dir / img_path.name
            if lbl.exists():
                all_pairs.append((img_path, lbl))
    print(f'Found {len(all_pairs)} image+label pairs across {len(IMAGE_CATEGORIES)} categories')

    # Process each tile
    valid_stems = []
    skipped     = 0

    iterable = tqdm(all_pairs, desc='Tiles') if tqdm else all_pairs
    for img_path, label_path in iterable:
        stem  = img_path.stem
        parts = stem.split('_')             # e.g. ['105','','14.562318','121.006165']
        lat   = float(parts[-2])
        lon   = float(parts[-1])

        minx, miny, maxx, maxy = get_bbox(lat, lon)

        # Spatial query: roads whose bbox intersects this tile
        patch_roads = osm.cx[minx:maxx, miny:maxy]

        adj_dict, nodes, edges = build_graph(patch_roads, minx, miny, maxx, maxy)

        if not adj_dict:
            skipped += 1
            if tqdm is None:
                print(f'  skip (no roads): {stem}')
            continue

        # Save graph .p
        with open(out / 'gt_graph' / f'{stem}__gt_graph.p', 'wb') as f:
            pickle.dump(adj_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Save keypoint mask
        kp_mask = make_keypoint_mask(nodes, edges)
        cv2.imwrite(str(out / 'processed' / f'keypoint_mask_{stem}.png'), kp_mask)

        # Copy image (RGB)
        shutil.copy2(img_path, out / 'images' / f'{stem}.png')

        # Copy road mask (existing thesis label — white = road surface)
        road_mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(out / 'processed' / f'road_mask_{stem}.png'), road_mask)

        valid_stems.append(stem)
        if tqdm is None and len(valid_stems) % 50 == 0:
            print(f'  {len(valid_stems)}/{len(all_pairs)} done …')

    print(f'\nProcessed: {len(valid_stems)} valid, {skipped} skipped (no OSM roads in tile)')

    # Build train/val/test split
    stems_sorted = sorted(valid_stems)
    n       = len(stems_sorted)
    n_test  = max(1, round(n * args.test_frac))
    n_val   = max(1, round(n * args.val_frac))
    n_train = n - n_test - n_val

    split = {
        'train':      stems_sorted[:n_train],
        'validation': stems_sorted[n_train:n_train + n_val],
        'test':       stems_sorted[n_train + n_val:],
    }
    with open(out / 'data_split.json', 'w') as f:
        json.dump(split, f, indent=2)

    print(f'Split  — train: {len(split["train"])}, '
          f'val: {len(split["validation"])}, '
          f'test: {len(split["test"])}')
    print(f'Output → {out.resolve()}')


if __name__ == '__main__':
    main()
