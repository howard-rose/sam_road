"""
TOPO metric for Manila dataset.
Adapted from spacenet_metrics/topo/main.py.

Key differences from SpaceNet:
  - GT graphs live in ../manila/gt_graph/{stem}__gt_graph.p
  - Data split is ../manila/data_split.json
  - Pixels are 0.6 m/px instead of 1.0 m/px; xy2latlon scales by PIXEL_SCALE=0.6
    so the physical distance thresholds (r, interval, matching_threshold) remain
    in real metres and are directly comparable between the two Manila conditions.
"""

import sys
import math
import os
import pickle
import json
import argparse
from pathlib import Path

# Reuse graph.py / topo.py helpers from spacenet_metrics/topo/ without copying
_SPACENET_TOPO = Path(__file__).parent.parent.parent / "spacenet_metrics" / "topo"
sys.path.insert(0, str(_SPACENET_TOPO))
import graph as splfy
import topo as topo_lib

parser = argparse.ArgumentParser()
parser.add_argument('-savedir', type=str, required=True)
parser.add_argument('-matching_threshold', type=float, default=0.00010)
parser.add_argument('-interval',           type=float, default=0.00005)
args = parser.parse_args()
print(args)

# Manila pixels are 0.6 m/px; scale so distances are in real metres for thresholds
PIXEL_SCALE = 0.6

lat_top_left =  41.0
lon_top_left = -71.0
min_lat = 41.0
max_lon = -71.0

with open('../manila/data_split.json', 'r') as jf:
    tile_list = json.load(jf)['test']


def xy2latlon(x, y):
    lat = lat_top_left - x * PIXEL_SCALE / 111111.0
    lon = lon_top_left + (y * PIXEL_SCALE / 111111.0) / math.cos(math.radians(lat_top_left))
    return lat, lon


def create_graph(m):
    global min_lat, max_lon
    graph = splfy.RoadGraph()
    nid = 0
    idmap = {}

    for k, v in m.items():
        n1 = k
        lat1, lon1 = xy2latlon(n1[0], n1[1])
        if lat1 < min_lat:
            min_lat = lat1
        if lon1 > max_lon:
            max_lon = lon1

        if n1 in idmap:
            id1 = idmap[n1]
        else:
            id1 = nid
            idmap[n1] = nid
            nid += 1

        for n2 in v:
            lat2, lon2 = xy2latlon(n2[0], n2[1])
            if n2 in idmap:
                id2 = idmap[n2]
            else:
                id2 = nid
                idmap[n2] = nid
                nid += 1
            graph.addEdge(id1, lat1, lon1, id2, lat2, lon2)

    graph.ReverseDirectionLink()
    for node in graph.nodes.keys():
        graph.nodeScore[node] = 100
    for edge in graph.edges.keys():
        graph.edgeScore[edge] = 100
    return graph


for tile_idx in tile_list:
    graph_prop_path = '../%s/graph/%s.p' % (args.savedir, tile_idx)
    graph_gt_path   = '../manila/gt_graph/%s__gt_graph.p' % tile_idx
    output_path     = '../%s/results/topo/%s.txt' % (args.savedir, tile_idx)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if not os.path.exists(graph_prop_path):
        continue

    map1 = pickle.load(open(graph_gt_path,   'rb'))
    map2 = pickle.load(open(graph_prop_path, 'rb'))

    graph_gt   = create_graph(map1)
    graph_prop = create_graph(map2)

    print("load gt/prop graphs for", tile_idx)

    region = [
        min_lat - 300 * 1.0 / 111111.0,
        lon_top_left - 500 * 1.0 / 111111.0,
        lat_top_left + 300 * 1.0 / 111111.0,
        max_lon + 500 * 1.0 / 111111.0,
    ]
    graph_gt.region   = region
    graph_prop.region = region

    losm = topo_lib.TOPOGenerateStartingPoints(
        graph_gt, region=region, image="NULL", check=False, direction=False, metaData=None)

    lmap = topo_lib.TOPOGeneratePairs(
        graph_prop, graph_gt, losm,
        threshold=args.matching_threshold, region=region)

    # 150 m matching radius — same physical scale as SpaceNet because xy2latlon
    # accounts for PIXEL_SCALE
    r = 0.00150

    topoResult = topo_lib.TOPOWithPairs(
        graph_prop, graph_gt, lmap, losm,
        r=r, step=args.interval, threshold=args.matching_threshold,
        outputfile=output_path, one2oneMatching=True, metaData=None)

    print('=========', output_path, '==================')
    pickle.dump([losm, topoResult, region],
                open(output_path.replace('txt', 'topo.p'), 'wb'))
